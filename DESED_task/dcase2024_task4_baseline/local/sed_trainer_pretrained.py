import os
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sed_scores_eval
import torch
import torch.nn.functional as F
import torchmetrics
from codecarbon import OfflineEmissionsTracker
from desed_task.data_augm import MixupAugmentor
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_psds_from_scores,
)
from desed_task.utils.postprocess import ClassWiseMedianFilter
from desed_task.utils.scaler import TorchScaler

# Import experiment directory management
from local.experiment_dir import ExecutionMode, ExperimentConfig, ExperimentDirManager

# Import SEBBs wrapper layer for type-safe interface
from local.sebbs_wrapper import SEBBsPredictor, SEBBsTuner

# Keep direct import for utilities
from sebbs.sebbs.utils import sed_scores_from_sebbs
from sed_scores_eval.base_modules.scores import create_score_dataframe, validate_score_dataframe

# データ不足の対策
from torch.utils.data.dataloader import DataLoader, default_collate
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

import wandb

from .classes_dict import (
    classes_labels_desed,
    classes_labels_maestro_real,
    classes_labels_maestro_real_eval,
)
from .utils import batched_decode_preds, log_sedeval_metrics

PROJECT_NAME = "SED-pl-noise"

# Type Aliases for complex data structures
# Training batch structure (without filenames) - 5 elements
TrainBatchType: TypeAlias = tuple[
    torch.Tensor,  # audio: (batch_size, audio_length) waveform
    torch.Tensor,  # labels: (batch_size, time_steps, num_classes) strong labels
    torch.Tensor,  # padded_indxs: (batch_size,) padding mask
    torch.Tensor,  # embeddings: (batch_size, embedding_dim, time_steps) BEATs features
    torch.Tensor,  # valid_class_mask: (batch_size, num_classes) class validity mask
]

# Evaluation batch structure (with filenames) - 6 elements
EvalBatchType: TypeAlias = tuple[
    torch.Tensor,  # audio: (batch_size, audio_length) waveform
    torch.Tensor,  # labels: (batch_size, time_steps, num_classes) strong labels
    torch.Tensor,  # padded_indxs: (batch_size,) padding mask
    list[str],  # filenames: audio file paths
    torch.Tensor,  # embeddings: (batch_size, embedding_dim, time_steps) BEATs features
    torch.Tensor,  # valid_class_mask: (batch_size, num_classes) class validity mask
]

# Student/teacher model prediction outputs
# Each PredictionPair contains strong (frame-level) and weak (clip-level) predictions
PredictionPair: TypeAlias = tuple[
    torch.Tensor,  # strong_preds: (batch_size, time_steps, num_classes) frame-level predictions
    torch.Tensor,  # weak_preds: (batch_size, num_classes) clip-level predictions
]

# Score dataframes for PSDS evaluation (keyed by audio clip ID)
# Maps clip IDs (e.g., "audio1-0-1000") to score DataFrames
# Each DataFrame contains columns: onset, offset, class1_score, class2_score, ...
ScoreDataFrameDict: TypeAlias = dict[str, pd.DataFrame]

# PSDS evaluation results structure
# Contains PSDS scores (float) and optional per-class breakdowns (dict[str, float])
# Example: {"psds": 0.653, "per_class": {"Blender": 0.75, "Cat": 0.68}}
PSDSResult: TypeAlias = dict[str, float | dict[str, float]]


class _NoneSafeIterator:
    """DataLoaderから返されるNoneバッチを内部でスキップ."""

    def __init__(self, dataloader_iter):
        self.dataloader_iter = dataloader_iter

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            batch = next(self.dataloader_iter)
            if batch is not None:
                return batch
            print("Skipping a batch that was None internally.")


class SafeCollate:
    """データセットから返される None 値をフィルタリングする collate_fn.

    フィルタリング後にバッチが空になった場合は None を返す.
    """

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        if not batch:
            # バッチが空ならNoneを返す（このNoneは_NoneSafeIteratorで捕捉される>）
            return None

        return default_collate(batch)


class SafeDataLoader(DataLoader):
    """Noneを返す可能性があるバッチを自動的にスキップするDataLoader.

    PyTorch Lightningなどのフレームワークで安全に使用できる。
    """

    def __init__(self, *args, **kwargs):
        # ユーザーがcollate_fnを指定していない場合のみ、SafeCollateを設定
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = SafeCollate()
        super().__init__(*args, **kwargs)

    def __iter__(self):
        dataloader_iter = super().__iter__()
        return _NoneSafeIterator(dataloader_iter)


class SEDTask4(pl.LightningModule):
    """PyTorch Lightning module for Sound Event Detection with teacher-student semi-supervised learning.

    This module implements the DCASE 2024 Task 4 baseline with a refactored architecture that uses
    helper methods to consolidate common operations across training, validation, and test steps.
    The refactoring eliminates code duplication while maintaining full backward compatibility with
    existing configurations and checkpoints.

    Architecture:
        - Teacher-student semi-supervised learning with EMA (Exponential Moving Average) updates
        - CRNN-based models (CNN + RNN) with optional pre-trained BEATs embeddings
        - Helper methods for embedding processing, prediction generation, loss computation, and metric updates
        - Support for strong (frame-level) and weak (clip-level) event detection
        - Optional cSEBBs post-processing for robust event segmentation

    Helper Methods:
        - _process_embeddings(): Conditional BEATs embedding extraction with model eval mode enforcement
        - _generate_predictions(): Unified student/teacher model inference from audio
        - _compute_step_loss(): Supervised loss computation with optional masking
        - _update_metrics(): Metric accumulation for torchmetrics instances

    Args:
        hparams: dict, the hyperparameter dictionary for the current experiment
        encoder: ManyHotEncoder object, encodes and decodes labels between one-hot and string representations
        sed_student: torch.nn.Module, the student model to be trained (teacher model created via EMA)
        opt: torch.optim.Optimizer object, the optimizer to be used for training
        train_data: torch.utils.data.Dataset subclass object, the training data to be used
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used
        test_data: torch.utils.data.Dataset subclass object, the test data to be used
        train_sampler: torch.utils.data.Sampler subclass object, the sampler for the training dataloader
        scheduler: BaseScheduler subclass object, learning rate scheduler (e.g., ramp-up during training)
        fast_dev_run: bool, whether to launch a development run with only one batch per set for testing

    """

    def __init__(
        self,
        hparams: dict,
        encoder: Any,  # ManyHotEncoder from desed_task.utils.encoder
        sed_student: Any,  # CRNN model
        pretrained_model: Any | None,  # BEATs or other pretrained model
        opt: torch.optim.Optimizer | None = None,
        train_data: Any | None = None,  # Dataset
        valid_data: Any | None = None,  # Dataset
        test_data: Any | None = None,  # Dataset
        train_sampler: Any | None = None,  # Sampler
        scheduler: Any | None = None,  # LR scheduler
        fast_dev_run: bool = False,
        evaluation: bool = False,
        sed_teacher: Any | None = None,  # CRNN model
        _test_state_dict: dict | None = None,  # Test state dict for mode detection
    ):
        super(SEDTask4, self).__init__()
        self.hparams.update(hparams)
        self._test_state_dict = _test_state_dict  # Store for mode detection

        # Set execution mode attributes BEFORE wandb initialization
        # These are needed by ExperimentDirManager.detect_execution_mode()
        self.fast_dev_run: bool = fast_dev_run
        self.evaluation: bool = evaluation

        if self.hparams["wandb"]["use_wandb"]:
            self._init_wandb_project()
        else:
            # wandbを使わない場合はNoneに設定
            self._wandb_checkpoint_dir: str | None = None

        self.encoder: Any = encoder  # ManyHotEncoder
        self.sed_student: Any = sed_student  # CRNN
        self.median_filter: ClassWiseMedianFilter = ClassWiseMedianFilter(
            self.hparams["net"]["median_filter"],
        )
        self.mixup_augmentor: MixupAugmentor = MixupAugmentor(self.hparams["training"])

        # CMT
        self.cmt_enabled: bool = self.hparams.get("cmt", {}).get("enabled", False)
        self.phi_clip: float = float(self.hparams.get("cmt", {}).get("phi_clip", 0.5))
        self.phi_frame: float = float(self.hparams.get("cmt", {}).get("phi_frame", 0.5))
        self.phi_neg: float = float(self.hparams.get("cmt", {}).get("phi_neg", 0.3))
        self.phi_pos: float = float(self.hparams.get("cmt", {}).get("phi_pos", 0.7))

        self.pos_neg_scale: bool = self.hparams.get("cmt", {}).get("pos_neg_scale", False)
        self.cmt_warmup_epochs: int = int(self.hparams.get("cmt", {}).get("warmup_epochs", 50))
        self.use_neg_sample: bool = self.hparams.get("cmt", {}).get("use_neg_sample", False)

        # cSEBBs param
        self.sebbs_enabled: bool = self.hparams.get("sebbs", {}).get("enabled", False)

        if self.hparams["pretrained"]["e2e"]:
            self.pretrained_model: Any = pretrained_model  # BEATs or other pretrained model
        # else we use pre-computed embeddings from hdf5

        self.sed_teacher: Any = (
            deepcopy(sed_student) if sed_teacher is None else sed_teacher
        )  # CRNN
        self.opt: torch.optim.Optimizer | None = opt
        self.train_data: Any | None = train_data  # Dataset
        self.valid_data: Any | None = valid_data  # Dataset
        self.test_data: Any | None = test_data  # Dataset
        self.train_sampler: Any | None = train_sampler  # Sampler
        self.scheduler: Any | None = scheduler  # LR scheduler

        self.num_workers: int = 1 if self.fast_dev_run else self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        self.mel_spec: MelSpectrogram = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )

        for param in self.sed_teacher.parameters():
            param.detach_()

        # instantiating losses
        self.supervised_loss: torch.nn.BCELoss = torch.nn.BCELoss()
        self.selfsup_loss: torch.nn.MSELoss | torch.nn.BCELoss
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro: torchmetrics.Metric = (
            torchmetrics.classification.f_beta.MultilabelF1Score(
                len(self.encoder.labels),
                average="macro",
            )
        )
        self.get_weak_teacher_f1_seg_macro: torchmetrics.Metric = (
            torchmetrics.classification.f_beta.MultilabelF1Score(
                len(self.encoder.labels),
                average="macro",
            )
        )

        self.scaler: TorchScaler = self._init_scaler()
        # buffer for event based scores which we compute using sed-eval

        self.val_buffer_sed_scores_eval_student: ScoreDataFrameDict = {}
        self.val_buffer_sed_scores_eval_teacher: ScoreDataFrameDict = {}

        self.val_tune_sebbs_student: dict = {}
        self.val_tune_sebbs_teacher: dict = {}

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2),
            1,
            1 / test_n_thresholds,
        )
        self.test_buffer_psds_eval_student: dict[float, pd.DataFrame] = {
            k: pd.DataFrame() for k in test_thresholds
        }
        self.test_buffer_psds_eval_teacher: dict[float, pd.DataFrame] = {
            k: pd.DataFrame() for k in test_thresholds
        }
        self.test_buffer_sed_scores_eval_student: ScoreDataFrameDict = {}
        self.test_buffer_sed_scores_eval_teacher: ScoreDataFrameDict = {}
        self.test_buffer_sed_scores_eval_unprocessed_student: ScoreDataFrameDict = {}
        self.test_buffer_sed_scores_eval_unprocessed_teacher: ScoreDataFrameDict = {}
        self.test_buffer_detections_thres05_student: pd.DataFrame = pd.DataFrame()
        self.test_buffer_detections_thres05_teacher: pd.DataFrame = pd.DataFrame()

    _exp_dir = None

    @property
    def exp_dir(self) -> str:
        if self._exp_dir is None:
            try:
                self._exp_dir = self.logger.log_dir  # type: ignore[union-attr]  # Logger may be None
            except Exception:
                self._exp_dir = self.hparams["log_dir"]
        return self._exp_dir

    def log(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        """Override LightningModule.log to mirror logs to wandb when enabled.

        This calls the original pl.LightningModule.log and then attempts to
        send the same key/value to wandb.log (if wandb is initialized and
        use_wandb is enabled in hparams). Conversion of tensors/ndarrays to
        Python scalars/lists is handled to avoid serialization issues.
        """
        # call parent logging first to preserve Lightning behavior
        res = super(SEDTask4, self).log(name, value, *args, **kwargs)
        try:
            self._maybe_wandb_log({name: value})
        except Exception:
            # avoid any logging errors interfering with training
            pass
        return res

    def log_dict(self, dictionary: dict[str, Any], *args: Any, **kwargs: Any) -> None:  # type: ignore[override]  # Lightning expects specific Mapping type
        """Mirror a dictionary of metrics to wandb in addition to Lightning's log_dict."""
        res = super(SEDTask4, self).log_dict(dictionary, *args, **kwargs)
        try:
            self._maybe_wandb_log(dictionary)
        except Exception:
            pass
        return res

    def _maybe_wandb_log(self, log_dict: dict[str, Any]) -> None:
        """Safely log a dict to wandb if enabled and initialized.

        Converts torch tensors and numpy arrays to Python scalars or lists.
        If wandb isn't active or hparams disable it, this is a no-op.
        """
        try:
            if not self.hparams.get("wandb", {}).get("use_wandb"):
                return
        except Exception:
            return

        # bail out if wandb is not initialized
        try:
            if wandb.run is None:
                return
        except Exception:
            return

        def _to_native(x):
            if isinstance(x, (wandb.Histogram, wandb.Image, wandb.Audio)):
                return x

            # torch tensors
            try:
                import numbers

                if isinstance(x, torch.Tensor):
                    if x.numel() == 1:
                        return x.detach().cpu().item()
                    return x.detach().cpu().numpy().tolist()
                # numpy arrays
                if isinstance(x, (np.ndarray,)):
                    return x.tolist()
                # built-in number types
                if isinstance(x, (int, float, bool, str)):
                    return x
                # pandas scalars
                try:
                    import pandas as _pd

                    if isinstance(x, (_pd.Series, _pd.DataFrame)):
                        return x.to_dict()
                except Exception:
                    pass
                # fallback: try to cast to float, else str
                try:
                    return float(x)
                except Exception:
                    return str(x)
            except Exception:
                return str(x)

        payload = {k: _to_native(v) for k, v in log_dict.items()}

        # デバッグ用: 複数のstep情報を追加
        try:
            # PyTorch Lightning標準のglobal_step（バッチ処理の累積回数）
            payload["_debug/global_step"] = int(self.global_step)
            # カスタムスケジューラのstep_num
            if hasattr(self, "scheduler") and self.scheduler is not None:
                payload["_debug/scheduler_step_num"] = int(self.scheduler["scheduler"].step_num)
            # 現在のepoch数（ラウンド数）
            payload["_debug/current_epoch"] = int(self.current_epoch)
        except Exception:
            pass

        # attempt to set step if available
        step = None
        try:
            global_step = getattr(self, "global_step", None)
            step = int(global_step) if global_step is not None else None
        except Exception:
            step = None
        try:
            if step is not None:
                wandb.log(payload, step=step)
            else:
                wandb.log(payload)
        except Exception:
            # never raise from logging
            pass

    def _init_wandb_project(self) -> None:
        """Initialize wandb project with execution mode-aware directory management.

        This method implements mode-aware wandb initialization:
        - Detects execution mode (train/test/inference/feature_extraction)
        - Legacy mode (--wandb_dir) takes priority over new mode
        - Creates hierarchical experiment directory structure
        - Manages artifact directories and manifest generation
        - Skips wandb initialization for inference/feature_extraction modes
        """
        # Detect execution mode
        self.execution_mode = ExperimentDirManager.detect_execution_mode(
            self.hparams,
            evaluation=self.evaluation,
            test_state_dict=getattr(self, "_test_state_dict", None),
            fast_dev_run=self.fast_dev_run,
        )
        print(f"Detected execution mode: {self.execution_mode.value}")

        # ExperimentConfig を使用
        if "experiment" in self.hparams:
            exp_config = ExperimentConfig(**self.hparams["experiment"])

            # wandB初期化の可否を判定
            if not ExperimentDirManager.should_initialize_wandb(self.execution_mode, exp_config):
                print(f"WandB disabled for {self.execution_mode.value} mode")
                self._wandb_checkpoint_dir = None

                # inferenceモード時は非wandBディレクトリを作成
                base_dir = ExperimentDirManager.build_experiment_path(exp_config)
                # Use microsecond precision for unique directory names in concurrent executions
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                inference_dir = base_dir / f"run-{timestamp}"
                inference_dir.mkdir(parents=True, exist_ok=True)

                artifact_dirs = ExperimentDirManager.create_artifact_dirs(inference_dir)
                self._inference_dir = str(inference_dir)

                # manifest生成（run_id=None）
                ExperimentDirManager.generate_manifest(
                    inference_dir,
                    run_id=None,
                    config=self.hparams,
                    mode=self.execution_mode,
                )
                print(f"Inference directory created: {inference_dir}")
                return

            # wandB初期化（trainまたはtestモード）
            base_dir = ExperimentDirManager.build_experiment_path(exp_config)
            wandb.init(
                project=PROJECT_NAME,
                name=f"{exp_config.mode.value}/{exp_config.category}/{exp_config.method}/{exp_config.variant}",
                dir=str(base_dir),
                config=self.hparams,
                tags=[exp_config.mode.value, exp_config.category, exp_config.method],
            )

            if wandb.run is not None:
                experiment_dir = Path(wandb.run.dir)
                artifact_dirs = ExperimentDirManager.create_artifact_dirs(experiment_dir)
                self._wandb_checkpoint_dir = str(artifact_dirs["checkpoints"])
                print(f"Checkpoint directory: {self._wandb_checkpoint_dir}")

                # manifest生成（mode含む）
                ExperimentDirManager.generate_manifest(
                    experiment_dir,
                    wandb.run.id,
                    self.hparams,
                    mode=self.execution_mode,
                )
                print(f"Experiment directory: {experiment_dir}")

                # ハイパーパラメータをWandBに保存
                wandb.config.update(self.hparams, allow_val_change=True)
                print("WandB config updated with hyperparameters")

                # 設定ファイル自体をWandBに保存
                if "config_file_path" in self.hparams and os.path.exists(
                    self.hparams["config_file_path"],
                ):
                    wandb.save(
                        self.hparams["config_file_path"],
                        base_path=os.path.dirname(self.hparams["config_file_path"]),
                    )
                    print(f"Configuration file saved to WandB: {self.hparams['config_file_path']}")
            else:
                self._wandb_checkpoint_dir = None
        else:
            # experimentセクションが設定されていない場合はエラー
            msg = (
                "The 'experiment' section is required in the configuration YAML. "
                "Please add an 'experiment' section with mode, category, method, and variant fields."
            )
            raise ValueError(msg)

    def lr_scheduler_step(self, scheduler: Any, optimizer_idx: int, metric: Any) -> None:
        scheduler.step()

    def on_train_start(self) -> None:
        os.makedirs(os.path.join(self.exp_dir, "codecarbon"), exist_ok=True)
        self.tracker_train = OfflineEmissionsTracker(
            "DCASE Task 4 SED TRAINING",
            output_dir=os.path.join(self.exp_dir, "codecarbon"),
            output_file="emissions_baseline_training.csv",
            log_level="warning",
            country_iso_code="FRA",
            gpu_ids=[torch.cuda.current_device()],
        )
        self.tracker_train.start()

        # Remove for debugging. Those warnings can be ignored during training otherwise.
        to_ignore = [
            ".*Trying to infer the `batch_size` from an ambiguous collection.*",
            ".*invalid value encountered in divide*",
            ".*mean of empty slice*",
            ".*self.log*",
        ]
        for message in to_ignore:
            warnings.filterwarnings("ignore", message)

    def update_ema(
        self,
        alpha: float,
        global_step: int,
        model: torch.nn.Module,
        ema_model: torch.nn.Module,
    ) -> None:
        """Update teacher model parameters.

        Args:
            alpha: float, the factor to be used between each updated step.
            global_step: int, the current global step to be used.
            model: torch.Module, student model to use
            ema_model: torch.Module, teacher model to use

        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self) -> TorchScaler:
        """Scaler inizialization

        Raises:
            NotImplementedError: in case of not Implemented scaler

        Returns:
            TorchScaler: returns the scaler

        """
        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
        if self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"],
                    ),
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"],
                ),
            )
        return scaler

    def take_log(self, mels: torch.Tensor) -> torch.Tensor:
        """Apply the log transformation to mel spectrograms.

        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input

        """
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        # clamp to reproduce old code
        return amp_to_db(mels).clamp(min=-50, max=80)

    def detect(
        self,
        mel_feats: torch.Tensor,
        model: torch.nn.Module,
        embeddings: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings is None:
            return model(self.scaler(self.take_log(mel_feats)), **kwargs)
        return model(
            self.scaler(self.take_log(mel_feats)),
            embeddings=embeddings,
            **kwargs,
        )

    def _process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from pretrained model if e2e mode enabled.

        Preconditions:
        - embeddings tensor must match pretrained_model expected input shape
        - self.pretrained_model must be initialized if hparams["pretrained"]["e2e"] is True

        Postconditions:
        - Returns processed embeddings if e2e=True, otherwise returns input unchanged
        - Ensures pretrained_model is in eval mode if frozen=True

        Invariants:
        - Does not modify model weights (inference only)
        """
        if self.hparams["pretrained"]["e2e"]:
            if self.pretrained_model.training and self.hparams["pretrained"]["freezed"]:
                self.pretrained_model.eval()
            return self.pretrained_model(embeddings)[self.hparams["net"]["embedding_type"]]
        return embeddings

    def _generate_predictions(
        self,
        audio: torch.Tensor,
        embeddings: torch.Tensor,
        classes_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Generate student and teacher model predictions from audio.

        Preconditions:
        - audio tensor shape: (batch_size, audio_length)
        - embeddings must be processed via _process_embeddings()
        - classes_mask optional tensor for masked class predictions

        Postconditions:
        - Returns ((strong_student, weak_student), (strong_teacher, weak_teacher))
        - Each prediction tensor shape: (batch_size, time_steps, num_classes) for strong,
          (batch_size, num_classes) for weak

        Invariants:
        - Mel spectrogram computed once and shared across both models
        """
        mels = self.mel_spec(audio)
        strong_student, weak_student = self.detect(
            mels,
            self.sed_student,
            embeddings=embeddings,
            classes_mask=classes_mask,
        )
        strong_teacher, weak_teacher = self.detect(
            mels,
            self.sed_teacher,
            embeddings=embeddings,
            classes_mask=classes_mask,
        )
        return (strong_student, weak_student), (strong_teacher, weak_teacher)

    def _compute_step_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute supervised loss for masked predictions.

        Preconditions:
        - predictions and labels must have compatible shapes for BCE loss
        - mask (if provided) must be boolean tensor matching batch dimension

        Postconditions:
        - Returns scalar loss tensor
        - If mask provided, loss computed only on masked subset

        Invariants:
        - Loss is non-negative
        """
        if mask is not None:
            predictions = predictions[mask]
            labels = labels[mask]
        return self.supervised_loss(predictions, labels)

    def _update_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        metric_name: str,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Update specified metric with predictions and labels.

        Preconditions:
        - metric_name must correspond to initialized torchmetrics instance (e.g., "weak_student_f1_seg_macro")
        - predictions and labels must match metric's expected format
        - mask (if provided) must be boolean tensor matching batch dimension

        Postconditions:
        - Updates internal metric state (accumulated for epoch-end computation)
        - No return value (side effect only)

        Invariants:
        - Metric state remains valid after update
        """
        metric = getattr(self, f"get_{metric_name}")
        if mask is not None:
            predictions = predictions[mask]
            labels = labels[mask]
        metric(predictions, labels.long() if "f1" in metric_name else labels)

    def training_step(self, batch: TrainBatchType, batch_indx: int) -> torch.Tensor:  # type: ignore[override]
        """Apply the training for one batch (a step). Used during trainer.fit.

        Args:
            batch: TrainBatchType, 5-element batch tuple (without filenames)
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.

        Note:
            type: ignore[override] - PyTorch Lightning base class uses Any for batch parameter,
            but we use specific TrainBatchType for type safety in this implementation.

        """
        audio, labels, padded_indxs, embeddings, valid_class_mask = batch

        features = self.mel_spec(audio)

        # バッチ数の累積和を取得
        indx_maestro, indx_synth, indx_strong, indx_weak, indx_unlabelled = np.cumsum(
            self.hparams["training"]["batch_size"],
        )
        batch_num = features.shape[0]

        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(features).bool()
        weak_mask = torch.zeros(batch_num).to(features).bool()
        mask_consistency = torch.zeros(batch_num).to(features).bool()
        mask_pure_unlabeled = torch.zeros(batch_num).to(features).bool()

        strong_mask[:indx_strong] = 1  # maestro,合成 強ラベル
        weak_mask[indx_strong:indx_weak] = 1  # 弱ラベル
        mask_consistency[indx_maestro:] = 1  # maestro以外
        mask_pure_unlabeled[indx_weak:] = 1  # 純粋なラベルなし

        # Mixup
        mixup_type = self.hparams["training"].get("mixup")
        if mixup_type is not None and self.hparams["training"]["mixup_prob"] > random.random():
            # NOTE: mix only within same dataset !
            features, embeddings, labels = self.mixup_augmentor.apply_mixup(
                features,
                embeddings,
                labels,
                indx_strong,
                indx_weak,
            )
            features, embeddings, labels = self.mixup_augmentor.apply_mixup(
                features,
                embeddings,
                labels,
                indx_maestro,
                indx_strong,
            )
            features, embeddings, labels = self.mixup_augmentor.apply_mixup(
                features,
                embeddings,
                labels,
                0,
                indx_maestro,
            )
            features, embeddings, labels = self.mixup_augmentor.apply_mixup(
                features,
                embeddings,
                labels,
                indx_weak,
                indx_unlabelled,
            )

        # mask labels for invalid datasets classes after mixup.
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        labels = labels.masked_fill(
            ~valid_class_mask[:, :, None].expand_as(labels),
            0.0,
        )
        labels_weak = labels_weak.masked_fill(~valid_class_mask[weak_mask], 0.0)

        # --- Student forward ---
        strong_preds_student, weak_preds_student = self.detect(
            features,
            self.sed_student,
            embeddings=embeddings,
            classes_mask=valid_class_mask,
        )

        # ---Supervised loss ---
        loss_strong = self._compute_step_loss(
            strong_preds_student,
            labels,
            mask=strong_mask,
        )
        loss_weak = self._compute_step_loss(
            weak_preds_student[weak_mask],
            labels_weak,
        )
        tot_loss_supervised = loss_strong + loss_weak

        # ---Teacher Forward (No Grad) ---
        with torch.no_grad():
            strong_preds_teacher, weak_preds_teacher = self.detect(
                features,
                self.sed_teacher,
                embeddings=embeddings,
                classes_mask=valid_class_mask,
            )

        # --- Consistency Loss (Mean Teacher) ---
        weight = self.hparams["training"]["const_max"]
        if self.current_epoch < self.hparams["training"]["epoch_decay"]:
            weight *= self.scheduler["scheduler"]._get_scaling_factor()  # type: ignore[index]  # scheduler is dict-like

        cmt_active = self.cmt_enabled and (self.current_epoch >= self.cmt_warmup_epochs)

        # CMT
        if cmt_active:
            # Compute relative index_weak after mask_consistency filtering
            # mask_consistency excludes maestro data, so we need to adjust index
            index_weak_relative = indx_weak - indx_maestro

            # Apply CMT processing
            with torch.no_grad():
                if self.use_neg_sample:
                    # Apply CMT postprocessing to teacher predictions
                    (
                        teacher_pseudo_w,
                        teacher_pseudo_s,
                    ) = self.apply_cmt_postprocessing(
                        weak_preds_teacher[mask_consistency],
                        strong_preds_teacher[mask_consistency],
                        index_weak=index_weak_relative,
                    )
                    # Compute confidence weights
                    confidence_w, confidence_s = self.compute_cmt_confidence_weights(
                        weak_preds_teacher[mask_consistency],
                        strong_preds_teacher[mask_consistency],
                        teacher_pseudo_w,
                        teacher_pseudo_s,
                        index_weak=index_weak_relative,
                    )
                else:
                    # Apply CMT postprocessing to teacher predictions
                    teacher_pseudo_w, teacher_pseudo_s = self.apply_cmt_postprocessing(
                        weak_preds_teacher[mask_consistency],
                        strong_preds_teacher[mask_consistency],
                        index_weak=index_weak_relative,
                    )

                    # Compute confidence weights
                    confidence_w, confidence_s = self.compute_cmt_confidence_weights(
                        weak_preds_teacher[mask_consistency],
                        strong_preds_teacher[mask_consistency],
                        teacher_pseudo_w,
                        teacher_pseudo_s,
                        index_weak=index_weak_relative,
                    )

            # Compute CMT consistency loss with confidence weighting
            weak_self_sup_loss, strong_self_sup_loss = self.compute_cmt_consistency_loss(
                weak_preds_student[mask_consistency],
                strong_preds_student[mask_consistency],
                teacher_pseudo_w,
                teacher_pseudo_s,
                confidence_w,
                confidence_s,
            )
        else:
            strong_self_sup_loss = self.selfsup_loss(
                strong_preds_student[mask_consistency],
                strong_preds_teacher.detach()[mask_consistency],
            )
            weak_self_sup_loss = self.selfsup_loss(
                weak_preds_student[mask_consistency],
                weak_preds_teacher.detach()[mask_consistency],
            )
        tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss) * weight

        tot_loss = tot_loss_supervised + tot_self_loss

        # 教師あり学習の損失
        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
        self.log("train/student/tot_supervised", tot_loss_supervised, prog_bar=True)

        # 半教師あり学習の損失
        self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)
        self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
        self.log("train/student/tot_self_loss", tot_self_loss, prog_bar=True)

        # 総合的なloss
        self.log("train/student/tot_loss", tot_loss)

        # 学習率など
        self.log("train/weight", weight)
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True)  # type: ignore[union-attr]  # opt initialized in setup

        # 各種step情報
        self.log("train/step/global_step", self.global_step)
        self.log("train/step/current_epoch", self.current_epoch)

        return tot_loss

    def apply_cmt_postprocessing(
        self,
        y_w: torch.Tensor,
        y_s: torch.Tensor,
        index_weak: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Confident Mean Teacher postprocessing only unlabeled data.

        y_w: (batch, classes)
        y_s: (batch, classes, frames)
        """
        y_w_unlabeled = y_w[index_weak:]  # (unlabeled_batch, classes)
        y_s_unlabeled = y_s[index_weak:]  # (unlabeled_batch, classes, frames)

        if self.use_neg_sample:
            # Weak (clip-level)
            y_tilde_w = torch.where(
                y_w_unlabeled >= self.phi_pos,
                torch.ones_like(y_w_unlabeled),  # 正例 → 1.0
                torch.where(
                    y_w_unlabeled <= self.phi_neg,
                    torch.zeros_like(y_w_unlabeled),  # 負例 → 0.0
                    y_w_unlabeled,  # 中間値 → そのまま（後で信頼度0）
                ),
            )

            # Strong (frame-level) - Weak制約なし、独立判定
            y_s_binary = torch.where(
                y_s_unlabeled >= self.phi_pos,
                torch.ones_like(y_s_unlabeled),  # 正例 → 1.0
                torch.where(
                    y_s_unlabeled <= self.phi_neg,
                    torch.zeros_like(y_s_unlabeled),  # 負例 → 0.0
                    y_s_unlabeled,  # 中間値 → そのまま（後で信頼度0）
                ),
            )

        else:
            y_tilde_w = (y_w_unlabeled > self.phi_clip).float()  # clip
            y_w_expanded = y_tilde_w.unsqueeze(-1).expand_as(y_s_unlabeled)
            y_s_binary = y_w_expanded * (y_s_unlabeled > self.phi_frame).float()  # frame

        y_tilde_s_list = []
        original_device = y_s.device
        y_s_numpy = y_s_binary.detach().cpu().numpy()  # y_s_binary: (batch, classes, frames)

        for i in range(y_s_unlabeled.shape[0]):
            sample = y_s_numpy[i].transpose(1, 0)  # (classes, frames) -> (frames, classes)
            filtered = self.median_filter(sample)
            y_tilde_s_list.append(filtered)

        y_tilde_s_np = np.stack(y_tilde_s_list, axis=0)  # (batch, frames, classes)
        y_tilde_s_tensor = torch.from_numpy(y_tilde_s_np).to(original_device)
        y_tilde_s = y_tilde_s_tensor.transpose(1, 2)  # -> (batch, classes, frames)

        y_tilde_w = torch.cat((y_w[:index_weak], y_tilde_w), dim=0)
        y_tilde_s = torch.cat((y_s[:index_weak], y_tilde_s), dim=0)

        return y_tilde_w, y_tilde_s

    def compute_cmt_confidence_weights(
        self,
        y_w: torch.Tensor,
        y_s: torch.Tensor,
        y_tilde_w: torch.Tensor,
        y_tilde_s: torch.Tensor,
        index_weak: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute confidence weights based on teacher's certainty.

        Correction: High confidence for both positive (near 1) and negative (near 0) predictions.
        """
        if self.use_neg_sample:
            # --- use neg_sample ---
            # Unlabeled部分のマスク生成
            y_w_unlabeled = y_w[index_weak:]
            y_s_unlabeled = y_s[index_weak:]

            pos_mask_w_unlabeled = (y_tilde_w[index_weak:] == 1.0).float()
            neg_mask_w_unlabeled = (y_tilde_w[index_weak:] == 0.0).float()

            pos_mask_s_unlabeled = (y_tilde_s[index_weak:] == 1.0).float()
            neg_mask_s_unlabeled = (y_tilde_s[index_weak:] == 0.0).float()

            # Unlabeled部分の信頼度計算
            c_w_pos_unlabeled = pos_mask_w_unlabeled * y_w_unlabeled
            c_w_neg_unlabeled = neg_mask_w_unlabeled * (1.0 - y_w_unlabeled)
            c_w_unlabeled = c_w_pos_unlabeled + c_w_neg_unlabeled

            c_s_pos_unlabeled = pos_mask_s_unlabeled * y_s_unlabeled
            c_s_neg_unlabeled = neg_mask_s_unlabeled * (1.0 - y_s_unlabeled)
            c_s_unlabeled = c_s_pos_unlabeled + c_s_neg_unlabeled

            # Labeled部分は常に信頼度1.0
            c_w_labeled = torch.ones_like(y_w[:index_weak])
            c_s_labeled = torch.ones_like(y_s[:index_weak])

            # 結合（pos/negは後でスケーリングに使用）
            c_w_pos = torch.cat((c_w_labeled, c_w_pos_unlabeled), dim=0)
            c_w_neg = torch.cat((torch.zeros_like(y_w[:index_weak]), c_w_neg_unlabeled), dim=0)
            c_w = torch.cat((c_w_labeled, c_w_unlabeled), dim=0)

            c_s_pos = torch.cat((c_s_labeled, c_s_pos_unlabeled), dim=0)
            c_s_neg = torch.cat((torch.zeros_like(y_s[:index_weak]), c_s_neg_unlabeled), dim=0)
            c_s = torch.cat((c_s_labeled, c_s_unlabeled), dim=0)

        else:
            # --- CMT ---
            # Unlabeled部分のみ計算（labeled部分はy_tildeが元のy_wなので除外）
            y_w_unlabeled = y_w[index_weak:]
            y_s_unlabeled = y_s[index_weak:]
            y_tilde_w_unlabeled = y_tilde_w[index_weak:]
            y_tilde_s_unlabeled = y_tilde_s[index_weak:]

            c_w_unlabeled = y_w_unlabeled * y_tilde_w_unlabeled
            y_w_expanded = y_w_unlabeled.unsqueeze(-1).expand_as(y_s_unlabeled)
            c_s_unlabeled = y_s_unlabeled * y_w_expanded * y_tilde_s_unlabeled

            # Labeled部分は常に信頼度1.0
            c_w_labeled = torch.ones_like(y_w[:index_weak])
            c_s_labeled = torch.ones_like(y_s[:index_weak])

            # 結合
            c_w = torch.cat((c_w_labeled, c_w_unlabeled), dim=0)
            c_s = torch.cat((c_s_labeled, c_s_unlabeled), dim=0)

        # 実装予定
        # 温度パラメータによる調整

        if self.use_neg_sample and self.pos_neg_scale:  # 正例と負例の量を調整
            # Unlabeled部分のみの正例と負例の総重みを計算
            pos_sum_w_unlabeled = c_w_pos_unlabeled.sum().clamp(min=1e-6)
            neg_sum_w_unlabeled = c_w_neg_unlabeled.sum().clamp(min=1e-6)

            pos_sum_s_unlabeled = c_s_pos_unlabeled.sum().clamp(min=1e-6)
            neg_sum_s_unlabeled = c_s_neg_unlabeled.sum().clamp(min=1e-6)

            # Unlabeled部分のみをスケーリング
            if pos_sum_w_unlabeled < neg_sum_w_unlabeled:
                # 負例が多い → 負例を縮小
                scale_factor = neg_sum_w_unlabeled / pos_sum_w_unlabeled
                c_w_unlabeled_scaled = c_w_pos_unlabeled + c_w_neg_unlabeled / scale_factor
            else:
                # 正例が多い → 正例を縮小
                scale_factor = pos_sum_w_unlabeled / neg_sum_w_unlabeled
                c_w_unlabeled_scaled = c_w_pos_unlabeled / scale_factor + c_w_neg_unlabeled

            if pos_sum_s_unlabeled < neg_sum_s_unlabeled:
                scale_factor = neg_sum_s_unlabeled / pos_sum_s_unlabeled
                c_s_unlabeled_scaled = c_s_pos_unlabeled + c_s_neg_unlabeled / scale_factor
            else:
                scale_factor = pos_sum_s_unlabeled / neg_sum_s_unlabeled
                c_s_unlabeled_scaled = c_s_pos_unlabeled / scale_factor + c_s_neg_unlabeled

            # Labeledとスケール済みUnlabeledを結合
            c_w = torch.cat((c_w_labeled, c_w_unlabeled_scaled), dim=0)
            c_s = torch.cat((c_s_labeled, c_s_unlabeled_scaled), dim=0)

            # ログを更新（スケール前の値を記録）
            self.log("train/cmt/pos_neg_ratio_weak", pos_sum_w_unlabeled / neg_sum_w_unlabeled)
            self.log("train/cmt/pos_neg_ratio_strong", pos_sum_s_unlabeled / neg_sum_s_unlabeled)

        return c_w, c_s

    def compute_cmt_consistency_loss(
        self,
        student_w: torch.Tensor,
        student_s: torch.Tensor,
        teacher_pseudo_w: torch.Tensor,
        teacher_pseudo_s: torch.Tensor,
        confidence_w: torch.Tensor,
        confidence_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted BCE loss."""
        # Weak Loss
        bce_w = F.binary_cross_entropy(student_w, teacher_pseudo_w, reduction="none")
        weighted_bce_w = confidence_w * bce_w
        loss_w_con = weighted_bce_w.mean()

        # Strong Loss
        bce_s = F.binary_cross_entropy(student_s, teacher_pseudo_s, reduction="none")
        weighted_bce_s = confidence_s * bce_s
        loss_s_con = weighted_bce_s.mean()

        self.log("train/cmt/confidence_weak_mean", confidence_w.mean())
        self.log("train/cmt/confidence_weak_std", confidence_w.std())
        self.log("train/cmt/confidence_strong_mean", confidence_s.mean())
        self.log("train/cmt/confidence_strong_std", confidence_s.std())

        return loss_w_con, loss_s_con

    def on_before_zero_grad(self, *args: Any, **kwargs: Any) -> None:
        # update EMA teacher
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler["scheduler"].step_num,  # type: ignore[index]  # scheduler is dict-like
            self.sed_student,
            self.sed_teacher,
        )

    def validation_step(self, batch: EvalBatchType, batch_indx: int) -> None:  # type: ignore[override]
        """Apply validation to a batch (step). Used during trainer.fit.

        Args:
            batch: EvalBatchType, 6-element batch tuple (with filenames)
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:

        """
        audio, labels, _, filenames, embeddings, valid_class_mask = batch

        embeddings = self._process_embeddings(embeddings)

        # prediction for student and teacher using helper method
        (
            (strong_preds_student, weak_preds_student),
            (
                strong_preds_teacher,
                weak_preds_teacher,
            ),
        ) = self._generate_predictions(audio, embeddings, classes_mask=valid_class_mask)

        # we derive masks for each dataset based on folders of filenames
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent) == str(Path(self.hparams["data"]["weak_folder"]))
                    for x in filenames
                ],
            )
            .to(audio)
            .bool()
        )
        mask_strong = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    in [
                        str(Path(self.hparams["data"]["synth_val_folder"])),
                        str(Path(self.hparams["data"]["real_maestro_train_folder"])),
                    ]
                    for x in filenames
                ],
            )
            .to(audio)
            .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()

            loss_weak_student = self._compute_step_loss(
                weak_preds_student[mask_weak],
                labels_weak,
            )
            loss_weak_teacher = self._compute_step_loss(
                weak_preds_teacher[mask_weak],
                labels_weak,
            )
            self.log("val/weak/student/loss_weak", loss_weak_student)
            self.log("val/weak/teacher/loss_weak", loss_weak_teacher)

            # accumulate f1 score for weak labels
            self._update_metrics(
                weak_preds_student[mask_weak],
                labels_weak,
                "weak_student_f1_seg_macro",
            )
            self._update_metrics(
                weak_preds_teacher[mask_weak],
                labels_weak,
                "weak_teacher_f1_seg_macro",
            )

        if torch.any(mask_strong):
            loss_strong_student = self._compute_step_loss(
                strong_preds_student,
                labels,
                mask=mask_strong,
            )
            loss_strong_teacher = self._compute_step_loss(
                strong_preds_teacher,
                labels,
                mask=mask_strong,
            )

            self.log("val/synth/student/loss_strong", loss_strong_student)
            self.log("val/synth/teacher/loss_strong", loss_strong_teacher)

            filenames_strong = [
                x
                for x in filenames
                if str(Path(x).parent)
                in [
                    str(Path(self.hparams["data"]["synth_val_folder"])),
                    str(Path(self.hparams["data"]["real_maestro_train_folder"])),
                ]
            ]

            (
                scores_unprocessed_student_strong,
                scores_postprocessed_student_strong,
                decoded_student_strong,
            ) = batched_decode_preds(
                strong_preds_student[mask_strong],
                filenames_strong,
                self.encoder,
                median_filter=self.median_filter,
                thresholds=[],
            )

            self.val_buffer_sed_scores_eval_student.update(
                scores_postprocessed_student_strong,
            )

            self.val_tune_sebbs_student.update(
                scores_unprocessed_student_strong,
            )

            (
                scores_unprocessed_teacher_strong,
                scores_postprocessed_teacher_strong,
                decoded_teacher_strong,
            ) = batched_decode_preds(
                strong_preds_teacher[mask_strong],
                filenames_strong,
                self.encoder,
                median_filter=self.median_filter,
                thresholds=[],
            )

            self.val_buffer_sed_scores_eval_teacher.update(
                scores_postprocessed_teacher_strong,
            )

            self.val_tune_sebbs_teacher.update(
                scores_unprocessed_teacher_strong,
            )

    def validation_epoch_end(self, outputs: Any) -> dict[str, torch.Tensor]:  # type: ignore[override]  # Returns dict but Lightning expects None
        """Fonction applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.

        """
        # desed weak dataset
        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()
        # desed synth dataset
        desed_ground_truth = sed_scores_eval.io.read_ground_truth_events(
            self.hparams["data"]["synth_val_tsv"],
        )

        desed_audio_durations = sed_scores_eval.io.read_audio_durations(
            self.hparams["data"]["synth_val_dur"],
        )

        # --- ここから修正 ---
        # 両方のメタデータに共通して存在するaudio_idのみに絞り込む
        common_audio_ids = desed_ground_truth.keys() & desed_audio_durations.keys()
        desed_ground_truth = {
            audio_id: desed_ground_truth[audio_id] for audio_id in common_audio_ids
        }
        desed_audio_durations = {
            audio_id: desed_audio_durations[audio_id] for audio_id in common_audio_ids
        }
        # --- ここまで修正 ---

        # drop audios without events
        desed_ground_truth = {
            audio_id: gt for audio_id, gt in desed_ground_truth.items() if len(gt) > 0
        }
        desed_audio_durations = {
            audio_id: desed_audio_durations[audio_id] for audio_id in desed_ground_truth.keys()
        }
        keys = ["onset", "offset"] + sorted(classes_labels_desed.keys())
        desed_scores = {
            clip_id: self.val_buffer_sed_scores_eval_student[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }

        psds1_sed_scores_eval_student, _ = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
        intersection_f1_macro_thres05_student_sed_scores_eval = (
            sed_scores_eval.intersection_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                dtc_threshold=0.5,
                gtc_threshold=0.5,
            )[0]["macro_average"]
        )
        collar_f1_macro_thres05_student_sed_scores_eval = sed_scores_eval.collar_based.fscore(
            desed_scores,
            desed_ground_truth,
            threshold=0.5,
            onset_collar=0.2,
            offset_collar=0.2,
            offset_collar_rate=0.2,
        )[0]["macro_average"]
        desed_scores = {
            clip_id: self.val_buffer_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }
        psds1_sed_scores_eval_teacher, _ = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
        intersection_f1_macro_thres05_teacher_sed_scores_eval = (
            sed_scores_eval.intersection_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                dtc_threshold=0.5,
                gtc_threshold=0.5,
            )[0]["macro_average"]
        )
        collar_f1_macro_thres05_teacher_sed_scores_eval = sed_scores_eval.collar_based.fscore(
            desed_scores,
            desed_ground_truth,
            threshold=0.5,
            onset_collar=0.2,
            offset_collar=0.2,
            offset_collar_rate=0.2,
        )[0]["macro_average"]

        # maestro
        maestro_ground_truth = pd.read_csv(
            self.hparams["data"]["real_maestro_train_tsv"],
            sep="\t",
        )
        maestro_ground_truth = maestro_ground_truth[maestro_ground_truth.confidence > 0.5]
        maestro_ground_truth = maestro_ground_truth[
            maestro_ground_truth.event_label.isin(classes_labels_maestro_real_eval)
        ]
        maestro_ground_truth = {  # type: ignore[assignment]  # DataFrame converted to dict by comprehension
            clip_id: events
            for clip_id, events in sed_scores_eval.io.read_ground_truth_events(
                maestro_ground_truth,
            ).items()
            if clip_id in self.val_buffer_sed_scores_eval_student
        }
        maestro_ground_truth = _merge_overlapping_events(maestro_ground_truth)
        maestro_audio_durations = {
            clip_id: sorted(events, key=lambda x: x[1])[-1][1]
            for clip_id, events in maestro_ground_truth.items()
        }
        event_classes_maestro_eval = sorted(classes_labels_maestro_real_eval)
        keys = ["onset", "offset"] + event_classes_maestro_eval
        maestro_scores_student = {
            clip_id: self.val_buffer_sed_scores_eval_student[clip_id][keys]
            for clip_id in maestro_ground_truth.keys()
        }
        segment_f1_macro_optthres_student = sed_scores_eval.segment_based.best_fscore(
            maestro_scores_student,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
        )[0]["macro_average"]
        segment_mauc_student = sed_scores_eval.segment_based.auroc(
            maestro_scores_student,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
        )[0]["mean"]
        segment_mpauc_student = sed_scores_eval.segment_based.auroc(
            maestro_scores_student,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
            max_fpr=0.1,
        )[0]["mean"]
        maestro_scores_teacher = {
            clip_id: self.val_buffer_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in maestro_ground_truth.keys()
        }
        segment_f1_macro_optthres_teacher = sed_scores_eval.segment_based.best_fscore(
            maestro_scores_teacher,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
        )[0]["macro_average"]
        segment_mauc_teacher = sed_scores_eval.segment_based.auroc(
            maestro_scores_teacher,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
        )[0]["mean"]
        segment_mpauc_teacher = sed_scores_eval.segment_based.auroc(
            maestro_scores_teacher,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
            max_fpr=0.1,
        )[0]["mean"]

        obj_metric_synth_type = self.hparams["training"].get("obj_metric_synth_type")
        if obj_metric_synth_type is None:
            synth_metric = psds1_sed_scores_eval_student
        elif obj_metric_synth_type == "collar":
            synth_metric = collar_f1_macro_thres05_student_sed_scores_eval
        elif obj_metric_synth_type == "intersection":
            synth_metric = intersection_f1_macro_thres05_student_sed_scores_eval
        elif obj_metric_synth_type == "psds":
            synth_metric = psds1_sed_scores_eval_student
        else:
            raise NotImplementedError(
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented.",
            )

        obj_metric_maestro_type = self.hparams["training"].get(
            "obj_metric_maestro_type",
        )
        if obj_metric_maestro_type is None:
            maestro_metric = segment_mpauc_student
        elif obj_metric_maestro_type == "fmo":
            maestro_metric = segment_f1_macro_optthres_student
        elif obj_metric_maestro_type == "mauc":
            maestro_metric = segment_mauc_student
        elif obj_metric_maestro_type == "mpauc":
            maestro_metric = segment_f1_macro_optthres_student
        else:
            raise NotImplementedError(
                f"obj_metric_maestro_type: {obj_metric_maestro_type} not implemented.",
            )

        obj_metric = torch.tensor(
            weak_student_f1_macro.item() + synth_metric + maestro_metric,
        )

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log(
            "val/student/weak_f1_macro_thres05/torchmetrics",
            weak_student_f1_macro,
        )
        self.log(
            "val/teacher/weak_f1_macro_thres05/torchmetrics",
            weak_teacher_f1_macro,
        )
        self.log(
            "val/student/intersection_f1_macro_thres05/sed_scores_eval",
            intersection_f1_macro_thres05_student_sed_scores_eval,
        )
        self.log(
            "val/teacher/intersection_f1_macro_thres05/sed_scores_eval",
            intersection_f1_macro_thres05_teacher_sed_scores_eval,
        )
        self.log(
            "val/student/collar_f1_macro_thres05/sed_scores_eval",
            collar_f1_macro_thres05_student_sed_scores_eval,
        )
        self.log(
            "val/teacher/collar_f1_macro_thres05/sed_scores_eval",
            collar_f1_macro_thres05_teacher_sed_scores_eval,
        )
        self.log("val/student/psds1/sed_scores_eval", psds1_sed_scores_eval_student)
        self.log("val/teacher/psds1/sed_scores_eval", psds1_sed_scores_eval_teacher)
        self.log(
            "val/student/segment_f1_macro_thresopt/sed_scores_eval",
            segment_f1_macro_optthres_student,
        )
        self.log("val/student/segment_mauc/sed_scores_eval", segment_mauc_student)
        self.log("val/student/segment_mpauc/sed_scores_eval", segment_mpauc_student)
        self.log(
            "val/teacher/segment_f1_macro_thresopt/sed_scores_eval",
            segment_f1_macro_optthres_teacher,
        )
        self.log("val/teacher/segment_mauc/sed_scores_eval", segment_mauc_teacher)
        self.log("val/teacher/segment_mpauc/sed_scores_eval", segment_mpauc_teacher)

        # free the buffers
        self.val_buffer_sed_scores_eval_student = {}
        self.val_buffer_sed_scores_eval_teacher = {}

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        return obj_metric  # type: ignore[return-value]  # Lightning expects None but we return metric dict

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]  # Returns dict but Lightning expects None
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch: EvalBatchType, batch_indx: int) -> None:  # type: ignore[override]
        """Apply Test to a batch (step), used only when (trainer.test is called).

        Args:
            batch: EvalBatchType, 6-element batch tuple (with filenames)
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:

        """
        audio, labels, padded_indxs, filenames, embeddings, valid_class_mask = batch

        embeddings = self._process_embeddings(embeddings)

        # prediction for student and teacher using helper method
        (
            (strong_preds_student, weak_preds_student),
            (
                strong_preds_teacher,
                weak_preds_teacher,
            ),
        ) = self._generate_predictions(audio, embeddings)

        if not self.evaluation:
            loss_strong_student = self._compute_step_loss(strong_preds_student, labels)
            loss_strong_teacher = self._compute_step_loss(strong_preds_teacher, labels)

            self.log("test/student/loss_strong", loss_strong_student)
            self.log("test/teacher/loss_strong", loss_strong_teacher)

        if self.sebbs_enabled:
            # desed synth dataset
            desed_ground_truth = sed_scores_eval.io.read_ground_truth_events(
                self.hparams["data"]["synth_val_tsv"],
            )

            desed_audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["synth_val_dur"],
            )

            # 両方のメタデータに共通して存在するaudio_idのみに絞り込む
            common_audio_ids = desed_ground_truth.keys() & desed_audio_durations.keys()
            desed_ground_truth = {
                audio_id: desed_ground_truth[audio_id] for audio_id in common_audio_ids
            }
            desed_audio_durations = {
                audio_id: desed_audio_durations[audio_id] for audio_id in common_audio_ids
            }

            # drop audios without events
            desed_ground_truth = {
                audio_id: gt for audio_id, gt in desed_ground_truth.items() if len(gt) > 0
            }
            desed_audio_durations = {
                audio_id: desed_audio_durations[audio_id] for audio_id in desed_ground_truth.keys()
            }
            keys = ["onset", "offset"] + sorted(classes_labels_desed.keys())
            desed_scores = {
                clip_id: self.val_tune_sebbs_student[clip_id][keys]
                for clip_id in desed_ground_truth.keys()
            }

            # # --- 1. DESEDクラス用のcSEBBsチューニング ---
            if not hasattr(self, "csebbs_predictor_desed"):
                print("\n=== Tuning cSEBBs for DESED classes ===")
                self.csebbs_predictor_desed, _ = SEBBsTuner.tune_for_psds(
                    scores=desed_scores,
                    ground_truth=desed_ground_truth,
                    audio_durations=desed_audio_durations,
                )
                print("✓ DESED cSEBBs tuning completed")

            # --- 1-2. DESEDクラス用のcSEBBsチューニング（教師モデル） ---
            if not hasattr(self, "csebbs_predictor_desed_teacher"):
                print("\n=== Tuning cSEBBs for DESED classes (Teacher) ===")
                desed_scores_teacher = {
                    clip_id: self.val_tune_sebbs_teacher[clip_id][keys]
                    for clip_id in desed_ground_truth.keys()
                }
                self.csebbs_predictor_desed_teacher, _ = SEBBsTuner.tune_for_psds(
                    scores=desed_scores_teacher,
                    ground_truth=desed_ground_truth,
                    audio_durations=desed_audio_durations,
                )
                print("✓ DESED cSEBBs tuning completed (Teacher)")

            # --- 2. MAESTROクラス用のcSEBBsチューニング ---
            # MAESTROとDESEDでは音響特性が異なるため、別々にチューニング
            if not hasattr(self, "csebbs_predictor_maestro"):
                print("\n=== Tuning cSEBBs for MAESTRO classes ===")

                # on_test_start()で読み込んだMAESTROのメタデータを取得
                # clip ベースの ground truth を使用（validation_epoch_end と同じパターン）
                maestro_ground_truth_all_clips = getattr(self, "_maestro_ground_truth_clips", {})

                if maestro_ground_truth_all_clips:
                    # validation buffer に存在する clip のみを抽出
                    maestro_ground_truth = {
                        clip_id: events
                        for clip_id, events in maestro_ground_truth_all_clips.items()
                        if clip_id in self.val_tune_sebbs_student
                    }
                    maestro_ground_truth = _merge_overlapping_events(maestro_ground_truth)
                    maestro_audio_durations = {
                        clip_id: sorted(events, key=lambda x: x[1])[-1][1]
                        for clip_id, events in maestro_ground_truth.items()
                    }

                    # MAESTROクラスのみを含むvalidationスコアを抽出
                    event_classes_maestro = sorted(classes_labels_maestro_real_eval)
                    keys = ["onset", "offset"] + event_classes_maestro

                    # maestro_ground_truth に存在する clip のスコアのみを抽出
                    maestro_val_scores = {
                        clip_id: self.val_tune_sebbs_student[clip_id][keys]
                        for clip_id in maestro_ground_truth.keys()
                    }

                    # --- デバッグログ: マッチング状況を確認 ---
                    total_val_clips = len(self.val_tune_sebbs_student)
                    total_gt_clips = len(maestro_ground_truth)

                    matched_clips = len(maestro_val_scores)

                    self.log("debug/maestro_student/total_validation_clips", total_val_clips)
                    self.log("debug/maestro_student/total_ground_truth_clips", total_gt_clips)
                    self.log("debug/maestro_student/matched_clips", matched_clips)

                    if matched_clips == 0:
                        # マッチング失敗時の詳細情報をprintで表示
                        sample_val_clips = list(self.val_tune_sebbs_student.keys())[:3]
                        sample_gt_clips = list(maestro_ground_truth.keys())[:3]
                        print("\n[DEBUG] MAESTRO Student matching failed!")
                        print(f"  Sample validation clip IDs: {sample_val_clips}")
                        print(f"  Sample ground truth clip IDs: {sample_gt_clips}")

                    segment_length = 1.0
                    if maestro_val_scores:
                        # 十分なvalidationデータがある場合はチューニング実行
                        self.csebbs_predictor_maestro, _ = SEBBsTuner.tune(
                            scores=maestro_val_scores,
                            ground_truth=maestro_ground_truth,
                            audio_durations=maestro_audio_durations,
                            selection_fn=select_best_auroc,  # AUROCを最大化
                            segment_length=segment_length,
                        )
                        print(
                            f"✓ MAESTRO cSEBBs tuning completed with {len(maestro_val_scores)} clips",
                        )
                    else:
                        # validationデータが見つからない場合はデフォルトパラメータを使用
                        print("⚠ Warning: No MAESTRO validation scores found")
                        print("  Using default cSEBBs parameters for MAESTRO")
                        self.csebbs_predictor_maestro = SEBBsPredictor(
                            step_filter_length=0.48,  # 中程度のフィルタ長
                            merge_threshold_abs=0.2,  # 中程度の絶対閾値
                            merge_threshold_rel=2.0,  # 中程度の相対閾値
                        )
                else:
                    # ground truthやdurationsが読み込まれていない場合
                    print("⚠ Warning: MAESTRO ground truth or durations not available")
                    print("  Using default cSEBBs parameters for MAESTRO")
                    self.csebbs_predictor_maestro = SEBBsPredictor(
                        step_filter_length=0.48,
                        merge_threshold_abs=0.2,
                        merge_threshold_rel=2.0,
                    )

            # --- 2-2. MAESTROクラス用のcSEBBsチューニング（教師モデル） ---
            if not hasattr(self, "csebbs_predictor_maestro_teacher"):
                print("\n=== Tuning cSEBBs for MAESTRO classes (Teacher) ===")

                # clip ベースの ground truth を使用（validation_epoch_end と同じパターン）
                maestro_ground_truth_all_clips = getattr(self, "_maestro_ground_truth_clips", {})

                if maestro_ground_truth_all_clips:
                    # validation buffer に存在する clip のみを抽出
                    maestro_ground_truth = {
                        clip_id: events
                        for clip_id, events in maestro_ground_truth_all_clips.items()
                        if clip_id in self.val_tune_sebbs_teacher
                    }
                    maestro_ground_truth = _merge_overlapping_events(maestro_ground_truth)
                    maestro_audio_durations = {
                        clip_id: sorted(events, key=lambda x: x[1])[-1][1]
                        for clip_id, events in maestro_ground_truth.items()
                    }

                    event_classes_maestro = sorted(classes_labels_maestro_real_eval)
                    keys = ["onset", "offset"] + event_classes_maestro

                    # maestro_ground_truth に存在する clip のスコアのみを抽出
                    maestro_val_scores_teacher = {
                        clip_id: self.val_tune_sebbs_teacher[clip_id][keys]
                        for clip_id in maestro_ground_truth.keys()
                    }

                    # --- デバッグログ: マッチング状況を確認 (Teacher) ---
                    total_val_clips_teacher = len(self.val_tune_sebbs_teacher)
                    total_gt_clips_teacher = len(maestro_ground_truth)
                    matched_clips_teacher = len(maestro_val_scores_teacher)

                    self.log(
                        "debug/maestro_teacher/total_validation_clips",
                        total_val_clips_teacher,
                    )
                    self.log(
                        "debug/maestro_teacher/total_ground_truth_clips",
                        total_gt_clips_teacher,
                    )
                    self.log("debug/maestro_teacher/matched_clips", matched_clips_teacher)

                    if matched_clips_teacher == 0:
                        # マッチング失敗時の詳細情報をprintで表示
                        sample_val_clips = list(self.val_tune_sebbs_teacher.keys())[:3]
                        sample_gt_clips = list(maestro_ground_truth.keys())[:3]
                        print("\n[DEBUG] MAESTRO Teacher matching failed!")
                        print(f"  Sample validation clip IDs: {sample_val_clips}")
                        print(f"  Sample ground truth clip IDs: {sample_gt_clips}")

                    if maestro_val_scores_teacher:
                        self.csebbs_predictor_maestro_teacher, _ = SEBBsTuner.tune_for_psds(
                            scores=maestro_val_scores_teacher,
                            ground_truth=maestro_ground_truth,
                            audio_durations=maestro_audio_durations,
                        )
                        print(
                            f"✓ MAESTRO cSEBBs tuning completed (Teacher) with {len(maestro_val_scores_teacher)} clips",
                        )
                    else:
                        print("⚠ Warning: No MAESTRO validation scores found (Teacher)")
                        print("  Using default cSEBBs parameters for MAESTRO (Teacher)")
                        self.csebbs_predictor_maestro_teacher = SEBBsPredictor(
                            step_filter_length=0.48,
                            merge_threshold_abs=0.2,
                            merge_threshold_rel=2.0,
                        )
                else:
                    print("⚠ Warning: MAESTRO ground truth or durations not available (Teacher)")
                    print("  Using default cSEBBs parameters for MAESTRO (Teacher)")
                    self.csebbs_predictor_maestro_teacher = SEBBsPredictor(
                        step_filter_length=0.48,
                        merge_threshold_abs=0.2,
                        merge_threshold_rel=2.0,
                    )

        # Student modelのスコア生成とcSEBBs後処理
        (
            scores_unprocessed_student_strong,  # median filter適用前のスコア
            scores_postprocessed_student_strong,  # 検証用に有効化したが,本当は使用しない
            decoded_student_strong,  # 閾値別のバイナリ検出結果
        ) = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.median_filter,
            thresholds=list(self.test_buffer_psds_eval_student.keys()) + [0.5],
        )

        if self.sebbs_enabled:
            scores_postprocessed_student_strong = get_sebbs(
                self,
                scores_unprocessed_student_strong,
                model_type="student",
            )

        # 後処理前後のスコアを保存（比較・分析用）
        self.test_buffer_sed_scores_eval_unprocessed_student.update(
            scores_unprocessed_student_strong,
        )
        # cSEBBs後処理済みスコア（最終的な評価に使用）
        self.test_buffer_sed_scores_eval_student.update(
            scores_postprocessed_student_strong,
        )
        for th in self.test_buffer_psds_eval_student.keys():
            self.test_buffer_psds_eval_student[th] = pd.concat(
                [self.test_buffer_psds_eval_student[th], decoded_student_strong[th]],
                ignore_index=True,
            )

        # Teacher modelについてもStudent modelと同様の処理を実行
        (
            scores_unprocessed_teacher_strong,  # median filter適用前のスコア
            scores_postprocessed_teacher_strong,  # 使用しない、cSEBBsで置き換える
            decoded_teacher_strong,  # 閾値別のバイナリ検出結果
        ) = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.median_filter,
            thresholds=list(self.test_buffer_psds_eval_teacher.keys()) + [0.5],
        )

        if self.sebbs_enabled:
            scores_postprocessed_teacher_strong = get_sebbs(
                self,
                scores_unprocessed_teacher_strong,
                model_type="teacher",
            )

        self.test_buffer_sed_scores_eval_unprocessed_teacher.update(
            scores_unprocessed_teacher_strong,
        )
        self.test_buffer_sed_scores_eval_teacher.update(
            scores_postprocessed_teacher_strong,
        )
        for th in self.test_buffer_psds_eval_teacher.keys():
            self.test_buffer_psds_eval_teacher[th] = pd.concat(
                [self.test_buffer_psds_eval_teacher[th], decoded_teacher_strong[th]],
                ignore_index=True,
            )

        # compute f1 score
        self.test_buffer_detections_thres05_student = pd.concat(
            [self.test_buffer_detections_thres05_student, decoded_student_strong[0.5]],
        )
        self.test_buffer_detections_thres05_teacher = pd.concat(
            [self.test_buffer_detections_thres05_teacher, decoded_teacher_strong[0.5]],
        )

    def _save_per_class_psds(
        self,
        single_class_psds_dict: dict[str, float],
        save_path: str,
        dataset_name: str,
        model_name: str,
        scenario_name: str | None = None,
    ) -> None:
        metrics_list = []
        for class_name, psds_value in single_class_psds_dict.items():
            metrics_list.append(
                {
                    "class": class_name,
                    "psds": float(psds_value),
                    "dataset": dataset_name,
                    "model": model_name,
                },
            )
            if scenario_name is not None:
                metrics_list[-1]["scenario"] = scenario_name

        df = pd.DataFrame(metrics_list)
        # PSDS降順でソート
        df = df.sort_values("psds", ascending=False)

        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, float_format="%.4f")

        print(f"\n[Per-class PSDS] Saved to: {save_path}")
        print(df.to_string(index=False))

    def _save_per_class_mpauc(
        self,
        auroc_results_dict: dict[str, float],
        save_path: str,
        dataset_name: str,
        model_name: str,
    ) -> None:
        metrics_list = []
        for class_name, mpauc_value in auroc_results_dict.items():
            if class_name == "mean":
                continue  # 全体平均はスキップ
            metrics_list.append(
                {
                    "class": class_name,
                    "mpauc": float(mpauc_value),
                    "dataset": dataset_name,
                    "model": model_name,
                },
            )

        if not metrics_list:
            print("[Warning] No per-class mpAUC found in auroc results")
            return

        df = pd.DataFrame(metrics_list)
        # mpAUC降順でソート
        df = df.sort_values("mpauc", ascending=False)

        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, float_format="%.4f")

        print(f"\n[Per-class mpAUC] Saved to: {save_path}")
        print(df.to_string(index=False))

    def on_test_epoch_end(self) -> None:
        # pub eval dataset
        save_dir = os.path.join(self.exp_dir, "metrics_test")
        print("save_dir", save_dir)

        # wandb runディレクトリ配下にclass-wise-csvを作成
        if wandb.run is not None:
            csv_dir = os.path.join(wandb.run.dir, "class-wise-csv")
        else:
            csv_dir = os.path.join(str(self._exp_dir), "class-wise-csv")
        print("csv_dir", csv_dir)

        results = {}
        if self.evaluation:
            # only save prediction scores
            save_dir_student_unprocessed = os.path.join(
                save_dir,
                "student_scores",
                "unprocessed",
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_unprocessed_student,
                save_dir_student_unprocessed,
            )
            print(f"\nRaw scores for student saved in: {save_dir_student_unprocessed}")

            save_dir_student_postprocessed = os.path.join(
                save_dir,
                "student_scores",
                "postprocessed",
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_student,
                save_dir_student_postprocessed,
            )
            print(
                f"\nPostprocessed scores for student saved in: {save_dir_student_postprocessed}",
            )

            save_dir_teacher_unprocessed = os.path.join(
                save_dir,
                "teacher_scores",
                "unprocessed",
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_unprocessed_teacher,
                save_dir_teacher_unprocessed,
            )
            print(f"\nRaw scores for teacher saved in: {save_dir_teacher_unprocessed}")

            save_dir_teacher_postprocessed = os.path.join(
                save_dir,
                "teacher_scores",
                "postprocessed",
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_teacher,
                save_dir_teacher_postprocessed,
            )
            print(
                f"\nPostprocessed scores for teacher saved in: {save_dir_teacher_postprocessed}",
            )

            self.tracker_eval.stop()
        else:
            # calculate the metrics
            # psds_eval
            psds1_student_psds_eval = compute_psds_from_operating_points(
                self.test_buffer_psds_eval_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )
            psds2_student_psds_eval = compute_psds_from_operating_points(
                self.test_buffer_psds_eval_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )
            psds1_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_buffer_psds_eval_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )
            psds2_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_buffer_psds_eval_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )
            # synth dataset
            intersection_f1_macro_thres05_student_psds_eval = compute_per_intersection_macro_f1(
                {"0.5": self.test_buffer_detections_thres05_student},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )
            intersection_f1_macro_thres05_teacher_psds_eval = compute_per_intersection_macro_f1(
                {"0.5": self.test_buffer_detections_thres05_teacher},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )
            # sed_eval
            collar_f1_macro_thres05_student = log_sedeval_metrics(
                self.test_buffer_detections_thres05_student,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student"),
            )[0]
            collar_f1_macro_thres05_teacher = log_sedeval_metrics(
                self.test_buffer_detections_thres05_teacher,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher"),
            )[0]

            # sed_scores_eval
            desed_ground_truth = sed_scores_eval.io.read_ground_truth_events(
                self.hparams["data"]["test_tsv"],
            )
            desed_audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["test_dur"],
            )

            # --- ここから修正 ---
            # 両方のメタデータに共通して存在するaudio_idのみに絞り込む
            common_audio_ids = desed_ground_truth.keys() & desed_audio_durations.keys()
            desed_ground_truth = {
                audio_id: desed_ground_truth[audio_id] for audio_id in common_audio_ids
            }
            desed_audio_durations = {
                audio_id: desed_audio_durations[audio_id] for audio_id in common_audio_ids
            }
            # --- ここまで修正 ---

            # drop audios without events
            desed_ground_truth = {
                audio_id: gt for audio_id, gt in desed_ground_truth.items() if len(gt) > 0
            }
            desed_audio_durations = {
                audio_id: desed_audio_durations[audio_id] for audio_id in desed_ground_truth.keys()
            }
            keys = ["onset", "offset"] + sorted(classes_labels_desed.keys())
            desed_scores = {
                clip_id: self.test_buffer_sed_scores_eval_student[clip_id][keys]
                for clip_id in desed_ground_truth.keys()
            }
            psds1_student_sed_scores_eval, psds1_student_per_class = compute_psds_from_scores(
                desed_scores,
                desed_ground_truth,
                desed_audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )
            psds2_student_sed_scores_eval, psds2_student_per_class = compute_psds_from_scores(
                desed_scores,
                desed_ground_truth,
                desed_audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )

            self._save_per_class_psds(
                psds1_student_per_class,
                os.path.join(csv_dir, "per_class_psds_desed_student_scenario1.csv"),
                dataset_name="DESED",
                model_name="student",
                scenario_name="scenario1",
            )

            self._save_per_class_psds(
                psds2_student_per_class,
                os.path.join(csv_dir, "per_class_psds_desed_student_scenario2.csv"),
                dataset_name="DESED",
                model_name="student",
                scenario_name="scenario2",
            )

            intersection_f1_macro_thres05_student_sed_scores_eval = (
                sed_scores_eval.intersection_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    dtc_threshold=0.5,
                    gtc_threshold=0.5,
                )[0]["macro_average"]
            )
            collar_f1_macro_thres05_student_sed_scores_eval = sed_scores_eval.collar_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                onset_collar=0.2,
                offset_collar=0.2,
                offset_collar_rate=0.2,
            )[0]["macro_average"]

            desed_scores = {
                clip_id: self.test_buffer_sed_scores_eval_teacher[clip_id][keys]
                for clip_id in desed_ground_truth.keys()
            }
            psds1_teacher_sed_scores_eval, psds1_teacher_per_class = compute_psds_from_scores(
                desed_scores,
                desed_ground_truth,
                desed_audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )
            psds2_teacher_sed_scores_eval, psds2_teacher_per_class = compute_psds_from_scores(
                desed_scores,
                desed_ground_truth,
                desed_audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )

            self._save_per_class_psds(
                psds1_teacher_per_class,
                os.path.join(csv_dir, "per_class_psds_desed_teacher_scenario1.csv"),
                dataset_name="DESED",
                model_name="teacher",
                scenario_name="scenario1",
            )

            self._save_per_class_psds(
                psds2_teacher_per_class,
                os.path.join(csv_dir, "per_class_psds_desed_teacher_scenario2.csv"),
                dataset_name="DESED",
                model_name="teacher",
                scenario_name="scenario2",
            )

            intersection_f1_macro_thres05_teacher_sed_scores_eval = (
                sed_scores_eval.intersection_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    dtc_threshold=0.5,
                    gtc_threshold=0.5,
                )[0]["macro_average"]
            )
            collar_f1_macro_thres05_teacher_sed_scores_eval = sed_scores_eval.collar_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                onset_collar=0.2,
                offset_collar=0.2,
                offset_collar_rate=0.2,
            )[0]["macro_average"]

            maestro_audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["real_maestro_val_dur"],
            )
            maestro_ground_truth_clips = pd.read_csv(
                self.hparams["data"]["real_maestro_val_tsv"],
                sep="\t",
            )
            maestro_clip_ids = [
                filename[:-4] for filename in maestro_ground_truth_clips["filename"]
            ]
            maestro_ground_truth_clips = maestro_ground_truth_clips[
                maestro_ground_truth_clips.confidence > 0.5
            ]
            maestro_ground_truth_clips = maestro_ground_truth_clips[
                maestro_ground_truth_clips.event_label.isin(
                    classes_labels_maestro_real_eval,
                )
            ]
            maestro_ground_truth_clips = sed_scores_eval.io.read_ground_truth_events(
                maestro_ground_truth_clips,
            )

            # clip ベースの ground truth を保存（validation_epoch_end と同じパターン）
            self._maestro_ground_truth_clips = maestro_ground_truth_clips

            # file ベースに変換したものも保存（test 全体の評価用）
            maestro_ground_truth = _merge_maestro_ground_truth(
                maestro_ground_truth_clips,
            )
            self._maestro_ground_truth = maestro_ground_truth
            maestro_audio_durations = {
                file_id: maestro_audio_durations[file_id] for file_id in maestro_ground_truth.keys()
            }
            self._maestro_audio_durations = maestro_audio_durations

            maestro_scores_student = {
                clip_id: self.test_buffer_sed_scores_eval_student[clip_id]
                for clip_id in maestro_clip_ids
            }
            maestro_scores_teacher = {
                clip_id: self.test_buffer_sed_scores_eval_teacher[clip_id]
                for clip_id in maestro_clip_ids
            }
            segment_length = 1.0
            event_classes_maestro = sorted(classes_labels_maestro_real_eval)  # 他の都合でevalに変更
            segment_scores_student = _get_segment_scores_and_overlap_add(
                frame_scores=maestro_scores_student,
                audio_durations=maestro_audio_durations,
                event_classes=event_classes_maestro,
                segment_length=segment_length,
            )
            sed_scores_eval.io.write_sed_scores(
                segment_scores_student,
                os.path.join(save_dir, "student", "maestro", "postprocessed"),
            )
            segment_scores_teacher = _get_segment_scores_and_overlap_add(
                frame_scores=maestro_scores_teacher,
                audio_durations=maestro_audio_durations,
                event_classes=event_classes_maestro,
                segment_length=segment_length,
            )
            sed_scores_eval.io.write_sed_scores(
                segment_scores_teacher,
                os.path.join(save_dir, "teacher", "maestro", "postprocessed"),
            )

            event_classes_maestro_eval = sorted(classes_labels_maestro_real_eval)
            keys = ["onset", "offset"] + event_classes_maestro_eval
            segment_scores_student = {
                clip_id: scores_df[keys] for clip_id, scores_df in segment_scores_student.items()
            }
            segment_scores_teacher = {
                clip_id: scores_df[keys] for clip_id, scores_df in segment_scores_teacher.items()
            }

            segment_f1_macro_optthres_student = sed_scores_eval.segment_based.best_fscore(
                segment_scores_student,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
            )[0]["macro_average"]
            segment_mauc_student = sed_scores_eval.segment_based.auroc(
                segment_scores_student,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
            )[0]["mean"]

            segment_mpauc_student_dict = sed_scores_eval.segment_based.auroc(
                segment_scores_student,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
                max_fpr=0.1,
            )[0]
            segment_mpauc_student = segment_mpauc_student_dict["mean"]

            self._save_per_class_mpauc(
                segment_mpauc_student_dict,
                os.path.join(csv_dir, "per_class_mpauc_maestro_student.csv"),
                dataset_name="MAESTRO",
                model_name="student",
            )

            segment_f1_macro_optthres_teacher = sed_scores_eval.segment_based.best_fscore(
                segment_scores_teacher,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
            )[0]["macro_average"]
            segment_mauc_teacher = sed_scores_eval.segment_based.auroc(
                segment_scores_teacher,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
            )[0]["mean"]

            segment_mpauc_teacher_dict = sed_scores_eval.segment_based.auroc(
                segment_scores_teacher,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
                max_fpr=0.1,
            )[0]

            segment_mpauc_teacher = segment_mpauc_teacher_dict["mean"]

            self._save_per_class_mpauc(
                segment_mpauc_teacher_dict,
                os.path.join(csv_dir, "per_class_mpauc_maestro_teacher.csv"),
                dataset_name="MAESTRO",
                model_name="teacher",
            )

            results.update(
                {
                    "test/student/psds1/psds_eval": psds1_student_psds_eval,
                    "test/student/psds2/psds_eval": psds2_student_psds_eval,
                    "test/teacher/psds1/psds_eval": psds1_teacher_psds_eval,
                    "test/teacher/psds2/psds_eval": psds2_teacher_psds_eval,
                    "test/student/intersection_f1_macro_thres05/psds_eval": intersection_f1_macro_thres05_student_psds_eval,
                    "test/teacher/intersection_f1_macro_thres05/psds_eval": intersection_f1_macro_thres05_teacher_psds_eval,
                    "test/student/collar_f1_macro_thres05/sed_eval": collar_f1_macro_thres05_student,
                    "test/teacher/collar_f1_macro_thres05/sed_eval": collar_f1_macro_thres05_teacher,
                    "test/student/psds1/sed_scores_eval": psds1_student_sed_scores_eval,
                    "test/student/psds2/sed_scores_eval": psds2_student_sed_scores_eval,
                    "test/teacher/psds1/sed_scores_eval": psds1_teacher_sed_scores_eval,
                    "test/teacher/psds2/sed_scores_eval": psds2_teacher_sed_scores_eval,
                    "test/student/intersection_f1_macro_thres05/sed_scores_eval": intersection_f1_macro_thres05_student_sed_scores_eval,
                    "test/teacher/intersection_f1_macro_thres05/sed_scores_eval": intersection_f1_macro_thres05_teacher_sed_scores_eval,
                    "test/student/collar_f1_macro_thres05/sed_scores_eval": collar_f1_macro_thres05_student_sed_scores_eval,
                    "test/teacher/collar_f1_macro_thres05/sed_scores_eval": collar_f1_macro_thres05_teacher_sed_scores_eval,
                    "test/student/segment_f1_macro_thresopt/sed_scores_eval": segment_f1_macro_optthres_student,
                    "test/student/segment_mauc/sed_scores_eval": segment_mauc_student,
                    "test/student/segment_mpauc/sed_scores_eval": segment_mpauc_student,
                    "test/teacher/segment_f1_macro_thresopt/sed_scores_eval": segment_f1_macro_optthres_teacher,
                    "test/teacher/segment_mauc/sed_scores_eval": segment_mauc_teacher,
                    "test/teacher/segment_mpauc/sed_scores_eval": segment_mpauc_teacher,
                },
            )
            self.tracker_devtest.stop()

        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)

        for key in results:
            self.log(key, results[key], prog_bar=True, logger=True)
        wandb.finish()

    def configure_optimizers(self) -> list[list[torch.optim.Optimizer] | list[dict]]:
        return [self.opt], [self.scheduler]  # type: ignore[return-value]  # opt and scheduler initialized in setup

    def train_dataloader(self) -> SafeDataLoader:
        self.train_loader = SafeDataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader

    def val_dataloader(self) -> SafeDataLoader:
        self.val_loader = SafeDataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self) -> SafeDataLoader:
        self.test_loader = SafeDataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader

    def on_train_end(self) -> None:
        # dump consumption
        self.tracker_train.stop()
        training_kwh = self.tracker_train._total_energy.kWh
        self.logger.log_metrics(  # type: ignore[union-attr]  # logger initialized before training
            {"/train/tot_energy_kWh": torch.tensor(float(training_kwh))},
        )

    def on_test_start(self) -> None:
        """Test開始時の初期化処理.

        cSEBBsのチューニングに必要なvalidationスコアを収集するため、
        test実行前に一度validationパスを実行

        処理の流れ:
            1. MAESTROのメタデータ（ground truth, audio durations）を読み込み
            2. Validationパスを実行してスコアを収集
            3. 収集したスコアはval_buffer_sed_scores_eval_studentに保存
            4. test_step内でこのバッファを使ってcSEBBsをチューニング
        """
        if self.sebbs_enabled:
            # MAESTROのground truthとaudio durationsを読み込み
            # 結果は self._maestro_ground_truth, self._maestro_audio_durations に保存
            print("Loading MAESTRO audio durations and ground truth for test...")
            self.load_maestro_audio_durations_and_gt()

            # Validation dataが存在しない場合はスキップ
            if self.valid_data is None:
                print("\n" + "=" * 70)
                print("WARNING: Validation data not available, skipping cSEBBs tuning")
                print("cSEBBs will use default parameters without validation-based tuning")
                print("=" * 70)
                # 空のバッファを初期化（test_stepでエラーが出ないように）
                self.val_tune_sebbs_student = {}
                self.val_tune_sebbs_teacher = {}
            else:
                # Validationパスを実行. 理由は以下:
                #   - cSEBBsのハイパーパラメータ（step_filter_length, merge_thresholds）を
                #     validation setでチューニングする必要がある
                #   - trainer.test()はbest checkpointをロード済みなので、
                #     ここでvalidationを実行すれば最良モデルでのスコアが得られる
                #   - これらのスコアをtest_step内でcSEBBsチューニングに使用
                print("\n" + "=" * 70)
                print("Running validation pass to collect scores for cSEBBs tuning")
                print("=" * 70)

                # バッファを初期化（以前のスコアをクリア）
                self.val_tune_sebbs_student = {}
                self.val_tune_sebbs_teacher = {}

                # Validationデータローダーを取得
                val_loader = self.val_dataloader()

                # モデルを評価モードに設定（Dropout等を無効化）
                self.sed_student.eval()
                self.sed_teacher.eval()

                # Validationパスを実行してスコアを収集
                # torch.no_grad()で勾配計算を無効化（メモリ節約＆高速化）
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        # バッチデータをアンパック
                        audio, labels, padded_indxs, filenames, embeddings, valid_class_mask = batch

                        # 全テンソルを現在のデバイス（GPU/CPU）に移動
                        audio = audio.to(self.device)
                        labels = labels.to(self.device)
                        embeddings = embeddings.to(self.device)
                        valid_class_mask = valid_class_mask.to(self.device)

                        # デバイス移動後のバッチを再構築
                        moved_batch = (
                            audio,
                            labels,
                            padded_indxs,
                            filenames,
                            embeddings,
                            valid_class_mask,
                        )

                        # validation_stepを実行
                        # この中でval_buffer_sed_scores_eval_studentにスコアが蓄積される
                        self.validation_step(moved_batch, batch_idx)

                print("\n✓ Validation pass complete")
                print(f"  Collected scores for {len(self.val_tune_sebbs_student)} clips")

        if self.evaluation:
            os.makedirs(os.path.join(self.exp_dir, "codecarbon"), exist_ok=True)
            self.tracker_eval = OfflineEmissionsTracker(
                "DCASE Task 4 SED EVALUATION",
                output_dir=os.path.join(self.exp_dir, "codecarbon"),
                output_file="emissions_basleline_eval.csv",
                log_level="warning",
                country_iso_code="FRA",
                gpu_ids=[torch.cuda.current_device()],
            )
            self.tracker_eval.start()
        else:
            os.makedirs(os.path.join(self.exp_dir, "codecarbon"), exist_ok=True)
            self.tracker_devtest = OfflineEmissionsTracker(
                "DCASE Task 4 SED DEVTEST",
                output_dir=os.path.join(self.exp_dir, "codecarbon"),
                output_file="emissions_baseline_test.csv",
                log_level="warning",
                country_iso_code="FRA",
                gpu_ids=[torch.cuda.current_device()],
            )

            self.tracker_devtest.start()

    def load_maestro_audio_durations_and_gt(
        self,
    ) -> tuple[dict[str, float], dict[str, list[tuple[float, float, str]]]]:
        """MAESTROのaudio durationsとground truthを読み込む。

        注: validation時と同じデータソース(real_maestro_train_tsv)を使用することで、
        cSEBBsチューニング時のスコアとground truthのマッチングを可能にする。
        """
        # --- 変更: trainデータを使用してvalidationスコアとマッチング ---
        gt_tsv_path = self.hparams["data"]["real_maestro_train_tsv"]

        # durationsの読み込み（trainセット用のdurationsファイル）
        # 設定ファイルにreal_maestro_train_durがあればそれを使用、なければフォールバック
        durations_path = self.hparams["data"].get(
            "real_maestro_train_dur",
            self.hparams["data"]["real_maestro_val_dur"],  # フォールバック
        )

        maestro_audio_durations = sed_scores_eval.io.read_audio_durations(durations_path)

        # --- 2. ground truth tsv の読み込みとフィルタ ---
        maestro_ground_truth_clips = pd.read_csv(gt_tsv_path, sep="\t")
        # 元ファイルの filename カラムが "xxxx.wav" のようになっている前提
        # ここで clip_id の仕様に合わせて切る（例: remove .wav）
        maestro_ground_truth_clips["file_id"] = maestro_ground_truth_clips["filename"].apply(
            lambda x: x[:-4] if isinstance(x, str) and x.lower().endswith(".wav") else x,
        )

        # confidence とラベルフィルタ
        maestro_ground_truth_clips = maestro_ground_truth_clips[
            maestro_ground_truth_clips.confidence > 0.5
        ]
        maestro_ground_truth_clips = maestro_ground_truth_clips[
            maestro_ground_truth_clips.event_label.isin(classes_labels_maestro_real_eval)
        ]

        # --- 3. read_ground_truth_events に通す（返り値が dict になる前提） ---
        maestro_ground_truth_clips = sed_scores_eval.io.read_ground_truth_events(
            maestro_ground_truth_clips,
        )

        # --- 4. マッピングを clip_id の集合に揃える ---
        maestro_ground_truth = _merge_maestro_ground_truth(
            maestro_ground_truth_clips,
        )  # 既存関数使用
        # maestro_audio_durations のキーが file_id と一致するか確かめる
        # ここで、該当する file_id のみ抽出
        maestro_audio_durations_filtered = {
            file_id: maestro_audio_durations[file_id]
            for file_id in maestro_ground_truth.keys()
            if file_id in maestro_audio_durations
        }

        missing = set(maestro_ground_truth.keys()) - set(maestro_audio_durations_filtered.keys())
        if missing:
            warnings.warn(
                f"maestro_audio_durations missing for {len(missing)} files. Examples: {list(missing)[:5]}. Using fallback for those clips.",
            )
            # 欠損は許容するが、fallback 用に空のエントリは作らない（fallbackはget()で対応）

        # キャッシュしておく（後で呼び出し直すときのため）
        self._maestro_audio_durations = maestro_audio_durations_filtered
        self._maestro_ground_truth = maestro_ground_truth
        self._maestro_ground_truth_clips = maestro_ground_truth_clips

        return maestro_audio_durations_filtered, maestro_ground_truth


def select_best_auroc(
    csebbs: list,
    ground_truth: dict,
    audio_durations: dict,
    audio_ids=None,
    segment_length=1.0,
    **kwargs,
):
    """cSEBBsのハイパーパラメータチューニングでAUROCを最大化する選択関数.

    csebbs.tune()からのインターフェースに適合し、
    内部でsed_scores_eval.segment_based.aurocを使用する

    Args:
        csebbs: [(CSEBBsPredictor, sebbs_dict), ...]のリスト
        ground_truth: {audio_id: [(onset, offset, label), ...]}
        audio_durations: {audio_id: duration}
        audio_ids: 評価対象のaudio_idリスト（オプション）
        segment_length: セグメント長（秒）
        **kwargs: aurocに渡す追加引数

    Returns:
        (best_predictor, best_scores): 最良のpredictor、および各クラスのAUROC値

    """
    if audio_ids is not None:
        ground_truth = {aid: ground_truth[aid] for aid in audio_ids}
        audio_durations = {aid: audio_durations[aid] for aid in audio_ids}

    best_auroc = -1
    best_predictor = None
    best_auroc_values = None

    for predictor, sebbs_dict in csebbs:
        # SEBBsをsed_scores_eval形式に変換
        if audio_ids is not None:
            sebbs_dict = {aid: sebbs_dict[aid] for aid in audio_ids}

        scores = sed_scores_from_sebbs(
            sebbs_dict,
            sound_classes=predictor.sound_classes,
            audio_duration=audio_durations,
        )

        # AUROC計算
        auroc_values, _ = sed_scores_eval.segment_based.auroc(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            segment_length=segment_length,
            **kwargs,
        )

        # 平均AUROCで評価
        mean_auroc = np.mean(list(auroc_values.values()))

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_predictor = predictor
            best_auroc_values = auroc_values

    return best_predictor, best_auroc_values


def _merge_maestro_ground_truth(clip_ground_truth):
    ground_truth = defaultdict(list)
    for clip_id in clip_ground_truth:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit("-", maxsplit=2)
        clip_onset_time = int(clip_onset_time) // 100
        ground_truth[file_id].extend(
            [
                (
                    clip_onset_time + event_onset_time,
                    clip_onset_time + event_offset_time,
                    event_class,
                )
                for event_onset_time, event_offset_time, event_class in clip_ground_truth[clip_id]
            ],
        )
    return _merge_overlapping_events(ground_truth)


def _merge_overlapping_events(ground_truth_events):
    for clip_id, events in ground_truth_events.items():
        per_class_events = defaultdict(list)
        for event in events:
            per_class_events[event[2]].append(event)
        ground_truth_events[clip_id] = []
        for event_class, events in per_class_events.items():
            events = sorted(events)
            merged_events = []
            current_offset = -1e6
            for event in events:
                if event[0] > current_offset:
                    merged_events.append(list(event))
                else:
                    merged_events[-1][1] = max(current_offset, event[1])
                current_offset = merged_events[-1][1]
            ground_truth_events[clip_id].extend(merged_events)
    return ground_truth_events


def _get_segment_scores_and_overlap_add(
    frame_scores,
    audio_durations,
    event_classes,
    segment_length=1.0,
):
    """>>> event_classes = ['a', 'b', 'c']
    >>> audio_durations = {'f1': 201.6, 'f2':133.1, 'f3':326}
    >>> frame_scores = {\
        f'{file_id}-{int(100*onset)}-{int(100*(onset+10.))}': create_score_dataframe(np.random.rand(156,3), np.arange(157.)*0.064, event_classes)\
        for file_id in audio_durations for onset in range(int((audio_durations[file_id]-9.)))\
    }
    >>> frame_scores.keys()
    >>> seg_scores = _get_segment_scores_and_overlap_add(frame_scores, audio_durations, event_classes, segment_length=1.)
    >>> [(key, validate_score_dataframe(value)[0][-3:]) for key, value in seg_scores.items()]
    """
    segment_scores_file = {}
    summand_count = {}
    keys = ["onset", "offset"] + event_classes
    for clip_id in frame_scores:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit("-", maxsplit=2)
        clip_onset_time = float(clip_onset_time) / 100
        clip_offset_time = float(clip_offset_time) / 100
        if file_id not in segment_scores_file:
            segment_scores_file[file_id] = np.zeros(
                (ceil(audio_durations[file_id] / segment_length), len(event_classes)),
            )
            summand_count[file_id] = np.zeros_like(segment_scores_file[file_id])
        segment_scores_clip = _get_segment_scores(
            frame_scores[clip_id][keys],
            clip_length=(clip_offset_time - clip_onset_time),
            segment_length=1.0,
        )[event_classes].to_numpy()
        seg_idx = int(clip_onset_time // segment_length)
        segment_scores_file[file_id][seg_idx : seg_idx + len(segment_scores_clip)] += (
            segment_scores_clip
        )
        summand_count[file_id][seg_idx : seg_idx + len(segment_scores_clip)] += 1
    return {
        file_id: create_score_dataframe(
            segment_scores_file[file_id] / np.maximum(summand_count[file_id], 1),
            np.minimum(
                np.arange(
                    0.0,
                    audio_durations[file_id] + segment_length,
                    segment_length,
                ),
                audio_durations[file_id],
            ),
            event_classes,
        )
        for file_id in segment_scores_file
    }


def _get_segment_scores(scores_df, clip_length, segment_length=1.0):
    """>>> scores_arr = np.random.rand(156,3)
    >>> timestamps = np.arange(157) * 0.064
    >>> event_classes = ["a", "b", "c"]
    >>> scores_df = create_score_dataframe(scores_arr, timestamps, event_classes)
    >>> seg_scores_df = _get_segment_scores(scores_df, clip_length=10.0, segment_length=1.0)
    """
    frame_timestamps, event_classes = validate_score_dataframe(scores_df)
    scores_arr = scores_df[event_classes].to_numpy()
    segment_scores = []
    segment_timestamps = []
    seg_onset_idx = 0
    seg_offset_idx = 0
    for seg_onset in np.arange(0.0, clip_length, segment_length):
        seg_offset = seg_onset + segment_length
        while frame_timestamps[seg_onset_idx + 1] <= seg_onset:
            seg_onset_idx += 1
        while seg_offset_idx < len(scores_arr) and frame_timestamps[seg_offset_idx] < seg_offset:
            seg_offset_idx += 1
        seg_weights = np.minimum(
            frame_timestamps[seg_onset_idx + 1 : seg_offset_idx + 1],
            seg_offset,
        ) - np.maximum(frame_timestamps[seg_onset_idx:seg_offset_idx], seg_onset)
        segment_scores.append(
            (seg_weights[:, None] * scores_arr[seg_onset_idx:seg_offset_idx]).sum(0)
            / seg_weights.sum(),
        )
        segment_timestamps.append(seg_onset)
    segment_timestamps.append(clip_length)
    return create_score_dataframe(
        np.array(segment_scores),
        np.array(segment_timestamps),
        event_classes,
    )


def get_sebbs(self, scores_all_classes, model_type="student"):
    """Apply cSEBBs post-processing to both DESED and MAESTRO classes.

    cSEBBs (change-point based Sound Event Bounding Boxes)は、
    フレームレベルの事後確率スコアから以下の処理によりイベント候補を生成:

    1. 変化点検出: ステップフィルタを用いてスコアの変化点を検出
    2. セグメント生成: 変化点を境界として候補セグメントを生成
    3. セグメントマージ: 類似スコアを持つ隣接セグメントを統合
    4. 信頼度計算: 各セグメントの平均スコアを信頼度として出力

    この関数では、DESEDとMAESTROの両クラスに対してcSEBBsを適用し、
    結果を統合して最終的なスコアDataFrameを生成する。

    Args:
        self: SEDTask4 instance (csebbs_predictor_desed, csebbs_predictor_maestroを保持)
        scores_all_classes: Dictionary of score DataFrames for all clips
                           各clipのDataFrameは全クラスのスコアを含む
                           形式: {clip_id: pd.DataFrame(onset, offset, class1, class2, ...)}
        model_type: 'student' or 'teacher'. モデルタイプを指定して適切なpredictorを選択

    Returns:
        sed_scores_postprocessed: Dictionary of post-processed score DataFrames
                                  cSEBBsにより生成されたイベント候補のスコア
                                  形式: {clip_id: pd.DataFrame(onset, offset, class1, class2, ...)}

    処理フロー:
        1. DESED/MAESTROクラスのスコアを個別に抽出
        2. 各データセットに最適化されたcSEBBs predictorで処理
        3. 生成されたイベント候補を統合
        4. sed_scores_eval形式のDataFrameに変換

    """
    # モデルタイプに応じたpredictorを選択
    if model_type == "teacher":
        csebbs_predictor_desed = self.csebbs_predictor_desed_teacher
        csebbs_predictor_maestro = self.csebbs_predictor_maestro_teacher
    else:
        csebbs_predictor_desed = self.csebbs_predictor_desed
        csebbs_predictor_maestro = self.csebbs_predictor_maestro

    # ステップ1: DESEDクラスに対するcSEBBs適用
    # DESEDは家庭内音（食器、掃除機、猫など）の10クラス
    desed_classes = list(classes_labels_desed.keys())
    keys_desed = ["onset", "offset"] + sorted(desed_classes)

    # 全クラスのスコアからDESEDクラスのみを抽出
    scores_desed_classes = {
        clip_id: scores_all_classes[clip_id][keys_desed] for clip_id in scores_all_classes.keys()
    }

    # cSEBBsでDESEDデータセットのスコアを後処理
    # 戻り値: {clip_id: [(onset, offset, class_name, confidence), ...]}
    csebbs_desed_events = csebbs_predictor_desed.predict(
        scores_desed_classes,
    )

    # ステップ2: MAESTROクラスに対するcSEBBs適用
    # MAESTROは都市音・屋内音（足音、会話、車など）の17クラス
    maestro_classes = sorted(classes_labels_maestro_real_eval)
    keys_maestro = ["onset", "offset"] + sorted(maestro_classes)

    # 全クラスのスコアからMAESTROクラスのみを抽出
    scores_maestro_classes = {
        clip_id: scores_all_classes[clip_id][keys_maestro] for clip_id in scores_all_classes.keys()
    }

    # cSEBBsでMAESTROデータセットのスコアを後処理
    # DESEDとは異なる音響特性に最適化されたパラメータを使用
    # 戻り値: {clip_id: [(onset, offset, class_name, confidence), ...]}
    csebbs_maestro_events = csebbs_predictor_maestro.predict(
        scores_maestro_classes,
    )

    # ステップ3: イベント候補の統合
    # DESEDとMAESTROの全クラスのリスト（重複なし、ソート済み）
    all_sound_classes = sorted(list(set(desed_classes + maestro_classes)))

    # 両データセットのclip IDを統合
    all_clip_ids = set(csebbs_desed_events.keys()) | set(csebbs_maestro_events.keys())

    # clip毎にDESEDとMAESTROのイベントを結合
    sebbs_all_events = {}
    for clip_id in all_clip_ids:
        # DESEDのイベントリストを取得（存在しない場合は空リスト）
        desed_events = csebbs_desed_events.get(clip_id, [])

        # MAESTROのイベントリストを取得（存在しない場合は空リスト）
        maestro_events = csebbs_maestro_events.get(clip_id, [])

        # イベントリストを結合してonset時間順にソート
        # 各イベント: (onset, offset, class_name, confidence)
        sebbs_all_events[clip_id] = sorted(
            desed_events + maestro_events,
            key=lambda x: x[0],  # onset時間でソート
        )

    # ステップ4: sed_scores_eval形式への変換
    # イベント候補リストからsed_scores_eval形式のDataFrameに変換
    # - 各イベントの時間範囲で該当クラスのスコアを設定
    # - イベントが存在しない時間・クラスは0.0で埋める
    # - 評価ツール(sed_scores_eval)で直接使用可能な形式
    sed_scores_postprocessed = sed_scores_from_sebbs(
        sebbs_all_events,
        sound_classes=all_sound_classes,
        fill_value=0.0,  # イベントが存在しない箇所は0.0
    )

    return sed_scores_postprocessed
