import os
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sed_scores_eval
import torch
import torchmetrics
from codecarbon import OfflineEmissionsTracker
from sed_scores_eval.base_modules.scores import (create_score_dataframe,
                                                 validate_score_dataframe)
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, MFCC

from desed_task.data_augm import mixup, cutmix
from desed_task.utils.postprocess import ClassWiseMedianFilter
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1, compute_psds_from_operating_points,
    compute_psds_from_scores)
from desed_task.utils.scaler import TorchScaler

from .classes_dict import (classes_labels_desed, classes_labels_maestro_real,
                           classes_labels_maestro_real_eval)
from .utils import batched_decode_preds, log_sedeval_metrics


from pathlib import Path

from sebbs.sebbs import csebbs
from sebbs.sebbs.utils import sed_scores_from_sebbs
from sebbs.scripts.utils  import get_segment_scores


# データ不足の対策
from torch.utils.data.dataloader import DataLoader, default_collate
import wandb
import torch.nn.functional as F


try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    print("Warning: sklearn.mixture.GaussianMixture not found. SAFT (GMM) may be disabled.")

PROJECT_NAME = "SED-pl-noise"


scores_path = "path/to/your/validation/scores" # モデルが出力したスコア
ground_truth_path = "path/to/your/validation/ground_truth.tsv"
durations_path = "path/to/your/validation/audio_durations.tsv"


class _NoneSafeIterator:
    """
    DataLoaderから返されるNoneバッチを内部でスキップするイテレータのラッパー。
    """
    def __init__(self, dataloader_iter):
        self.dataloader_iter = dataloader_iter

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # 元のイテレータから次のバッチを取得
            batch = next(self.dataloader_iter)
            # バッチがNoneでなければ、それを返す
            if batch is not None:
                return batch
            # バッチがNoneなら、ループを続けて次の有効なバッチを探す
            print("Skipping a batch that was None internally.")


class SafeCollate:
    """
    データセットから返される None 値をフィルタリングする collate_fn。
    フィルタリング後にバッチが空になった場合は None を返す。
    """
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        if not batch:
            # バッチが空ならNoneを返す（このNoneは_NoneSafeIteratorで捕捉される>）
            return None

        return default_collate(batch)


class SafeDataLoader(DataLoader):
    """
    Noneを返す可能性があるバッチを自動的にスキップするDataLoader。
    PyTorch Lightningなどのフレームワークで安全に使用できる。
    """
    def __init__(self, *args, **kwargs):
        # ユーザーがcollate_fnを指定していない場合のみ、SafeCollateを設定
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = SafeCollate()
        super().__init__(*args, **kwargs)

    def __iter__(self):
        # DataLoaderのデフォルトのイテレータを取得
        dataloader_iter = super().__iter__()
        # Noneをスキップする機能を持つラッパーで包んで返す
        return _NoneSafeIterator(dataloader_iter)


class SEDTask4(pl.LightningModule):
    """Pytorch lightning module for the SED 2021 baseline
    Args:
        hparams: dict, the dictionary to be used for the current experiment/
        encoder: ManyHotEncoder object, object to encode and decode labels.
        sed_student: torch.Module, the student model to be trained. The teacher model will be
        opt: torch.optimizer.Optimizer object, the optimizer to be used
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        scheduler: BaseScheduler subclass object, the scheduler to be used.
                   This is used to apply ramp-up during training for example.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
            to test the code runs.
    """

    def __init__(
        self,
        hparams,
        encoder,
        sed_student,
        pretrained_model,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False,
        sed_teacher=None,
    ):
        super(SEDTask4, self).__init__()
        self.hparams.update(hparams)

        if self.hparams["net"]["use_wandb"]:
            self._init_wandb_project()
        else:
            # wandbを使わない場合はNoneに設定
            self._wandb_checkpoint_dir = None

        self.encoder = encoder
        self.sed_student = sed_student
        self.median_filter = ClassWiseMedianFilter(self.hparams["net"]["median_filter"])


        # ===========================================================
        # SAT-SED (Self-Adaptive Thresholding) Parameters
        # ===========================================================
        self.sat_enabled = self.hparams.get("sat", {}).get("enabled", False)
        
        if self.sat_enabled:
            # 継続的な警告を避けるため、一度だけ警告
            # if not self.hparams["training"]["self_sup_loss"] == "bce":
            #      warnings.warn(f"SAT-SED is enabled, but self_sup_loss is '{self.hparams['training']['self_sup_loss']}'. "
            #                    f"SAT-SED pseudo-labeling is designed to work with BCE loss (self.selfsup_loss).")
            
            self.K = len(self.encoder.labels)  # クラス数 (K)
            # EMA係数 (lambda)
            self.sat_lambda = self.hparams.get("sat", {}).get("lambda", 0.999) 
            # 疑似ラベル損失の重み (w_u)
            self.sat_w_u = self.hparams.get("sat", {}).get("w_u", 0.5) 
            # ウォームアップエポック
            # self.sat_warmup_epochs = self.hparams.get("sat", {}).get("warmup_epochs", 0)
            # self.gmm_fixed = self.hparams.get("sat", {}).get("gmm_fixed", False)

            self.cutmix_alpha = self.hparams.get("sat", {}).get("cutmix_alpha", 1.0)

            # SACT (Clip) 用バッファ
            # register_buffer は、モデルの state_dict に含まれるが、optimizerの対象にならない
            # tau_s (Eq 2)
            self.register_buffer("global_clip_threshold", torch.tensor(1.0 / self.K)) 
            # p_tilde_s (Eq 3)
            self.register_buffer("local_clip_probabilities", torch.full((self.K,), 1.0 / self.K)) 
            
            # SAFT (Frame) GMMパラメータ (Eq 7)
            # GMMをsklearnで使うためのimportチェック
            try:
                # __init__時点で GMM がインポート可能か確認
                from sklearn.mixture import GaussianMixture
                self.gmm_imported = True
            except ImportError:
                self.gmm_imported = False
                # 学習開始時に一度だけ警告
                warnings.warn("sklearn.mixture.GaussianMixture not found. SAFT (GMM fitting) will be disabled.")
        # ===========================================================



        # CMT parameters
        self.cmt_enabled = self.hparams.get("cmt", {}).get("enabled", False)
        self.cmt_phi_clip = self.hparams.get("cmt", {}).get("phi_clip", 0.5)
        self.cmt_phi_frame = self.hparams.get("cmt", {}).get("phi_frame", 0.5)
        self.cmt_scale = self.hparams.get("cmt", {}).get("scale", False)
        self.cmt_warmup_epochs = int(self.hparams.get("cmt", {}).get("warmup_epochs", 50))


        # cSEBBs param
        self.sebbs_enabled = self.hparams.get("sebbs", {}).get("enabled", False)



        if self.hparams["pretrained"]["e2e"]:
            self.pretrained_model = pretrained_model
        # else we use pre-computed embeddings from hdf5

        if sed_teacher is None:
            self.sed_teacher = deepcopy(sed_student)
        else:
            self.sed_teacher = sed_teacher
        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        self.evaluation = evaluation

        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
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
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = (
            torchmetrics.classification.f_beta.MultilabelF1Score(
                len(self.encoder.labels),
                average="macro",
            )
        )
        self.get_weak_teacher_f1_seg_macro = (
            torchmetrics.classification.f_beta.MultilabelF1Score(
                len(self.encoder.labels), average="macro"
            )
        )

        self.scaler = self._init_scaler()
        # buffer for event based scores which we compute using sed-eval

        self.val_buffer_sed_scores_eval_student = {}
        self.val_buffer_sed_scores_eval_teacher = {}

        self.val_tune_sebbs_student = {}
        self.val_tune_sebbs_teacher = {}

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_buffer_psds_eval_student = {
            k: pd.DataFrame() for k in test_thresholds
        }
        self.test_buffer_psds_eval_teacher = {
            k: pd.DataFrame() for k in test_thresholds
        }
        self.test_buffer_sed_scores_eval_student = {}
        self.test_buffer_sed_scores_eval_teacher = {}
        self.test_buffer_sed_scores_eval_unprocessed_student = {}
        self.test_buffer_sed_scores_eval_unprocessed_teacher = {}
        self.test_buffer_detections_thres05_student = pd.DataFrame()
        self.test_buffer_detections_thres05_teacher = pd.DataFrame()

    _exp_dir = None

    @property
    def exp_dir(self):
        if self._exp_dir is None:
            try:
                self._exp_dir = self.logger.log_dir
            except Exception as e:
                self._exp_dir = self.hparams["log_dir"]
        return self._exp_dir

    def log(self, name, value, *args, **kwargs):
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

    def log_dict(self, dictionary, *args, **kwargs):
        """Mirror a dictionary of metrics to wandb in addition to Lightning's log_dict."""
        res = super(SEDTask4, self).log_dict(dictionary, *args, **kwargs)
        try:
            self._maybe_wandb_log(dictionary)
        except Exception:
            pass
        return res

    def _maybe_wandb_log(self, log_dict):
        """Safely log a dict to wandb if enabled and initialized.

        Converts torch tensors and numpy arrays to Python scalars or lists.
        If wandb isn't active or hparams disable it, this is a no-op.
        """
        try:
            if not self.hparams.get("net", {}).get("use_wandb"):
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
            step = int(getattr(self, "global_step", None))
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


    def _init_wandb_project(self):
        wandb.init(project=PROJECT_NAME, name=self.hparams["net"]["wandb_dir"])

        # wandb runディレクトリ内にcheckpointsディレクトリを作成
        # 結果: wandb/run-20250102_123456-abcd1234/checkpoints/
        if wandb.run is not None:
            self._wandb_checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
            os.makedirs(self._wandb_checkpoint_dir, exist_ok=True)
            print(f"Checkpoint directory: {self._wandb_checkpoint_dir}")
        else:
            self._wandb_checkpoint_dir = None

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
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
        # to_ignore = []
        to_ignore = [
            ".*Trying to infer the `batch_size` from an ambiguous collection.*",
            ".*invalid value encountered in divide*",
            ".*mean of empty slice*",
            ".*self.log*",
        ]
        for message in to_ignore:
            warnings.filterwarnings("ignore", message)

    def update_ema(self, alpha, global_step, model, ema_model):
        """Update teacher model parameters

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


        #--- ユークリッド距離とコサイン類似度 ---#
            # あまり参考にならなかったのでコメントアウト
        # # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (global_step + 1), alpha)

        # with torch.no_grad():
        #     # 累積用の変数を初期化 (すべてGPU上の0次元Tensorとして扱うと効率的ですが、ここではわかりやすさ優先でスカラ計算します)
        #     diff_sq_sum = 0.0   # ユークリッド距離用: sum((S - T)^2)
        #     dot_prod_sum = 0.0  # コサイン類似度分子: sum(S * T)
        #     norm_s_sq = 0.0     # コサイン類似度分母: sum(S^2)
        #     norm_t_sq = 0.0     # コサイン類似度分母: sum(T^2)

        #     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                
        #         # --- 1. 計算パート ---
                
        #         # ユークリッド距離用: 差の二乗
        #         # (Mean Teacherでは差が非常に小さいので、展開式 a^2+b^2-2ab ではなく、直接差を取るほうが精度が良いです)
        #         diff_sq_sum += torch.sum((ema_param - param) ** 2)
                
        #         # コサイン類似度用: 内積とそれぞれのノルム二乗
        #         dot_prod_sum += torch.sum(ema_param * param)
        #         norm_s_sq += torch.sum(param ** 2)
        #         norm_t_sq += torch.sum(ema_param ** 2)

        #         # --- 2. 更新パート (EMA) ---
                
        #         # 教師モデル(ema_param)を更新
        #         ema_param.mul_(alpha).add_(param, alpha=1 - alpha)

        #     # --- 3. 最終計算パート (CPUへ転送) ---
            
        #     # GPU上のTensorから値を取り出す (.item())
        #     diff_sq_sum = diff_sq_sum.item() if torch.is_tensor(diff_sq_sum) else diff_sq_sum
        #     dot_prod_sum = dot_prod_sum.item() if torch.is_tensor(dot_prod_sum) else dot_prod_sum
        #     norm_s_sq = norm_s_sq.item() if torch.is_tensor(norm_s_sq) else norm_s_sq
        #     norm_t_sq = norm_t_sq.item() if torch.is_tensor(norm_t_sq) else norm_t_sq

        #     # ユークリッド距離
        #     euclidean_dist = diff_sq_sum ** 0.5
            
        #     # コサイン類似度 (分母が0にならないよう微小値を足すのが一般的ですが、学習済みモデルならまず0にはなりません)
        #     denominator = (norm_s_sq ** 0.5) * (norm_t_sq ** 0.5)
        #     cosine_sim = dot_prod_sum / denominator if denominator > 0 else 0.0

        # return euclidean_dist, cosine_sim

    def _init_scaler(self):
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
        elif self.hparams["scaler"]["statistic"] == "dataset":
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
                        self.hparams["scaler"]["savepath"]
                    )
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
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
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

    #CMT
    def apply_cmt_postprocessing(self, y_w, y_s, phi_clip=0.5, phi_frame=0.5):
        """Apply Confident Mean Teacher postprocessing to teacher predictions.
        (Input shape: batch, classes, frames)
        OK
        """
        # strong, weak
        # y_s: (batch, classes, frames), y_w: (batch, classes)

        # Step 1: Apply clip-level threshold
        y_tilde_w = (y_w > phi_clip).float()
        y_w_expanded = y_tilde_w.unsqueeze(-1).expand_as(y_s) # クリップとフレームでサイズが合わないので拡張

        # Step 2 & 3: Apply two-stage thresholding, frame-level
        y_s_temp = y_s.clone()
        # クリップ予測の二値化とフレーム予測の二値化をかける クリップの予測が0の時,フレームも0とする
        y_s_binary = y_w_expanded * ((y_s_temp > phi_frame).float())

        # Step 4: Apply class-wise constraint and median filter
        # Expand y_tilde_w to (batch, classes, 1) for broadcasting

        y_tilde_s = []

        for i in range(y_s.shape[0]):
            constrained_s = y_s_binary[i]
            constrained_s = constrained_s.transpose(0, 1).detach().cpu().numpy()
            # frames, classes
            filtered = self.median_filter(constrained_s)
            y_tilde_s.append(filtered) 

        # 形状を整える
        original_device = y_s.device
        y_tilde_s = np.stack(y_tilde_s, axis=0) # batch, frames, classes
        y_tilde_s = torch.from_numpy(y_tilde_s).to(original_device)
        y_tilde_s = y_tilde_s.transpose(1, 2) # (batch, frames, classes) -> (batch, classes, frames)

        return y_tilde_w, y_tilde_s

    def compute_cmt_confidence_weights(self, y_w, y_s, y_tilde_w, y_tilde_s):
        """Compute confidence weights for CMT consistency loss.
        (Input shape: batch, classes, frames)
        OK
        """
        # クリップレベルの信頼度重みを算出
        # 擬似ラベルが教師信号として有効なら,予測値を重みとする
        c_w = y_w * y_tilde_w
        y_w_expanded = y_w.unsqueeze(-1).expand_as(y_s)
        c_s = y_s * y_w_expanded * y_tilde_s
        # c_w = y_w * (y_tilde_w == 1).float()
        # # Expand y_w to match dimensions of y_s: (batch, classes, frames)
        # y_w_expanded = y_w.unsqueeze(-1).expand_as(y_s)
        # # クリップごとの確率とフレームを束ねた確率を乗算
        # c_s = y_s * y_w_expanded * (y_tilde_s == 1).float()

        return c_w, c_s



    def compute_cmt_consistency_loss(self, student_w, student_s, teacher_pseudo_w, teacher_pseudo_s, 
                                   confidence_w, confidence_s):
        """Compute CMT consistency loss with confidence weighting.
        
        Args:
            student_w: torch.Tensor, student weak predictions [batch_size, n_classes]
            student_s: torch.Tensor, student strong predictions [batch, classes, frames]
            teacher_pseudo_w: torch.Tensor, teacher pseudo weak labels [batch_size, n_classes]
            teacher_pseudo_s: torch.Tensor, teacher pseudo strong labels [batch_size, n_classes, n_frames] [batch_size, n_frames, n_classes]
            confidence_w: torch.Tensor, confidence weights for weak [batch_size, n_classes]
            confidence_s: torch.Tensor, confidence weights for strong [batch_size, n_classes, n_frames]
            
        Returns:
            loss_w_con: torch.Tensor, weighted weak consistency loss
            loss_s_con: torch.Tensor, weighted strong consistency loss
        """

        # Weak consistency loss: ℓ_w,con = (1/|K|) ∑_{k=1}^K c_w(k) · BCE(ỹ_w(k), f_{θ_s}(x)_w(k))
        bce_w = torch.nn.functional.binary_cross_entropy(
            student_w, # 生徒モデルのクリップ予測
            teacher_pseudo_w, # 教師モデルのクリップ予測
            reduction='none' # 後で平均を取るので,縮約時には計算しない
        )
        weighted_bce_w = confidence_w * bce_w # 信頼度重みをbce損失にかける
        
        # 小さい確率を反映する
        # confidence_w_neg = 1 - confidence_w
        # loss_pos = confidence_w * (teacher_pseudo_w * bce_w) 
        # loss_neg = confidence_w_neg * ((1 - teacher_pseudo_w) * bce_w) 
        # weighted_bce_w = (loss_pos + loss_neg).mean()


        if self.cmt_scale:
            mask_w = (confidence_w > 0).float()
            num_nonzero = mask_w.sum().clamp(min=1.0)
            loss_w_con = weighted_bce_w.sum() / num_nonzero  # 非ゼロサンプルで割る
        else:
            loss_w_con = weighted_bce_w.mean() # 平均を取る
        
        # Strong consistency loss: ℓ_s,con = (1/|Ω|) ∑_{t,k} c_s(t,k) · BCE(ỹ_s(t,k), f_{θ_s}(x)_s(t,k))
        bce_s = torch.nn.functional.binary_cross_entropy(
            student_s, # 生徒モデルのフレーム予測
            teacher_pseudo_s,  # 教師モデルのフレーム予測
            reduction='none'
        )

        weighted_bce_s = confidence_s * bce_s # 信頼度重みをbce損失にかける

        if self.cmt_scale:
            mask_s = (confidence_s > 0).float()
            num_nonzero = mask_s.sum().clamp(min=1.0)
            loss_s_con = weighted_bce_s.sum() / num_nonzero  # 非ゼロサンプルで割る
        else:
            loss_s_con = weighted_bce_s.mean() # 平均を取る
        
        return loss_w_con, loss_s_con

    def detect(self, mel_feats, model, embeddings=None, **kwargs):
        if embeddings is None:
            return model(self.scaler(self.take_log(mel_feats)), **kwargs)
        else:
            return model(
                self.scaler(self.take_log(mel_feats)), embeddings=embeddings, **kwargs
            )

    def apply_mixup(self, features, embeddings, labels, start_indx, stop_indx):
        # made a dedicated method as we need to apply mixup only
        # within each dataset that has the same classes
        mixup_type = self.hparams["training"].get("mixup")
        batch_num = features.shape[0]
        current_mask = torch.zeros(batch_num).to(features).bool()
        current_mask[start_indx:stop_indx] = 1
        features[current_mask], labels[current_mask] = mixup(
            features[current_mask], labels[current_mask], mixup_label_type=mixup_type
        )

        if embeddings is not None:
            # apply mixup also on embeddings
            embeddings[current_mask], labels[current_mask] = mixup(
                embeddings[current_mask],
                labels[current_mask],
                mixup_label_type=mixup_type,
            )

        return features, embeddings, labels

    def _unpack_batch(self, batch):

        if not self.hparams["pretrained"]["e2e"]:
            return batch
        else:
            # untested
            raise NotImplementedError
            # we train e2e
            if len(batch) > 3:
                audio, labels, padded_indxs, ast_feats = batch
                pretrained_input = ast_feats
            else:
                audio, labels, padded_indxs = batch
                pretrained_input = audio

    def training_step(self, batch, batch_indx):
        """Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch: torch.Tensor, batch input tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.
        """

        audio, labels, padded_indxs, embeddings, valid_class_mask = self._unpack_batch(
            batch
        )

        features = self.mel_spec(audio)

        indx_maestro, indx_synth, indx_strong, indx_weak, indx_unlabelled = np.cumsum(
            self.hparams["training"]["batch_size"]
        ) # バッチ数の累積和を各要素で得る

        batch_num = features.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(features).bool()
        weak_mask = torch.zeros(batch_num).to(features).bool()
        mask_unlabeled = torch.zeros(batch_num).to(features).bool() # 一貫性損失用
        full_mask_unlabeled = torch.zeros(batch_num).to(features).bool() # SAT用

        strong_mask[:indx_strong] = 1 # maestro,合成(synthは確か合成音),強ラベルデータ
        weak_mask[indx_strong:indx_weak] = 1 # 弱ラベルデータ
        mask_unlabeled[indx_maestro:] = 1 # maestro以外: 合成,強,弱,ラベルなしデータ
        full_mask_unlabeled[indx_weak:] = 1 # 本当にunlabeledしか含まれていない

        # deriving weak labels
        mixup_type = self.hparams["training"].get("mixup")
        if (
            mixup_type is not None
            and self.hparams["training"]["mixup_prob"] > random.random()
        ):
            # NOTE: mix only within same dataset !
            features, embeddings, labels = self.apply_mixup(
                features, embeddings, labels, indx_strong, indx_weak
            )
            features, embeddings, labels = self.apply_mixup(
                features, embeddings, labels, indx_maestro, indx_strong
            )
            features, embeddings, labels = self.apply_mixup(
                features, embeddings, labels, 0, indx_maestro
            )
            # ラベルなしデータを拡張
            features, embeddings, labels = self.apply_mixup(
                features, embeddings, labels, indx_weak, indx_unlabelled
            )


        # mask labels for invalid datasets classes after mixup.
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        labels = labels.masked_fill(
            ~valid_class_mask[:, :, None].expand_as(labels), 0.0
        )
        labels_weak = labels_weak.masked_fill(~valid_class_mask[weak_mask], 0.0)

        # sed student forward
        strong_preds_student, weak_preds_student = self.detect(
            features,
            self.sed_student,
            embeddings=embeddings,
            classes_mask=valid_class_mask,
        )
        # 生徒モデルにラベルなしデータも入れるが,lossに使う際にmaskする

        # print(f"[DEBUG] Student | strong: {strong_preds_student}, weak: {weak_preds_student}")

        # supervised loss on strong labels
        loss_strong = self.supervised_loss(
            strong_preds_student[strong_mask],
            labels[strong_mask],
        )
        # supervised loss on weakly labelled

        loss_weak = self.supervised_loss(
            weak_preds_student[weak_mask],
            labels_weak,
        )
        # total supervised loss
        tot_loss_supervised = loss_strong + loss_weak

        with torch.no_grad(): 
            strong_preds_teacher, weak_preds_teacher = self.detect(
                features,
                self.sed_teacher,
                embeddings=embeddings,
                classes_mask=valid_class_mask,    
            )


        weight = (
            self.hparams["training"]["const_max"]
            * self.scheduler["scheduler"]._get_scaling_factor()
        ) if self.current_epoch < self.hparams["training"]["epoch_decay"] else self.hparams["training"]["const_max"]
        # should we apply the valid mask for classes also here ?


        cmt_active = self.cmt_enabled and (self.current_epoch >= self.cmt_warmup_epochs)

        # # CMT
        if cmt_active:
            # Apply CMT processing
            with torch.no_grad():
                # Apply CMT postprocessing to teacher predictions
                # full_mask_unlabeledに修正
                teacher_pseudo_w, teacher_pseudo_s = self.apply_cmt_postprocessing(
                    weak_preds_teacher[mask_unlabeled], 
                    strong_preds_teacher[mask_unlabeled],
                    phi_clip=self.cmt_phi_clip,
                    phi_frame=self.cmt_phi_frame,
                )
                
                # Compute confidence weights
                confidence_w, confidence_s = self.compute_cmt_confidence_weights(
                    weak_preds_teacher[mask_unlabeled],
                    strong_preds_teacher[mask_unlabeled],
                    teacher_pseudo_w,
                    teacher_pseudo_s
                )

                # # Debug statistics
                # pseudo_label_ratio_w = teacher_pseudo_w.mean()
                # pseudo_label_ratio_s = teacher_pseudo_s.mean()
                # confidence_w_mean = confidence_w.mean()
                # confidence_s_mean = confidence_s.mean()
                # teacher_pred_w_mean = weak_preds_teacher[mask_unlabeled].mean()
                # teacher_pred_s_mean = strong_preds_teacher[mask_unlabeled].mean()
            
            # Compute CMT consistency loss with confidence weighting
            weak_self_sup_loss, strong_self_sup_loss = self.compute_cmt_consistency_loss(
                weak_preds_student[mask_unlabeled],
                strong_preds_student[mask_unlabeled],
                teacher_pseudo_w,
                teacher_pseudo_s,
                confidence_w,
                confidence_s
            )
        else:
            # Original Mean Teacher consistency loss
            strong_self_sup_loss = self.selfsup_loss(
                strong_preds_student[mask_unlabeled],
                strong_preds_teacher[mask_unlabeled],
            )
            weak_self_sup_loss = self.selfsup_loss(
                weak_preds_student[mask_unlabeled],
                weak_preds_teacher[mask_unlabeled],
            )

        tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss) * weight


        # ===========================================================
        # SAT
        # ===========================================================
        loss_pseudo = torch.tensor(0.0).to(self.device)

        if self.sat_enabled:
            # --- 必要な変数を取得 ---
            # 教師モデルによる予測. 疑似ラベル用
            # 弱拡張ラベルなしデータで予測を得て,疑似ラベル作成に使用
            q_c = weak_preds_teacher[full_mask_unlabeled]   # クリップ予測 (B_u, K)
            q_f = strong_preds_teacher[full_mask_unlabeled] # フレーム予測 (B_u, K, T)
            
            # ===========================================================
            # 強拡張 (CutMix) の適用
            # ===========================================================

            # ラベルなしデータの特徴量を取得
            features_unlabeled = features[full_mask_unlabeled]
            embeddings_unlabeled = embeddings[full_mask_unlabeled]
            classes_mask_unlabeled = valid_class_mask[full_mask_unlabeled]

            # # CutMix強拡張を適用
            # cutmix_prob = self.hparams.get("sat", {}).get("cutmix_prob", 1.0)
            # cutmix_alpha = self.hparams.get("sat", {}).get("cutmix_alpha", 1.0)

            # if random.random() < cutmix_prob:
            #     # CutMixを適用（ラベルは不要なのでNone）
            #     features_SA, q_f_mixed, q_c_mixed = cutmix(
            #         features_unlabeled,
            #         target_f=q_f,
            #         target_c =q_c,
            #         alpha=cutmix_alpha
            #     )
            # else:
            #     # CutMixを適用しない場合は元のデータをそのまま使用
            #     features_SA = features_unlabeled

            # # 強拡張データで生徒モデルをフォワード
            # # 注: 疑似ラベル損失で生徒モデルを訓練するため、勾配計算が必要
            # strong_preds_student_SA, weak_preds_student_SA = self.detect(
            #     features_SA,
            #     self.sed_student,
            #     embeddings=embeddings_unlabeled,  # embeddingsは元のものを使用
            #     classes_mask=classes_mask_unlabeled,
            # )

            # # 疑似ラベル損失計算用の予測値
            # s_c = weak_preds_student_SA  # クリップレベル予測 (B_u, K)
            # s_f = strong_preds_student_SA  # フレームレベル予測 (B_u, K, T)

            # B, K, T: バッチ,クラス,フレーム

            # バッチサイズとクラス数を取得
            if q_c.dim() == 0 or q_f.dim() == 0:
                 # バッチにラベルなしデータが1つもなかった場合
                 pass # loss_pseudo は 0.0 のまま
            else:
                B_u, K = q_c.shape
                _, _, T = q_f.shape # q_fは (B_u, K, T)
            
                # ===========================================================
                # 1. SACT (Clip-level Adaptive Thresholding) (ステップ 6, 7)
                # ===========================================================
                
                # ステップ 6.1: バッチの基礎閾値 v_b の計算 (Eq 1)
                
                # 期待個数 (形状 [B_u])
                expected_counts = torch.ceil(torch.sum(q_c, dim=1)) # ok
                # 降順ソート (形状 [B_u, K])
                sorted_preds, _ = torch.sort(q_c, dim=1, descending=True) # ok
                # インデックスの準備 (0-based index)
                indices = (expected_counts - 1).long().clamp(0, K - 1) # ok
                # 基礎閾値 v_b を取得 (形状 [B_u])
                v_b = sorted_preds.gather(1, indices.unsqueeze(1)).squeeze(1) # 後で確認

                # ステップ 6.2: グローバル閾値 tau_s のEMA更新 (Eq 2)
                batch_mean_v_b = torch.mean(v_b)
                self.global_clip_threshold = self.sat_lambda * self.global_clip_threshold.data + (1 - self.sat_lambda) * batch_mean_v_b # ok
                tau_s = self.global_clip_threshold

                # ステップ 6.3: ローカル確率 p_tilde_s(k) のEMA更新 (Eq 3)
                batch_mean_q_c = torch.mean(q_c, dim=0) # (形状 [K])
                self.local_clip_probabilities = self.sat_lambda * self.local_clip_probabilities.data + (1 - self.sat_lambda) * batch_mean_q_c # ok
                p_tilde_s = self.local_clip_probabilities

                # ステップ 6.4: 適応的閾値 tau_s^c(k) の計算 (Eq 4)
                # (p_tilde_sの最大値で正規化)
                adaptive_clip_thresholds = (p_tilde_s / torch.max(p_tilde_s)) * tau_s # (形状 [K]) # ok

                # ステップ 7: クリップ疑似ラベル L_Clip_c の生成
                L_Clip_c = (q_c > adaptive_clip_thresholds).float() # (形状 [B_u, K]) # ok?
                
                
                # ===========================================================
                # 2. SAFT (Frame-level Adaptive Thresholding) (ステップ 11, 12)
                # ===========================================================
                
                # (形状 [K])
                adaptive_frame_thresholds_k = torch.zeros(K, device=self.device)
                filtered_q_f = q_f * L_Clip_c.unsqueeze(2)

                # GMMがインポートされているか、計算コスト高騰を許容する場合
                if self.gmm_imported:
                    try:
                        # ステップ 11.1: フレーム予測のフィルタリング (Eq 6)
                        # L_Clip_c を (B, K, 1) に拡張してブロードキャスト
                        # filtered_q_f = q_f * L_Clip_c.unsqueeze(2) # (B, K, T) #ok?
                        
                        # ステップ 11.2-4: GMMによる閾値計算 (Eq 7-9)
                        for k in range(10):
                            # クラス k の、フィルタリング後の0より大きい予測値を取得
                            class_k_preds = filtered_q_f[:, k, :]
                            active_preds_k = class_k_preds[class_k_preds > 1e-8] # ゼロに近い値を除外
                            
                            # self.log(
                            #     f"debug/sample_count_class_{k}", 
                            #     float(active_preds_k.numel()), 
                            #     on_step=False, 
                            #     on_epoch=True)

                            if (self.current_epoch % 10 == 0) and (batch_indx == 0):
                                if active_preds_k.numel() > 0:
                                    # Tensor -> Numpy
                                    data_for_hist = active_preds_k.detach().cpu().numpy()
                                    
                                    # wandb.Histogramオブジェクトを作成
                                    hist = wandb.Histogram(data_for_hist)
                                    
                                    # 【重要】Lightningのlogを経由せず、直接自作のメソッドを呼ぶ
                                    # これにより step数(_debug/global_step) が自動付与され、かつLightningのエラーを回避できる
                                    self._maybe_wandb_log({
                                        f"distribution/pred_hist_class_{k}": hist
                                    })

                            if active_preds_k.numel() >= 20: # 20は最小限の目安（sklearnのデフォルト等考慮） 
                                # sklearn入力用にnumpy変換 (N, 1)
                                X = active_preds_k.detach().cpu().numpy().reshape(-1, 1)
                                
                                # GMMフィッティング
                                # n_init=1だと初期値依存で失敗しやすいため、計算コスト許容なら5程度推奨
                                gmm = GaussianMixture(n_components=2, max_iter=100, n_init=3, covariance_type='full', random_state=42)
                                gmm.fit(X)
                                
                                if not gmm.converged_:
                                    # 収束しなかった場合はFallback（クリップ閾値などを使用）
                                    adaptive_frame_thresholds_k[k] = adaptive_clip_thresholds[k]
                                else:
                                    # active (平均値が高い方) のクラスタを見つける
                                    idx_active = np.argmax(gmm.means_)
                                    
                                    # Eq 8: "maximum probability in active mode"
                                    # ガウス分布の頂点である平均値を採用（論文の意図に即した解釈）
                                    mu_a_k = float(gmm.means_[idx_active][0])  # numpy.float32をPython floatに変換
                                    adaptive_frame_thresholds_k[k] = mu_a_k
                                    print("[DEBUG] success")
                            else:
                                # サンプル不足時はクリップ単位の閾値を流用（または固定値0.5など）
                                adaptive_frame_thresholds_k[k] = adaptive_clip_thresholds[k]
                                if self.current_epoch > 30:
                                    print("[DEBUG] lack sample: ", active_preds_k.numel())
                                else:
                                    pass
                            
                            # クリップの閾値
                            self.log(f"debug/adaptive_clip_thresholds_{k}", adaptive_clip_thresholds[k])
                            # フレームの閾値
                            self.log(f"debug/adaptive_frame_thresholds_{k}", adaptive_frame_thresholds_k[k])
                            # クラスごとに予測値の最大（空でない場合のみ）
                            if active_preds_k.numel() > 0:
                                self.log(f"debug/active_preds_{k}_max", active_preds_k.max())

                    except Exception as e:
                        # GMMフィット失敗時（特異行列エラーなど）の安全策
                        print(f"GMM fit failed for class {k}: {e}")
                        adaptive_frame_thresholds_k[k] = adaptive_clip_thresholds[k]
                        # print("[DEBUG] fit failed")
                else:
                    # サンプル不足時はクリップ単位の閾値を流用（または固定値0.5など）
                    adaptive_frame_thresholds_k[k] = adaptive_clip_thresholds[k]
                    print("[DEBUG] lack sample for GMM")

                # ステップ 11.5 & 12: フレーム疑似ラベル L_Frame_f の生成 (Eq 10)
                # 閾値 (形状 [K]) を (1, K, 1) に拡張してブロードキャスト
                L_Frame_f = (filtered_q_f > adaptive_frame_thresholds_k.view(1, K, 1)).float() # (B_u, K, T)
                
                # ===========================================================
                # 3. 疑似ラベル損失 (L^u) の計算 (ステップ 16)
                # ===========================================================

                # CutMix強拡張を適用
                cutmix_prob = self.hparams.get("sat", {}).get("cutmix_prob", 1.0)

                if random.random() < cutmix_prob:
                    # CutMixを適用（ラベルは不要なのでNone）
                    features_SA, c_mixed, f_mixed = cutmix(
                        features_unlabeled,
                        target_f=L_Frame_f,
                        target_c=L_Clip_c,
                        alpha=self.cutmix_alpha
                    )
                    L_Frame_f = f_mixed
                    L_Clip_c = c_mixed
                else:
                    # CutMixを適用しない場合は元のデータをそのまま使用
                    features_SA = features_unlabeled

                # 強拡張データで生徒モデルをフォワード
                # 注: 疑似ラベル損失で生徒モデルを訓練するため、勾配計算が必要
                strong_preds_student_SA, weak_preds_student_SA = self.detect(
                    features_SA,
                    self.sed_student,
                    embeddings=embeddings_unlabeled,  # embeddingsは元のものを使用
                    classes_mask=classes_mask_unlabeled,
                )

                # 疑似ラベル損失計算用の予測値
                s_c = weak_preds_student_SA  # クリップレベル予測 (B_u, K)
                s_f = strong_preds_student_SA  # フレームレベル予測 (B_u, K, T)

                # クリップ疑似ラベル損失 (B_u, K)
                criterion = torch.nn.BCELoss()
                loss_pseudo_clip = criterion(
                    s_c,
                    L_Clip_c
                )
                
                # フレーム疑似ラベル損失 (B_u, K, T)
                loss_pseudo_frame = criterion(
                    s_f,
                    L_Frame_f
                )
                
                loss_pseudo = self.sat_w_u * (loss_pseudo_clip + loss_pseudo_frame)

                # ロギング
                self.log("train/sat/loss_pseudo_clip", loss_pseudo_clip)
                self.log("train/sat/loss_pseudo_frame", loss_pseudo_frame)
                self.log("train/sat/loss_pseudo_total", loss_pseudo, prog_bar=True)
                self.log("train/sat/global_clip_threshold", tau_s)
                # 疑似ラベルの密度（どのくらいの割合が1になったか）
                self.log("train/sat/L_Clip_c_ratio", L_Clip_c.mean())
                self.log("train/sat/L_Frame_f_ratio", L_Frame_f.mean())

        # ========================================================================
        # 最終損失の計算
        # ========================================================================
        
        # 既存の損失に、計算した疑似ラベル損失を加える
        tot_loss = tot_loss_supervised + tot_self_loss + loss_pseudo

        # tot_loss = tot_loss_supervised + tot_self_loss

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
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True)

        # 各種step情報
        # self.log("train/step/scheduler_step_num", self.scheduler["scheduler"].step_num, prog_bar=True)
        self.log("train/step/global_step", self.global_step)
        self.log("train/step/current_epoch", self.current_epoch)

        # モデルの状態を見る
        # self.log("debug/student_self_strong_mean", strong_preds_student[mask_unlabeled].mean().item())
        # self.log("debug/student_self_strong_max", strong_preds_student[mask_unlabeled].max().item())
        # self.log("debug/teacher_self_strong_mean", strong_preds_teacher[mask_unlabeled].mean().item())
        # 生のMSEはstrong_self_sup_loss

        # # CMT specific logging
        # if self.cmt_enabled:
        #     self.log("train/cmt/phi_clip", self.cmt_phi_clip)
        #     self.log("train/cmt/phi_frame", self.cmt_phi_frame)
        #     if cmt_active:
        #         self.log("train/cmt/pseudo_label_ratio_w", pseudo_label_ratio_w)
        #         self.log("train/cmt/pseudo_label_ratio_s", pseudo_label_ratio_s)
        #         self.log("train/cmt/confidence_w_mean", confidence_w_mean)
        #         self.log("train/cmt/confidence_s_mean", confidence_s_mean)
        #         self.log("train/cmt/teacher_pred_w_mean", teacher_pred_w_mean)
        #         self.log("train/cmt/teacher_pred_s_mean", teacher_pred_s_mean)
        #         self.log("train/cmt/nonzero_samples_w", (confidence_w > 0).float().mean())
        #         self.log("train/cmt/nonzero_samples_s", (confidence_s > 0).float().mean())

        return tot_loss

    def on_before_zero_grad(self, *args, **kwargs):
        # update EMA teacher
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler["scheduler"].step_num,
            self.sed_student,
            self.sed_teacher,
        )
        

        # self.log("debug/ema_weight_distance", euclidean_dist)
        # self.log("debug/ema_cosine_sim", cosine_sim)

    def validation_step(self, batch, batch_indx):
        """Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames, embeddings, valid_class_mask = (
            self._unpack_batch(batch)
        )

        if self.hparams["pretrained"]["e2e"]:
            # extract embeddings here
            if self.pretrained_model.training and self.hparams["pretrained"]["freezed"]:
                # check that is freezed
                self.pretrained_model.eval()
            embeddings = self.pretrained_model(embeddings)[
                self.hparams["net"]["embedding_type"]
            ]

        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(
            mels, self.sed_student, embeddings=embeddings, classes_mask=valid_class_mask
        )
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(
            mels, self.sed_teacher, embeddings=embeddings, classes_mask=valid_class_mask
        )

        # we derive masks for each dataset based on folders of filenames
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["weak_folder"]))
                    for x in filenames
                ]
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
                ]
            )
            .to(audio)
            .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()

            loss_weak_student = self.supervised_loss(
                weak_preds_student[mask_weak], labels_weak
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[mask_weak], labels_weak
            )
            self.log("val/weak/student/loss_weak", loss_weak_student)
            self.log("val/weak/teacher/loss_weak", loss_weak_teacher)

            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro(
                weak_preds_student[mask_weak], labels_weak.long()
            )
            self.get_weak_teacher_f1_seg_macro(
                weak_preds_teacher[mask_weak], labels_weak.long()
            )

        if torch.any(mask_strong):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_strong], labels[mask_strong]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_strong], labels[mask_strong]
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
                scores_postprocessed_student_strong
            )

            self.val_tune_sebbs_student.update(
                scores_unprocessed_student_strong
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
                scores_postprocessed_teacher_strong
            )

            self.val_tune_sebbs_teacher.update(
                scores_unprocessed_teacher_strong
            )

        return

    def validation_epoch_end(self, outputs):
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
            self.hparams["data"]["synth_val_tsv"]
        )

        desed_audio_durations = sed_scores_eval.io.read_audio_durations(
            self.hparams["data"]["synth_val_dur"]
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
            audio_id: desed_audio_durations[audio_id]
            for audio_id in desed_ground_truth.keys()
        }
        keys = ["onset", "offset"] + sorted(classes_labels_desed.keys())
        desed_scores = {
            clip_id: self.val_buffer_sed_scores_eval_student[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }


        psds1_sed_scores_eval_student = compute_psds_from_scores(
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
        collar_f1_macro_thres05_student_sed_scores_eval = (
            sed_scores_eval.collar_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                onset_collar=0.2,
                offset_collar=0.2,
                offset_collar_rate=0.2,
            )[0]["macro_average"]
        )
        desed_scores = {
            clip_id: self.val_buffer_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }
        psds1_sed_scores_eval_teacher = compute_psds_from_scores(
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
        collar_f1_macro_thres05_teacher_sed_scores_eval = (
            sed_scores_eval.collar_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                onset_collar=0.2,
                offset_collar=0.2,
                offset_collar_rate=0.2,
            )[0]["macro_average"]
        )

        # maestro
        maestro_ground_truth = pd.read_csv(
            self.hparams["data"]["real_maestro_train_tsv"], sep="\t"
        )
        maestro_ground_truth = maestro_ground_truth[
            maestro_ground_truth.confidence > 0.5
        ]
        maestro_ground_truth = maestro_ground_truth[
            maestro_ground_truth.event_label.isin(classes_labels_maestro_real_eval)
        ]
        maestro_ground_truth = {
            clip_id: events
            for clip_id, events in sed_scores_eval.io.read_ground_truth_events(
                maestro_ground_truth
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
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented."
            )

        obj_metric_maestro_type = self.hparams["training"].get(
            "obj_metric_maestro_type"
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
                f"obj_metric_maestro_type: {obj_metric_maestro_type} not implemented."
            )

        obj_metric = torch.tensor(
            weak_student_f1_macro.item() + synth_metric + maestro_metric
        )

        # # クラス別の統計を計算
        # for i, class_name in enumerate(self.encoder.labels):
        #     precision = 
        #     recall = 
        #     f1 = 2 * precision * recall / (precision + recall)
            
        #     self.log(f"val/class/{class_name}/precision", precision)
        #     self.log(f"val/class/{class_name}/recall", recall)
        #     self.log(f"val/class/{class_name}/f1", f1)

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log(
            "val/student/weak_f1_macro_thres05/torchmetrics", weak_student_f1_macro
        )
        self.log(
            "val/teacher/weak_f1_macro_thres05/torchmetrics", weak_teacher_f1_macro
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

        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames, embeddings, valid_class_mask = (
            self._unpack_batch(batch)
        )

        if self.hparams["pretrained"]["e2e"]:
            # extract embeddings here
            if self.pretrained_model.training and self.hparams["pretrained"]["freezed"]:
                # check that is freezed
                self.pretrained_model.eval()
            embeddings = self.pretrained_model(embeddings)[
                self.hparams["net"]["embedding_type"]
            ]

        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(
            mels, self.sed_student, embeddings
        )
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(
            mels, self.sed_teacher, embeddings
        )

        if not self.evaluation:
            loss_strong_student = self.supervised_loss(strong_preds_student, labels)
            loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

            self.log("test/student/loss_strong", loss_strong_student)
            self.log("test/teacher/loss_strong", loss_strong_teacher)


        if self.sebbs_enabled:
            # desed synth dataset
            desed_ground_truth = sed_scores_eval.io.read_ground_truth_events(
                self.hparams["data"]["synth_val_tsv"]
            )

            desed_audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["synth_val_dur"]
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
                audio_id: desed_audio_durations[audio_id]
                for audio_id in desed_ground_truth.keys()
            }
            keys = ["onset", "offset"] + sorted(classes_labels_desed.keys())
            desed_scores = {
                clip_id: self.val_tune_sebbs_student[clip_id][keys]
                for clip_id in desed_ground_truth.keys()
            }

            # ========================================================================
            # cSEBBs (change-point based Sound Event Bounding Boxes) のチューニング
            # ========================================================================
            # cSEBBsは、フレームレベルのスコアから変化点検出によりイベント境界を推定し、
            # 適応的なセグメントマージによって最終的なイベント候補(bounding boxes)を生成する。
            # ここでは、validation setを用いてハイパーパラメータをチューニング

            # # --- 1. DESEDクラス用のcSEBBsチューニング ---
            if not hasattr(self, "csebbs_predictor_desed"):
                print("\n=== Tuning cSEBBs for DESED classes ===")
                # ハイパーパラメータ:
                #   - step_filter_length: 変化点検出用のステップフィルタ長
                #   - merge_threshold_abs: セグメント統合の絶対閾値
                #   - merge_threshold_rel: セグメント統合の相対閾値
                # これらをグリッドサーチでPSDSが最大となるように最適化
                self.csebbs_predictor_desed, _ = csebbs.tune(
                    scores=desed_scores,
                    ground_truth=desed_ground_truth,
                    audio_durations=desed_audio_durations,
                    selection_fn=csebbs.select_best_psds  # PSDS1を最大化
                )
                print(f"✓ DESED cSEBBs tuning completed")

            # --- 1-2. DESEDクラス用のcSEBBsチューニング（教師モデル） ---
            if not hasattr(self, "csebbs_predictor_desed_teacher"):
                print("\n=== Tuning cSEBBs for DESED classes (Teacher) ===")
                desed_scores_teacher = {
                    clip_id: self.val_tune_sebbs_teacher[clip_id][keys]
                    for clip_id in desed_ground_truth.keys()
                }
                self.csebbs_predictor_desed_teacher, _ = csebbs.tune(
                    scores=desed_scores_teacher,
                    ground_truth=desed_ground_truth,
                    audio_durations=desed_audio_durations,
                    selection_fn=csebbs.select_best_psds
                )
                print(f"✓ DESED cSEBBs tuning completed (Teacher)")

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
                        print(f"\n[DEBUG] MAESTRO Student matching failed!")
                        print(f"  Sample validation clip IDs: {sample_val_clips}")
                        print(f"  Sample ground truth clip IDs: {sample_gt_clips}")

                    segment_length = 1.0
                    if maestro_val_scores:
                        # 十分なvalidationデータがある場合はチューニング実行
                        self.csebbs_predictor_maestro, _ = csebbs.tune(
                            scores=maestro_val_scores,
                            ground_truth=maestro_ground_truth,
                            audio_durations=maestro_audio_durations,
                            selection_fn=select_best_auroc,  # AUROCを最大化
                            segment_length=segment_length
                        )
                        print(f"✓ MAESTRO cSEBBs tuning completed with {len(maestro_val_scores)} clips")
                    else:
                        # validationデータが見つからない場合はデフォルトパラメータを使用
                        print("⚠ Warning: No MAESTRO validation scores found")
                        print("  Using default cSEBBs parameters for MAESTRO")
                        self.csebbs_predictor_maestro = csebbs.CSEBBsPredictor(
                            step_filter_length=0.48,   # 中程度のフィルタ長
                            merge_threshold_abs=0.2,   # 中程度の絶対閾値
                            merge_threshold_rel=2.0    # 中程度の相対閾値
                        )
                else:
                    # ground truthやdurationsが読み込まれていない場合
                    print("⚠ Warning: MAESTRO ground truth or durations not available")
                    print("  Using default cSEBBs parameters for MAESTRO")
                    self.csebbs_predictor_maestro = csebbs.CSEBBsPredictor(
                        step_filter_length=0.48,
                        merge_threshold_abs=0.2,
                        merge_threshold_rel=2.0
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
                    
                    self.log("debug/maestro_teacher/total_validation_clips", total_val_clips_teacher)
                    self.log("debug/maestro_teacher/total_ground_truth_clips", total_gt_clips_teacher)
                    self.log("debug/maestro_teacher/matched_clips", matched_clips_teacher)
                    
                    if matched_clips_teacher == 0:
                        # マッチング失敗時の詳細情報をprintで表示
                        sample_val_clips = list(self.val_tune_sebbs_teacher.keys())[:3]
                        sample_gt_clips = list(maestro_ground_truth.keys())[:3]
                        print(f"\n[DEBUG] MAESTRO Teacher matching failed!")
                        print(f"  Sample validation clip IDs: {sample_val_clips}")
                        print(f"  Sample ground truth clip IDs: {sample_gt_clips}")
                    
                    if maestro_val_scores_teacher:
                        self.csebbs_predictor_maestro_teacher, _ = csebbs.tune(
                            scores=maestro_val_scores_teacher,
                            ground_truth=maestro_ground_truth,
                            audio_durations=maestro_audio_durations,
                            selection_fn=csebbs.select_best_psds
                        )
                        print(f"✓ MAESTRO cSEBBs tuning completed (Teacher) with {len(maestro_val_scores_teacher)} clips")
                    else:
                        print("⚠ Warning: No MAESTRO validation scores found (Teacher)")
                        print("  Using default cSEBBs parameters for MAESTRO (Teacher)")
                        self.csebbs_predictor_maestro_teacher = csebbs.CSEBBsPredictor(
                            step_filter_length=0.48,
                            merge_threshold_abs=0.2,
                            merge_threshold_rel=2.0
                        )
                else:
                    print("⚠ Warning: MAESTRO ground truth or durations not available (Teacher)")
                    print("  Using default cSEBBs parameters for MAESTRO (Teacher)")
                    self.csebbs_predictor_maestro_teacher = csebbs.CSEBBsPredictor(
                        step_filter_length=0.48,
                        merge_threshold_abs=0.2,
                        merge_threshold_rel=2.0
                    )


        # ========================================================================
        # Student modelのスコア生成とcSEBBs後処理
        # ========================================================================

        # batched_decode_preds()は以下を実行:
        #   1. median filterによるスコアの平滑化
        #   2. 複数の閾値でのバイナリ検出（PSDS計算用）
        #   3. sed_scores_eval形式への変換
        (
            scores_unprocessed_student_strong,  # median filter適用前のスコア
            scores_postprocessed_student_strong, # 検証用に有効化したが,本当は使用しない
            decoded_student_strong,  # 閾値別のバイナリ検出結果
        ) = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.median_filter,
            thresholds=list(self.test_buffer_psds_eval_student.keys()) + [0.5],
        )

        # median filterの代わりにcSEBBsを使用してイベント境界を精緻化
        # cSEBBsの利点:
        #   - 変化点検出による正確なonset/offset推定
        #   - 適応的なセグメントマージによるノイズ除去
        #   - フレームレベル閾値の影響を受けないイベント検出
        # 入力: median filter適用前のスコア（より細かい時間分解能を保持）
        # 出力: cSEBBsにより生成されたイベント候補のスコア

        if self.sebbs_enabled:
            scores_postprocessed_student_strong = get_sebbs(
                self, scores_unprocessed_student_strong, model_type='student'
            )

        # 後処理前後のスコアを保存（比較・分析用）
        self.test_buffer_sed_scores_eval_unprocessed_student.update(
            scores_unprocessed_student_strong
        )
        # cSEBBs後処理済みスコア（最終的な評価に使用）
        self.test_buffer_sed_scores_eval_student.update(
            scores_postprocessed_student_strong
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
                self, scores_unprocessed_teacher_strong, model_type='teacher'
            )

        self.test_buffer_sed_scores_eval_unprocessed_teacher.update(
            scores_unprocessed_teacher_strong
        )
        self.test_buffer_sed_scores_eval_teacher.update(
            scores_postprocessed_teacher_strong
        )
        for th in self.test_buffer_psds_eval_teacher.keys():
            self.test_buffer_psds_eval_teacher[th] = pd.concat(
                [self.test_buffer_psds_eval_teacher[th], decoded_teacher_strong[th]],
                ignore_index=True,
            )

        # compute f1 score
        self.test_buffer_detections_thres05_student = pd.concat(
            [self.test_buffer_detections_thres05_student, decoded_student_strong[0.5]]
        )
        self.test_buffer_detections_thres05_teacher = pd.concat(
            [self.test_buffer_detections_thres05_teacher, decoded_teacher_strong[0.5]]
        )

    def on_test_epoch_end(self):
        # pub eval dataset
        save_dir = os.path.join(self.exp_dir, "metrics_test")
        print("save_dir", save_dir)
        results = {}
        if self.evaluation:
            # only save prediction scores
            save_dir_student_unprocessed = os.path.join(
                save_dir, "student_scores", "unprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_unprocessed_student,
                save_dir_student_unprocessed,
            )
            print(f"\nRaw scores for student saved in: {save_dir_student_unprocessed}")

            save_dir_student_postprocessed = os.path.join(
                save_dir, "student_scores", "postprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_student,
                save_dir_student_postprocessed,
            )
            print(
                f"\nPostprocessed scores for student saved in: {save_dir_student_postprocessed}"
            )

            save_dir_teacher_unprocessed = os.path.join(
                save_dir, "teacher_scores", "unprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_unprocessed_teacher,
                save_dir_teacher_unprocessed,
            )
            print(f"\nRaw scores for teacher saved in: {save_dir_teacher_unprocessed}")

            save_dir_teacher_postprocessed = os.path.join(
                save_dir, "teacher_scores", "postprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_teacher,
                save_dir_teacher_postprocessed,
            )
            print(
                f"\nPostprocessed scores for teacher saved in: {save_dir_teacher_postprocessed}"
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
            intersection_f1_macro_thres05_student_psds_eval = (
                compute_per_intersection_macro_f1(
                    {"0.5": self.test_buffer_detections_thres05_student},
                    self.hparams["data"]["test_tsv"],
                    self.hparams["data"]["test_dur"],
                )
            )
            intersection_f1_macro_thres05_teacher_psds_eval = (
                compute_per_intersection_macro_f1(
                    {"0.5": self.test_buffer_detections_thres05_teacher},
                    self.hparams["data"]["test_tsv"],
                    self.hparams["data"]["test_dur"],
                )
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
                self.hparams["data"]["test_tsv"]
            )
            desed_audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["test_dur"]
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
                audio_id: gt
                for audio_id, gt in desed_ground_truth.items()
                if len(gt) > 0
            }
            desed_audio_durations = {
                audio_id: desed_audio_durations[audio_id]
                for audio_id in desed_ground_truth.keys()
            }
            keys = ["onset", "offset"] + sorted(classes_labels_desed.keys())
            desed_scores = {
                clip_id: self.test_buffer_sed_scores_eval_student[clip_id][keys]
                for clip_id in desed_ground_truth.keys()
            }
            psds1_student_sed_scores_eval = compute_psds_from_scores(
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
            psds2_student_sed_scores_eval = compute_psds_from_scores(
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
            intersection_f1_macro_thres05_student_sed_scores_eval = (
                sed_scores_eval.intersection_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    dtc_threshold=0.5,
                    gtc_threshold=0.5,
                )[0]["macro_average"]
            )
            collar_f1_macro_thres05_student_sed_scores_eval = (
                sed_scores_eval.collar_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    onset_collar=0.2,
                    offset_collar=0.2,
                    offset_collar_rate=0.2,
                )[0]["macro_average"]
            )

            desed_scores = {
                clip_id: self.test_buffer_sed_scores_eval_teacher[clip_id][keys]
                for clip_id in desed_ground_truth.keys()
            }
            psds1_teacher_sed_scores_eval = compute_psds_from_scores(
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
            psds2_teacher_sed_scores_eval = compute_psds_from_scores(
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
            intersection_f1_macro_thres05_teacher_sed_scores_eval = (
                sed_scores_eval.intersection_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    dtc_threshold=0.5,
                    gtc_threshold=0.5,
                )[0]["macro_average"]
            )
            collar_f1_macro_thres05_teacher_sed_scores_eval = (
                sed_scores_eval.collar_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    onset_collar=0.2,
                    offset_collar=0.2,
                    offset_collar_rate=0.2,
                )[0]["macro_average"]
            )

            maestro_audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["real_maestro_val_dur"]
            )
            maestro_ground_truth_clips = pd.read_csv(
                self.hparams["data"]["real_maestro_val_tsv"], sep="\t"
            )
            maestro_clip_ids = [filename[:-4] for filename in maestro_ground_truth_clips["filename"]]
            maestro_ground_truth_clips = maestro_ground_truth_clips[
                maestro_ground_truth_clips.confidence > 0.5
            ]
            maestro_ground_truth_clips = maestro_ground_truth_clips[
                maestro_ground_truth_clips.event_label.isin(
                    classes_labels_maestro_real_eval
                )
            ]
            maestro_ground_truth_clips = sed_scores_eval.io.read_ground_truth_events(
                maestro_ground_truth_clips
            )

            # clip ベースの ground truth を保存（validation_epoch_end と同じパターン）
            self._maestro_ground_truth_clips = maestro_ground_truth_clips

            # file ベースに変換したものも保存（test 全体の評価用）
            maestro_ground_truth = _merge_maestro_ground_truth(
                maestro_ground_truth_clips
            )
            self._maestro_ground_truth = maestro_ground_truth
            maestro_audio_durations = {
                file_id: maestro_audio_durations[file_id]
                for file_id in maestro_ground_truth.keys()
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
            event_classes_maestro = sorted(classes_labels_maestro_real_eval) # 他の都合でevalに変更
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
                clip_id: scores_df[keys]
                for clip_id, scores_df in segment_scores_student.items()
            }
            segment_scores_teacher = {
                clip_id: scores_df[keys]
                for clip_id, scores_df in segment_scores_teacher.items()
            }

            segment_f1_macro_optthres_student = (
                sed_scores_eval.segment_based.best_fscore(
                    segment_scores_student,
                    maestro_ground_truth,
                    maestro_audio_durations,
                    segment_length=segment_length,
                )[0]["macro_average"]
            )
            segment_mauc_student = sed_scores_eval.segment_based.auroc(
                segment_scores_student,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
            )[0]["mean"]
            segment_mpauc_student = sed_scores_eval.segment_based.auroc(
                segment_scores_student,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
                max_fpr=0.1,
            )[0]["mean"]
            segment_f1_macro_optthres_teacher = (
                sed_scores_eval.segment_based.best_fscore(
                    segment_scores_teacher,
                    maestro_ground_truth,
                    maestro_audio_durations,
                    segment_length=segment_length,
                )[0]["macro_average"]
            )
            segment_mauc_teacher = sed_scores_eval.segment_based.auroc(
                segment_scores_teacher,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
            )[0]["mean"]
            segment_mpauc_teacher = sed_scores_eval.segment_based.auroc(
                segment_scores_teacher,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
                max_fpr=0.1,
            )[0]["mean"]

            results.update({
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
            })
            self.tracker_devtest.stop()

        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=True)
        wandb.finish()

    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):
        self.train_loader = SafeDataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = SafeDataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
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
        self.logger.log_metrics(
            {"/train/tot_energy_kWh": torch.tensor(float(training_kwh))}
        )

    def on_test_start(self) -> None:
        """Test開始時の初期化処理。

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
                print("\n" + "="*70)
                print("WARNING: Validation data not available, skipping cSEBBs tuning")
                print("cSEBBs will use default parameters without validation-based tuning")
                print("="*70)
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
                print("\n" + "="*70)
                print("Running validation pass to collect scores for cSEBBs tuning")
                print("="*70)

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
                        moved_batch = (audio, labels, padded_indxs, filenames, embeddings, valid_class_mask)

                        # validation_stepを実行
                        # この中でval_buffer_sed_scores_eval_studentにスコアが蓄積される
                        self.validation_step(moved_batch, batch_idx)

                print(f"\n✓ Validation pass complete")
                print(f"  Collected scores for {len(self.val_tune_sebbs_student)} clips")

            # 重要: validation_epoch_end()は呼び出さない
            # 理由: validation_epoch_end()内でバッファがクリアされてしまい、
            #       test_step内でのcSEBBsチューニングに使えなくなるため

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


    def load_maestro_audio_durations_and_gt(self):
        """
        MAESTROのaudio durationsとground truthを読み込む。
        
        注: validation時と同じデータソース(real_maestro_train_tsv)を使用することで、
        cSEBBsチューニング時のスコアとground truthのマッチングを可能にする。
        """
        # --- 変更: trainデータを使用してvalidationスコアとマッチング ---
        gt_tsv_path = self.hparams["data"]["real_maestro_train_tsv"]
        
        # durationsの読み込み（trainセット用のdurationsファイル）
        # 設定ファイルにreal_maestro_train_durがあればそれを使用、なければフォールバック
        durations_path = self.hparams["data"].get(
            "real_maestro_train_dur",
            self.hparams["data"]["real_maestro_val_dur"]  # フォールバック
        )

        maestro_audio_durations = sed_scores_eval.io.read_audio_durations(durations_path)

        # --- 2. ground truth tsv の読み込みとフィルタ ---
        maestro_ground_truth_clips = pd.read_csv(gt_tsv_path, sep="\t")
        # 元ファイルの filename カラムが "xxxx.wav" のようになっている前提
        # ここで clip_id の仕様に合わせて切る（例: remove .wav）
        maestro_ground_truth_clips["file_id"] = maestro_ground_truth_clips["filename"].apply(lambda x: x[:-4] if isinstance(x, str) and x.lower().endswith(".wav") else x)

        # confidence とラベルフィルタ
        maestro_ground_truth_clips = maestro_ground_truth_clips[maestro_ground_truth_clips.confidence > 0.5]
        maestro_ground_truth_clips = maestro_ground_truth_clips[
            maestro_ground_truth_clips.event_label.isin(classes_labels_maestro_real_eval)
        ]

        # --- 3. read_ground_truth_events に通す（返り値が dict になる前提） ---
        maestro_ground_truth_clips = sed_scores_eval.io.read_ground_truth_events(maestro_ground_truth_clips)

        # --- 4. マッピングを clip_id の集合に揃える ---
        maestro_ground_truth = _merge_maestro_ground_truth(maestro_ground_truth_clips)  # 既存関数使用
        # maestro_audio_durations のキーが file_id と一致するか確かめる
        # ここで、該当する file_id のみ抽出
        maestro_audio_durations_filtered = {
            file_id: maestro_audio_durations[file_id]
            for file_id in maestro_ground_truth.keys()
            if file_id in maestro_audio_durations
        }

        # maestro_audio_durations_filtered = {
        #     clip_id: sorted(events, key=lambda x: x[1])[-1][1]
        #     for clip_id, events in maestro_ground_truth.items()
        # }


        missing = set(maestro_ground_truth.keys()) - set(maestro_audio_durations_filtered.keys())
        if missing:
            warnings.warn(f"maestro_audio_durations missing for {len(missing)} files. Examples: {list(missing)[:5]}. Using fallback for those clips.")
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
    **kwargs
):
    """
    cSEBBsのハイパーパラメータチューニングでAUROCを最大化する選択関数

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
            audio_duration=audio_durations
        )

        # AUROC計算
        auroc_values, _ = sed_scores_eval.segment_based.auroc(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            segment_length=segment_length,
            **kwargs
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
                for event_onset_time, event_offset_time, event_class in clip_ground_truth[
                    clip_id
                ]
            ]
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
    frame_scores, audio_durations, event_classes, segment_length=1.0
):
    """
    >>> event_classes = ['a', 'b', 'c']
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
                (ceil(audio_durations[file_id] / segment_length), len(event_classes))
            )
            summand_count[file_id] = np.zeros_like(segment_scores_file[file_id])
        segment_scores_clip = _get_segment_scores(
            frame_scores[clip_id][keys],
            clip_length=(clip_offset_time - clip_onset_time),
            segment_length=1.0,
        )[event_classes].to_numpy()
        seg_idx = int(clip_onset_time // segment_length)
        segment_scores_file[file_id][
            seg_idx : seg_idx + len(segment_scores_clip)
        ] += segment_scores_clip
        summand_count[file_id][seg_idx : seg_idx + len(segment_scores_clip)] += 1
    return {
        file_id: create_score_dataframe(
            segment_scores_file[file_id] / np.maximum(summand_count[file_id], 1),
            np.minimum(
                np.arange(
                    0.0, audio_durations[file_id] + segment_length, segment_length
                ),
                audio_durations[file_id],
            ),
            event_classes,
        )
        for file_id in segment_scores_file
    }


def _get_segment_scores(scores_df, clip_length, segment_length=1.0):
    """
    >>> scores_arr = np.random.rand(156,3)
    >>> timestamps = np.arange(157)*0.064
    >>> event_classes = ['a', 'b', 'c']
    >>> scores_df = create_score_dataframe(scores_arr, timestamps, event_classes)
    >>> seg_scores_df = _get_segment_scores(scores_df, clip_length=10., segment_length=1.)
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
        while (
            seg_offset_idx < len(scores_arr)
            and frame_timestamps[seg_offset_idx] < seg_offset
        ):
            seg_offset_idx += 1
        seg_weights = np.minimum(
            frame_timestamps[seg_onset_idx + 1 : seg_offset_idx + 1], seg_offset
        ) - np.maximum(frame_timestamps[seg_onset_idx:seg_offset_idx], seg_onset)
        segment_scores.append(
            (seg_weights[:, None] * scores_arr[seg_onset_idx:seg_offset_idx]).sum(0)
            / seg_weights.sum()
        )
        segment_timestamps.append(seg_onset)
    segment_timestamps.append(clip_length)
    return create_score_dataframe(
        np.array(segment_scores), np.array(segment_timestamps), event_classes
    )

def get_sebbs(self, scores_all_classes, model_type='student'):
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
    if model_type == 'teacher':
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
        clip_id: scores_all_classes[clip_id][keys_desed]
        for clip_id in scores_all_classes.keys()
    }

    # cSEBBsでDESEDデータセットのスコアを後処理
    # 戻り値: {clip_id: [(onset, offset, class_name, confidence), ...]}
    csebbs_desed_events = csebbs_predictor_desed.predict(
        scores_desed_classes
    )

    # ステップ2: MAESTROクラスに対するcSEBBs適用
    # MAESTROは都市音・屋内音（足音、会話、車など）の17クラス
    maestro_classes = sorted(classes_labels_maestro_real_eval)
    keys_maestro = ["onset", "offset"] + sorted(maestro_classes)

    # 全クラスのスコアからMAESTROクラスのみを抽出
    scores_maestro_classes = {
        clip_id: scores_all_classes[clip_id][keys_maestro]
        for clip_id in scores_all_classes.keys()
    }

    # cSEBBsでMAESTROデータセットのスコアを後処理
    # DESEDとは異なる音響特性に最適化されたパラメータを使用
    # 戻り値: {clip_id: [(onset, offset, class_name, confidence), ...]}
    csebbs_maestro_events = csebbs_predictor_maestro.predict(
        scores_maestro_classes
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
            key=lambda x: x[0]  # onset時間でソート
        )

    # ステップ4: sed_scores_eval形式への変換
    # イベント候補リストからsed_scores_eval形式のDataFrameに変換
    # - 各イベントの時間範囲で該当クラスのスコアを設定
    # - イベントが存在しない時間・クラスは0.0で埋める
    # - 評価ツール(sed_scores_eval)で直接使用可能な形式
    sed_scores_postprocessed = sed_scores_from_sebbs(
        sebbs_all_events,
        sound_classes=all_sound_classes,
        fill_value=0.0  # イベントが存在しない箇所は0.0
    )

    return sed_scores_postprocessed