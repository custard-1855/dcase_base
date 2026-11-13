import glob
import json
import os
from collections import defaultdict
from copy import deepcopy
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import soundfile
import torch
from desed_task.evaluation.evaluation_measures import compute_sed_eval_metrics
from sed_scores_eval.base_modules.scores import create_score_dataframe, validate_score_dataframe
from thop import clever_format, profile
from torch.utils.data.dataloader import DataLoader, default_collate


def resolve_data_paths(config_data):
    """`data.root` からの相対で書かれたパスを絶対パスにする (その場で書き換える).

    文字列の値はすべてパスとみなす. 絶対パスはそのまま残る.
    `root` が無い設定 (optuna.yaml や, 以前に保存した設定) は何も変えない.
    """
    root = config_data.get("root")
    if root is None:
        return config_data
    for key, value in config_data.items():
        if key != "root" and isinstance(value, str):
            config_data[key] = os.path.join(root, value)
    return config_data


def process_tsvs(tsv, alias_map=None):
    if alias_map is None:
        return tsv
    orig_tsv = deepcopy(tsv)
    for k in alias_map.keys():
        if k in tsv["event_label"].unique():
            mask = tsv["event_label"] == k
            tsv.loc[mask, "event_label"] = alias_map[k]
            orig_tsv = pd.concat([orig_tsv, tsv[mask]])
        # use alias_map to duplicate events
        # for each entry in tsv create a new one with the classes map
        # e.g. dog_bark --> dog
    return orig_tsv


def batched_decode_preds(
    strong_preds,
    filenames,
    encoder,
    thresholds=[0.5],
    median_filter=None,
    pad_indx=None,
):
    """Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary

    Args:
        strong_preds: torch.Tensor, batch of strong predictions.
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int, the number of frames for which to apply median window (smoothing).
        pad_indx: list, the list of indexes which have been used for padding.

    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.

    """
    # Init a dataframe per threshold
    scores_raw = {}
    scores_postprocessed = {}
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for j in range(strong_preds.shape[0]):  # over batches
        audio_id = Path(filenames[j]).stem
        filename = audio_id + ".wav"
        c_scores = strong_preds[j]
        if pad_indx is not None:
            true_len = int(c_scores.shape[-1] * pad_indx[j].item())
            c_scores = c_scores[:true_len]
        c_scores = c_scores.transpose(0, 1).detach().cpu().numpy()
        scores_raw[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
            event_classes=encoder.labels,
        )
        if median_filter is not None:
            c_scores = median_filter(c_scores)
        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
            event_classes=encoder.labels,
        )
        for c_th in thresholds:
            pred = c_scores > c_th
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename
            prediction_dfs[c_th] = pd.concat(
                [prediction_dfs[c_th], pred],
                ignore_index=True,
            )

    return scores_raw, scores_postprocessed, prediction_dfs


def convert_to_event_based(weak_dataframe):
    """Convert a weakly labeled DataFrame ('filename', 'event_labels') to a DataFrame strongly labeled
    ('filename', 'onset', 'offset', 'event_label').

    Args:
        weak_dataframe: pd.DataFrame, the dataframe to be converted.

    Returns:
        pd.DataFrame, the dataframe strongly labeled.

    """
    new = []
    for i, r in weak_dataframe.iterrows():
        events = r["event_labels"].split(",")
        for e in events:
            new.append(
                {"filename": r["filename"], "event_label": e, "onset": 0, "offset": 1},
            )
    return pd.DataFrame(new)


def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    """Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1

    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t")

    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))

    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures


def parse_jams(jams_list, encoder, out_json):
    if len(jams_list) == 0:
        raise IndexError("jams list is empty ! Wrong path ?")

    backgrounds = []
    sources = []
    for jamfile in jams_list:
        with open(jamfile) as f:
            jdata = json.load(f)

        # check if we have annotations for each source in scaper
        assert len(jdata["annotations"][0]["data"]) == len(
            jdata["annotations"][-1]["sandbox"]["scaper"]["isolated_events_audio_path"],
        )

        for indx, sound in enumerate(jdata["annotations"][0]["data"]):
            source_name = Path(
                jdata["annotations"][-1]["sandbox"]["scaper"]["isolated_events_audio_path"][indx],
            ).stem
            source_file = os.path.join(
                Path(jamfile).parent,
                Path(jamfile).stem + "_events",
                source_name + ".wav",
            )

            if sound["value"]["role"] == "background":
                backgrounds.append(source_file)
            else:  # it is an event
                if sound["value"]["label"] not in encoder.labels:  # correct different labels
                    if sound["value"]["label"].startswith("Frying"):
                        sound["value"]["label"] = "Frying"
                    elif sound["value"]["label"].startswith("Vacuum_cleaner"):
                        sound["value"]["label"] = "Vacuum_cleaner"
                    else:
                        raise NotImplementedError

                sources.append(
                    {
                        "filename": source_file,
                        "onset": sound["value"]["event_time"],
                        "offset": sound["value"]["event_time"] + sound["value"]["event_duration"],
                        "event_label": sound["value"]["label"],
                    },
                )

    os.makedirs(Path(out_json).parent, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"backgrounds": backgrounds, "sources": sources}, f, indent=4)


def generate_tsv_wav_durations(audio_dir, out_tsv):
    """Generate a dataframe with filename and duration of the file

    Args:
        audio_dir: str, the path of the folder where audio files are (used by glob.glob)
        out_tsv: str, the path of the output tsv file

    Returns:
        pd.DataFrame: the dataframe containing filenames and durations

    """
    meta_list = []
    for file in glob.glob(os.path.join(audio_dir, "*.wav")):
        d = soundfile.info(file).duration
        meta_list.append([os.path.basename(file), d])
    meta_df = pd.DataFrame(meta_list, columns=["filename", "duration"])
    if out_tsv is not None:
        meta_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.1f")

    return meta_df


def calculate_macs(model, config, dataset=None):
    """The function calculate the multiply–accumulate operation (MACs) of the model given as input.

    Args:
        model: deep learning model to calculate the macs for
        config: config used to train the model
        dataset: dataset used to train the model

    Returns:

    """
    n_frames = int(
        (
            (config["feats"]["sample_rate"] * config["data"]["audio_max_len"])
            / config["feats"]["hop_length"]
        )
        + 1,
    )
    input_size = [1, config["feats"]["n_mels"], n_frames]
    input = torch.randn(input_size)

    if "use_embeddings" in config["net"] and config["net"]["use_embeddings"]:
        audio, label, padded_indxs, path, embeddings, _ = dataset[0]
        embeddings = embeddings.repeat(1, 1, 1)
        macs, params = profile(model, inputs=(input, None, embeddings))
    else:
        macs, params = profile(model, inputs=(input,))

    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


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


def _codecarbon_gpu_ids() -> list[int] | None:
    """codecarbon に渡す GPU id. CPU 実行 (--gpus 0) では CUDA に触らない."""
    if torch.cuda.is_available():
        return [torch.cuda.current_device()]
    return None


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
