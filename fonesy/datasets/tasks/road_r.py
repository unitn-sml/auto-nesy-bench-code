"""ROAD-R (``road-r``): video clip of one detected road agent -> 41 labels.

Each sample is an 8-frame clip ending at an annotated key-frame, cropped to
the detection box: ``(3, 8, 96, 96)``, for the SlowFast backbone. Labels are
the agent (10), traffic-light color (3), actions (16) and locations (12) of
the box. ``make road-r`` downloads the ROAD videos and annotations and
extracts the frames::

    fonesy/data/road-r/road_trainval_v1.0.json
    fonesy/data/road-r/rgb-images/<video>/<frame:05d>.jpg

``_AGENT_ID_TO_VAR`` / ``_ACTION_ID_TO_VAR`` / ``_LOC_ID_TO_VAR`` map the
annotation ids to variables (classes not in the requirements are dropped).
Boxes violating the ROAD-R requirements are discarded. Splits are 80/10/10
by video.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from fonesy.datasets.tasks.base import Task
from fonesy.general_utils import log
from fonesy.utils.oracles import load_oracle

_ORACLE_VARIABLE_NAMES = load_oracle("road-r").VARIABLE_NAMES
_road_r_check = load_oracle("road-r").check

N_VARS = 41
_DEFAULT_FRAME_STRIDE = 10
_CROP_SIZE = 96
_CLIP_LEN = 8              # frames per clip (SlowFast SEQ_LEN); keep in sync with extractor
_CLIP_FRAME_STRIDE = 1     # spacing between a clip's frames (1 = consecutive)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# Raw id -> 0-based index in the 41-dim label vector (None = class the oracle
# does not model). Derived by name-matching the annotation file's label lists
# against road_r.py's VARIABLE_NAMES:
#   agent_ids  index into all_agent_labels (11): "SmalVeh" (id 4) is dropped.
#   action_ids index into all_action_labels (22): "Rev"/"MovRht"/"MovLft"
#              (ids 6/14/15) are dropped; the rest keep order as vars 10-28.
#   loc_ids    index into loc_labels (12): identical order -> vars 29-40.
_AGENT_ID_TO_VAR = [0, 1, 2, 3, None, 4, 5, 6, 7, 8, 9]
_ACTION_ID_TO_VAR = [
    10, 11, 12, 13, 14, 15, None, 16, 17, 18, 19,
    20, 21, 22, None, None, 23, 24, 25, 26, 27, 28,
]
_LOC_ID_TO_VAR = list(range(29, 41))
_ID_MAPS = {
    "agent_ids": _AGENT_ID_TO_VAR,
    "action_ids": _ACTION_ID_TO_VAR,
    "loc_ids": _LOC_ID_TO_VAR,
}


class _RoadRAnnotation:
    """One (video, frame, box) sample: crop coordinates + a 41-dim label vector."""

    __slots__ = ("video_name", "frame_num", "box", "labels")

    def __init__(self, video_name: str, frame_num: int, box: list[float], labels: list[int]) -> None:
        self.video_name = video_name
        self.frame_num = frame_num
        self.box = box  # [x1, y1, x2, y2], normalized 0-1
        self.labels = labels  # 41-dim 0/1, VARIABLE_NAMES order


def _anno_to_labels(anno: dict) -> list[int] | None:
    """Map one box's ``agent_ids``/``action_ids``/``loc_ids`` to a 41-dim 0/1 vector.

    Ids for classes the oracle doesn't model are ignored; returns ``None`` only
    if an id is outside its list entirely (a genuinely malformed entry).
    """
    labels = [0] * N_VARS
    for key, id_map in _ID_MAPS.items():
        for raw_id in anno.get(key, []):
            if not 0 <= raw_id < len(id_map):
                return None
            var = id_map[raw_id]
            if var is not None:
                labels[var] = 1
    return labels


def _parse_annotations(annotation_path: Path) -> list[_RoadRAnnotation]:
    """Parse ``road_trainval_v1.0.json`` into a flat list of per-box annotations.

    Only frames flagged ``annotated`` and carrying at least one box are kept;
    boxes with an out-of-range label id are skipped. Logical-requirement
    filtering happens later, in ``RoadRTask`` (it needs the frame-stride subset
    first, to keep the ~476k -> validity check affordable).
    """
    data = json.loads(annotation_path.read_text())

    annotations: list[_RoadRAnnotation] = []
    skipped_malformed = 0
    for video_name, video_data in data["db"].items():
        for frame_num_str, frame_data in video_data["frames"].items():
            if not (frame_data.get("annotated") and frame_data.get("annos")):
                continue
            frame_num = int(frame_num_str)
            for anno in frame_data["annos"].values():
                labels = _anno_to_labels(anno)
                if labels is None:
                    skipped_malformed += 1
                    continue
                annotations.append(
                    _RoadRAnnotation(video_name, frame_num, anno["box"], labels)
                )
    if skipped_malformed:
        log(f"ROAD-R: skipped {skipped_malformed} boxes with out-of-range label ids", "WARNING")
    return annotations


def _split_by_video(annotations: list[_RoadRAnnotation]) -> tuple[list, list, list]:
    """Deterministic 80/10/10 split by video name (never split one video across sets)."""
    video_names = sorted({a.video_name for a in annotations})
    n = len(video_names)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1)) if n > 1 else 0
    train_videos = set(video_names[:n_train])
    val_videos = set(video_names[n_train : n_train + n_val])
    test_videos = set(video_names[n_train + n_val :])

    train = [a for a in annotations if a.video_name in train_videos]
    val = [a for a in annotations if a.video_name in val_videos]
    test = [a for a in annotations if a.video_name in test_videos]
    return train, val, test


def _box_pixels(box: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    """Map a (maybe normalized) ``[x1,y1,x2,y2]`` box to a non-degenerate pixel box."""
    x1, y1, x2, y2 = box
    # Box coords may be normalized (0-1) or already pixel-space; treat as
    # normalized if all four are within [0, 1] (the common ROAD-R convention).
    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
    if x2 - x1 < 1:
        x2 = min(float(w), x1 + 1)
    if y2 - y1 < 1:
        y2 = min(float(h), y1 + 1)
    return x1, y1, x2, y2


class _RoadRSplit(Dataset):
    """Yields ``(clip, labels[41])`` — ``clip`` is ``(3, T, _CROP_SIZE, _CROP_SIZE)``,
    the ``T`` frames ending at the annotated key-frame, each cropped to the box."""

    def __init__(
        self,
        root: Path,
        annotations: list[_RoadRAnnotation],
        transform,
        clip_len: int = _CLIP_LEN,
        clip_frame_stride: int = _CLIP_FRAME_STRIDE,
    ) -> None:
        self._root = root
        self._annotations = annotations
        self._transform = transform
        self._clip_len = clip_len
        self._clip_frame_stride = clip_frame_stride

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int):
        anno = self._annotations[idx]
        video_dir = self._root / "rgb-images" / anno.video_name
        # Oldest -> newest; the key-frame f is last. Frames before the video
        # start clamp to 1 (the extractor writes frame 1 for exactly this).
        frame_nums = [
            max(1, anno.frame_num - i * self._clip_frame_stride)
            for i in range(self._clip_len - 1, -1, -1)
        ]
        crops = []
        for n in frame_nums:
            image = Image.open(video_dir / f"{n:05d}.jpg").convert("RGB")
            w, h = image.size
            crop = image.crop(_box_pixels(anno.box, w, h))
            crops.append(self._transform(crop))
        clip = torch.stack(crops, dim=1)  # (3, T, H, W)
        labels = torch.tensor(anno.labels, dtype=torch.float32)
        return clip, labels


class RoadRTask(Task):
    """ROAD-R: per-box static-image crops, 41 Agent/TL-colour/Action/Location labels."""

    n_classes: int = 2   # unused directly by SPL (no separate classifier head)
    in_channels: int = 3
    image_size: int = _CROP_SIZE

    def __init__(self, root: str, frame_stride: int = _DEFAULT_FRAME_STRIDE) -> None:
        dataset_root = Path(root) / "road-r"
        annotation_path = dataset_root / "road_trainval_v1.0.json"
        frames_dir = dataset_root / "rgb-images"
        if not annotation_path.exists() or not frames_dir.exists():
            raise FileNotFoundError(
                f"ROAD-R data not found at '{dataset_root}'.\n"
                "Run 'make road-r'."
            )
        assert len(_ORACLE_VARIABLE_NAMES) == N_VARS

        all_annotations = _parse_annotations(annotation_path)

        # Keep only the frame-stride subset that was actually extracted to disk,
        # and only for videos whose frame directory exists.
        available_videos = {p.name for p in frames_dir.iterdir() if p.is_dir()}
        sampled = [
            a for a in all_annotations
            if a.frame_num % frame_stride == 0
            and a.video_name in available_videos
            and (frames_dir / a.video_name / f"{a.frame_num:05d}.jpg").exists()
        ]

        # Drop boxes that violate any of the 243 ROAD-R logical requirements:
        # SPL's loss is the circuit NLL of the label vector, which is +inf for
        # an assignment the ground-truth circuit rules out.
        valid = [a for a in sampled if _road_r_check(a.labels)]
        n_dropped = len(sampled) - len(valid)
        log(
            f"ROAD-R: {len(all_annotations)} boxes total; {len(sampled)} in the "
            f"stride-{frame_stride} on-disk subset; dropped {n_dropped} that "
            f"violate a logical requirement; {len(valid)} usable.",
            "INFO",
        )
        if not valid:
            raise RuntimeError(
                "No usable ROAD-R annotations after filtering — check that frames "
                f"were extracted with a stride dividing {frame_stride}."
            )

        train, val, test = _split_by_video(valid)
        log(
            f"ROAD-R splits (by video): train={len(train)} val={len(val)} test={len(test)}",
            "INFO",
        )

        transform = transforms.Compose([
            transforms.Resize((_CROP_SIZE, _CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
        self._train_ds = _RoadRSplit(dataset_root, train, transform)
        self._val_ds = _RoadRSplit(dataset_root, val, transform)
        self._test_ds = _RoadRSplit(dataset_root, test, transform)

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=torch.cuda.is_available(),
        )

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self._val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=torch.cuda.is_available(),
        )

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self._test_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=torch.cuda.is_available(),
        )
