r"""Extract the ROAD-R video frames needed by the ``road-r`` task.

For every annotated key-frame ``f`` (a multiple of ``--stride``) the frames of
the clip ending at ``f`` (``f - i * clip_frame_stride`` for ``i < clip_len``)
are written to::

    fonesy/data/road-r/rgb-images/<video>/<frame:05d>.jpg

Frame numbers (OpenCV's 1-based read index) match the annotation file.

Usage::

    python scripts/extract_road_r_frames.py [--stride 10] [--clip-len 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ROOT = _REPO_ROOT / "fonesy" / "data" / "road-r"


def _wanted_frame_numbers(
    video_data: dict, stride: int, clip_len: int, clip_frame_stride: int
) -> set[int]:
    """1-based frame numbers needed to build every key-frame's clip.

    A key-frame ``f`` is annotated, carries boxes and satisfies ``f % stride ==
    0``; its clip is ``{f - i*cs : i in [0, clip_len)}`` (clamped to >= 1).
    """
    numf = int(video_data.get("numf", 0)) or None
    wanted: set[int] = set()
    for frame_num_str, frame_data in video_data["frames"].items():
        f = int(frame_num_str)
        if f % stride != 0:
            continue
        if not (frame_data.get("annotated") and frame_data.get("annos")):
            continue
        for i in range(clip_len):
            n = f - i * clip_frame_stride
            if n < 1:
                n = 1
            if numf is None or n <= numf:
                wanted.add(n)
    return wanted


def _extract_one(video_path: Path, wanted: set[int], out_dir: Path, quality: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = {n for n in wanted if not (out_dir / f"{n:05d}.jpg").exists()}
    if not todo:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    last = max(todo)
    idx = 0
    written = 0
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1  # 1-based, matches the annotation file
        if idx in todo:
            cv2.imwrite(
                str(out_dir / f"{idx:05d}.jpg"),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            written += 1
    cap.release()
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=_DEFAULT_ROOT,
                   help=f"ROAD-R data root (default: {_DEFAULT_ROOT})")
    p.add_argument("--videos-dir", type=Path, default=None,
                   help="Directory holding <video_name>.mp4 (default: <root>/videos)")
    p.add_argument("--stride", type=int, default=10,
                   help="Key-frame stride: a box at frame f is used only if f %% stride == 0 "
                        "(default: 10, matching RoadRTask's default frame_stride)")
    p.add_argument("--clip-len", type=int, default=8,
                   help="Frames per clip ending at each key-frame (default: 8, matching "
                        "RoadRTask's _CLIP_LEN for the SlowFast backbone; use 1 for single frames)")
    p.add_argument("--clip-frame-stride", type=int, default=1,
                   help="Spacing between the frames of one clip (default: 1, = consecutive; "
                        "matches RoadRTask's _CLIP_FRAME_STRIDE)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    args = p.parse_args()

    annotation_path = args.root / "road_trainval_v1.0.json"
    videos_dir = args.videos_dir or (args.root / "videos")
    frames_root = args.root / "rgb-images"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    db = json.loads(annotation_path.read_text())["db"]
    total_written = 0
    for i, (video_name, video_data) in enumerate(sorted(db.items()), 1):
        video_path = videos_dir / f"{video_name}.mp4"
        if not video_path.exists():
            print(f"[{i}/{len(db)}] SKIP {video_name} (no mp4)")
            continue
        wanted = _wanted_frame_numbers(
            video_data, args.stride, args.clip_len, args.clip_frame_stride
        )
        out_dir = frames_root / video_name
        written = _extract_one(video_path, wanted, out_dir, args.quality)
        total_written += written
        have = len(list(out_dir.glob("*.jpg")))
        print(f"[{i}/{len(db)}] {video_name}: wanted {len(wanted)}, wrote {written}, "
              f"on disk {have}")

    print(f"\nDone. Wrote {total_written} new frames under {frames_root}")


if __name__ == "__main__":
    main()
