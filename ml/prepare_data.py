"""Prepare the coarse-v0 YOLOv8-cls dataset.

Maps the Kaggle dog-skin dataset's coarse class folders to our canonical labels
(see label_map.json) and writes an Ultralytics classification layout:

    <out>/train/<canonical>/*.jpg
    <out>/val/<canonical>/*.jpg

Stdlib only — runs without torch/ultralytics, so the mapping can be verified
before touching a GPU box.

Usage:
    # use an already-downloaded copy (folder containing the class subdirs):
    python prepare_data.py --source-dir /path/to/Dogs --out ./prepared
    # or download from Kaggle first (needs ~/.kaggle/kaggle.json + `kaggle` CLI):
    python prepare_data.py --download --out ./prepared
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL_MAP = json.loads((HERE / "label_map.json").read_text())
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_class_root(base: Path) -> Path:
    """Locate the directory whose immediate subdirs are the source classes."""
    wanted = set(LABEL_MAP["map"].keys())
    for d in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        subdirs = {c.name for c in d.iterdir() if c.is_dir()}
        if wanted & subdirs:
            return d
    raise SystemExit(f"Could not find source class folders {sorted(wanted)} under {base}")


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    ref = LABEL_MAP["dataset"]
    print(f"Downloading {ref} from Kaggle …")
    subprocess.run(["kaggle", "datasets", "download", "-d", ref, "--unzip", "-p", str(dest)], check=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, help="Existing folder containing the class subdirs")
    ap.add_argument("--download", action="store_true", help="Download from Kaggle instead")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = ap.parse_args()

    if args.download:
        base = download(args.out.parent / "_raw")
    elif args.source_dir:
        base = args.source_dir
    else:
        raise SystemExit("Provide --source-dir or --download")

    class_root = find_class_root(base)
    rng = random.Random(args.seed)
    if args.out.exists():
        shutil.rmtree(args.out)

    totals: dict[str, dict[str, int]] = {}
    for src_name, canonical in LABEL_MAP["map"].items():
        src = class_root / src_name
        if not src.is_dir():
            print(f"  ! missing source class: {src_name} (skipping)")
            continue
        imgs = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(imgs)
        n_val = round(len(imgs) * args.val_frac)
        for split, files in (("val", imgs[:n_val]), ("train", imgs[n_val:])):
            dest_dir = args.out / split / canonical
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(files):
                dest = dest_dir / f"{src_name}_{i:04d}{p.suffix.lower()}"
                if args.copy:
                    shutil.copy2(p, dest)
                else:
                    dest.symlink_to(p.resolve())
            totals.setdefault(canonical, {"train": 0, "val": 0})[split] += len(files)

    print(f"\nPrepared dataset at {args.out} (val_frac={args.val_frac}, seed={args.seed})")
    print(f"{'canonical label':30} {'train':>6} {'val':>5} {'total':>6}")
    gt = gv = 0
    for label, c in sorted(totals.items()):
        print(f"{label:30} {c['train']:>6} {c['val']:>5} {c['train'] + c['val']:>6}")
        gt += c["train"]
        gv += c["val"]
    print(f"{'TOTAL':30} {gt:>6} {gv:>5} {gt + gv:>6}")
    print("\nUnsupported in v0 (no data):", ", ".join(LABEL_MAP["unsupported_in_v0"]))


if __name__ == "__main__":
    main()
