#!/usr/bin/env python3
"""
classify_images.py
------------------
Batch‑classify a folder of images with any pretrained model available in `timm`
and save the results to CSV.

CSV columns:
    image       – file name of the image
    label       – human‑readable class name
    label_idx   – integer index of the predicted class

Usage example
-------------
python classify_images.py \
    --images ./500Stimuli \
    --model mobilenetv3_large_100 \
    --out predictions.csv \
    --batch-size 32 \
    --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json
import urllib.request
import tempfile

from PIL import Image
from torchvision.transforms.functional import to_tensor

import torch
from torch.utils.data import DataLoader, Dataset
import timm
import timm.data
import pandas as pd
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def get_imagenet_labels() -> list[str]:
    """
    Load ImageNet‑1k English labels from a bundled text file.
    If the file is missing (e.g. first clone), download it once.
    """
    url = "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
    labels_path = Path(__file__).with_name("imagenet_1k_labels.txt")
    if not labels_path.exists():
        print("Downloading label list …")
        labels_path.write_text(urllib.request.urlopen(url).read().decode("utf‑8"))
    return labels_path.read_text().strip().split("\n")


class ImageFolderDataset(Dataset):
    """
    Returns (tensor, filename) for every image in `root`.
    Uses Pillow to load .tif / .tiff / .png / .jpg etc.,
    converts grayscale to RGB, then hands it to the timm transform.
    """

    def __init__(self, root: Path, transform):
        self.paths = sorted(
            p
            for p in root.iterdir()
            if p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path)

        # Ensure 3‑channel RGB no matter what
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        elif img.mode == "RGBA":  # strip alpha channel
            img = img.convert("RGB")

        tensor = self.transform(img)  # timm’s pipeline → tensor
        return tensor, path.name


def load_model(name: str, device: torch.device):
    model = timm.create_model(name, pretrained=True)
    model.eval()
    model.to(device)
    return model


def build_transform(model):
    cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    return timm.data.create_transform(**cfg)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def run(args):
    device = torch.device(args.device)
    model = load_model(args.model, device)
    transform = build_transform(model)

    dataset = ImageFolderDataset(args.images, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    labels = get_imagenet_labels()

    rows = []
    with torch.inference_mode():
        for batch, names in tqdm(loader, desc="Inferring", unit="batch"):
            batch = batch.to(device, non_blocking=True)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            for name, idx in zip(names, preds):
                rows.append({"image": name, "label": labels[idx], "label_idx": idx})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {len(out_df)} predictions to {args.out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Classify images in a folder with a pretrained timm model."
    )
    p.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to directory containing images.",
    )
    p.add_argument(
        "--model",
        default="mobilenetv3_large_100",
        help="timm model name (default: %(default)s). "
        "Run `python -c 'import timm, pprint; pprint(timm.list_models(pretrained=True))'` "
        "for the full list.",
    )
    p.add_argument("--out", type=Path, required=True, help="CSV output path.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device (default: %(default)s).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.images.is_dir():
        raise SystemExit(f"{args.images} is not a directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
