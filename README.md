# Image‑Labeller

> Batch‑classify a folder of images with any pretrained vision model shipped in **[timm](https://github.com/huggingface/pytorch-image-models)** and export the predictions to CSV.

---

## Quick start

```bash
# clone the repo you created on GitHub
git clone git@github.com:<user>/image-classifier.git
cd image-classifier

# create Conda environment (Python 3.11, CPU build of PyTorch)
conda env create -f env.yml
conda activate image-labeller
```

Move or symlink your image folder into the repo (e.g. `data/500Stimuli/`).  
Then run:

```bash
python classify_images.py \
    --images data/500Stimuli \
    --model mobilenetv3_large_100 \
    --out predictions.csv \
    --batch-size 32            # tune for your CPU / GPU \
```

The script prints progress and writes `predictions.csv` with three columns:

| image | label | label_idx |
|-------|-------|-----------|
| 0001.tif | tabby, tabby_cat | 281 |
| 0003.tif | Egyptian_cat | 285 |
| … | … | … |

---

## Command‑line reference

| Flag | Default | Description |
|------|---------|-------------|
| `--images PATH` | — | Directory containing images (`.tif`, `.png`, `.jpg`, …). |
| `--model NAME` | `mobilenetv3_large_100` | Any name returned by `timm.list_models(pretrained=True)`. |
| `--out FILE` | — | CSV output path. Parent dirs are created automatically. |
| `--batch-size N` | `32` | Increase for GPUs, lower for CPU boxes. |
| `--device {cpu,cuda}` | `cpu` | Inference device. |

---

## Extending / customising

* **Different models** – Just pass another `--model` name; the script will download its weights automatically on first use and resolve the correct preprocessing pipeline.
* **CUDA** – Create the environment on a CUDA‑enabled host and replace the two PyTorch lines in `env.yml` with  
  `- pytorch::pytorch pytorch::pytorch-cuda=11.8 torchvision`.
* **Non‑ImageNet models** – For models trained on other class sets, replace `get_imagenet_labels()` with your own label‑list loader.

---

## License

MIT © 2025 Your Name
