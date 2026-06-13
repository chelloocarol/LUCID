"""
Modal GPU training script for LUCIDMine.

Usage:
  SSL_CERT_FILE=... REQUESTS_CA_BUNDLE=... modal run modal_train.py
  SSL_CERT_FILE=... REQUESTS_CA_BUNDLE=... modal run modal_train.py --detach

The script:
  1. Builds a Modal image with code (model/loss/metric/data/weights) + dataset
  2. Runs full 100-epoch training on an A10G GPU
  3. Saves checkpoints to a Modal Volume
  4. Downloads best.pth / last.pth to experiment/LUCIDMine/modal_run/
"""
import os
from pathlib import Path
import modal

# ---- Modal app ----
app = modal.App("lucidmine-train")

MINUTES = 60
HOURS   = 60 * MINUTES

# Persistent volume for checkpoints
volume = modal.Volume.from_name("lucidmine-checkpoints", create_if_missing=True)
VOLUME_PATH = "/results"

LUCID = str(Path(__file__).parent.resolve())
DATA  = "/home/user/lucidmine-40-video-dataset/data"

# Image: PyTorch + deps + local code (skip experiment/ to save bandwidth)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "einops==0.8.0",
        "Pillow==9.2.0",
        "numpy==1.24.4",
        "opencv-python-headless==4.6.0.66",
        "tqdm",
    )
    # Code directories (lightweight)
    .add_local_dir(f"{LUCID}/model",   remote_path="/workspace/LUCID/model")
    .add_local_dir(f"{LUCID}/loss",    remote_path="/workspace/LUCID/loss")
    .add_local_dir(f"{LUCID}/metric",  remote_path="/workspace/LUCID/metric")
    .add_local_dir(f"{LUCID}/data",    remote_path="/workspace/LUCID/data")
    .add_local_dir(f"{LUCID}/CLIP",    remote_path="/workspace/LUCID/CLIP")
    .add_local_dir(f"{LUCID}/weights", remote_path="/workspace/LUCID/weights")
    # Training script
    .add_local_file(f"{LUCID}/train_lucidmine.py",  remote_path="/workspace/LUCID/train_lucidmine.py")
    # Dataset (568 MB — uploaded once, cached in image layer)
    .add_local_dir(DATA, remote_path="/workspace/dataset")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * HOURS,
    volumes={VOLUME_PATH: volume},
)
def train():
    import csv, os, subprocess, sys

    sys.path.insert(0, "/workspace/LUCID")
    os.makedirs(f"{VOLUME_PATH}/lucidmine_modal", exist_ok=True)

    # Rewrite manifest paths: /home/user/lucidmine-40-video-dataset/data → /workspace/dataset
    LOCAL_PREFIX = "/home/user/lucidmine-40-video-dataset/data"
    DS_ROOT = "/workspace/dataset"
    manifest_in  = "/workspace/LUCID/data/mine_manifest.csv"
    manifest_out = f"{VOLUME_PATH}/mine_manifest.csv"

    with open(manifest_in, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in ("input_path", "target_path", "mask_path"):
            row[col] = row[col].replace(LOCAL_PREFIX, DS_ROOT)
    with open(manifest_out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest rewritten: {len(rows)} rows to {manifest_out}")

    # Run training
    subprocess.run(
        [
            sys.executable, "/workspace/LUCID/train_lucidmine.py",
            "--manifest",      manifest_out,
            "--init_ckpt",     "/workspace/LUCID/weights/Student.pth",
            "--exp_name",      "lucidmine_modal_v2",
            "--exp_dir",       VOLUME_PATH,
            "--warmup_epochs", "20",
            "--adapt_epochs",  "80",
            "--batch_size",    "8",
            "--crop_size",     "256",
            "--num_workers",   "4",
        ],
        check=True,
    )

    volume.commit()
    print("Training complete. Checkpoints in Modal volume 'lucidmine-checkpoints'.")
    return f"{VOLUME_PATH}/lucidmine_modal_v2/best.pth"


@app.local_entrypoint()
def main():
    print("Submitting LUCIDMine training (v2, SSIM fp32 fix) to Modal (A10G GPU) ...")
    result = train.remote()
    print(f"Job complete. Best checkpoint: {result}")

    # Download checkpoints locally
    os.makedirs("experiment/LUCIDMine/modal_run_v2", exist_ok=True)
    for fname in ("best.pth", "last.pth"):
        cmd = ["modal", "volume", "get",
               "lucidmine-checkpoints", f"lucidmine_modal_v2/{fname}",
               f"experiment/LUCIDMine/modal_run_v2/{fname}"]
        import subprocess
        subprocess.run(cmd, check=False)
    print("Downloaded to experiment/LUCIDMine/modal_run_v2/")
