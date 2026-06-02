"""
Modal GPU training script for LUCIDMine.

Usage:
  modal run modal_train.py
  modal run modal_train.py --detach   # fire-and-forget

The script:
  1. Mounts the local LUCID repo and dataset into Modal
  2. Runs full 100-epoch training on an A10G GPU
  3. Saves checkpoints to /results/ inside the Modal volume,
     and downloads best.pth / last.pth back to local ./experiment/LUCIDMine/modal_run/

Prerequisites:
  pip install modal
  modal token new          # one-time login
"""
import os
import sys

import modal

# ---- Modal app ----
app = modal.App("lucidmine-train")

MINUTES = 60
HOURS   = 60 * MINUTES

# Persistent volume for checkpoints
volume = modal.Volume.from_name("lucidmine-checkpoints", create_if_missing=True)
VOLUME_PATH = "/results"

# Image: start from official PyTorch image, add deps
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
)

# Local mounts
lucid_mount   = modal.Mount.from_local_dir("/home/user/LUCID",                          remote_path="/workspace/LUCID")
dataset_mount = modal.Mount.from_local_dir("/home/user/lucidmine-40-video-dataset/data", remote_path="/workspace/dataset")


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * HOURS,
    volumes={VOLUME_PATH: volume},
    mounts=[lucid_mount, dataset_mount],
)
def train():
    import subprocess, shutil, sys

    sys.path.insert(0, "/workspace/LUCID")
    os.makedirs(f"{VOLUME_PATH}/lucidmine_modal", exist_ok=True)

    # Rewrite manifest paths inside Modal (dataset is at /workspace/dataset)
    import csv

    DS_ROOT = "/workspace/dataset"
    manifest_out = f"{VOLUME_PATH}/mine_manifest.csv"
    rows = []
    for split in ["train", "val", "test"]:
        src = f"/workspace/LUCID/data/mine_manifest_{split}.csv"
        with open(src, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem  = row["stem"]
                video = row["video"]
                row["input_path"]  = f"{DS_ROOT}/{split}/input/{video}__{stem}.jpg"
                row["target_path"] = f"{DS_ROOT}/{split}/target/{video}__{stem}.jpg"
                row["mask_path"]   = f"{DS_ROOT}/{split}/mask/{video}__{stem}.png"
                rows.append(row)
    with open(manifest_out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest written: {len(rows)} rows")

    # Run training
    result = subprocess.run(
        [
            sys.executable, "/workspace/LUCID/train_lucidmine.py",
            "--manifest",      manifest_out,
            "--init_ckpt",     "/workspace/LUCID/weights/Student.pth",
            "--exp_name",      "lucidmine_modal",
            "--exp_dir",       VOLUME_PATH,
            "--warmup_epochs", "20",
            "--adapt_epochs",  "80",
            "--batch_size",    "8",
            "--crop_size",     "256",
            "--num_workers",   "4",
        ],
        check=True,
    )

    # Sync volume
    volume.commit()
    print("Training complete. Checkpoints in Modal volume 'lucidmine-checkpoints'.")
    return f"{VOLUME_PATH}/lucidmine_modal/best.pth"


@app.local_entrypoint()
def main():
    print("Submitting LUCIDMine training job to Modal (A10G GPU)...")
    result = train.remote()
    print(f"Job complete. Best checkpoint: {result}")

    # Download best and last checkpoints locally
    import subprocess
    os.makedirs("experiment/LUCIDMine/modal_run", exist_ok=True)
    subprocess.run([
        "modal", "volume", "get",
        "lucidmine-checkpoints", "lucidmine_modal/best.pth",
        "experiment/LUCIDMine/modal_run/best.pth"
    ])
    subprocess.run([
        "modal", "volume", "get",
        "lucidmine-checkpoints", "lucidmine_modal/last.pth",
        "experiment/LUCIDMine/modal_run/last.pth"
    ])
    print("Downloaded to experiment/LUCIDMine/modal_run/")
