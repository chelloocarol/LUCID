"""Run latency/FPS benchmark on Modal A10G GPU."""
import os
import subprocess

import modal

LUCID = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "einops==0.8.0",
        "numpy",
        "Pillow",
        "open_clip_torch==2.24.0",
    )
    .add_local_dir(f"{LUCID}/model", remote_path="/workspace/LUCID/model")
    .add_local_dir(f"{LUCID}/CLIP", remote_path="/workspace/LUCID/CLIP")
    .add_local_dir(f"{LUCID}/weights", remote_path="/workspace/LUCID/weights")
    .add_local_file(f"{LUCID}/tools/benchmark_latency.py", remote_path="/workspace/LUCID/tools/benchmark_latency.py")
)

app = modal.App("lucidmine-benchmark", image=image)


@app.function(gpu="A10G", timeout=300)
def run_benchmark():
    import subprocess
    results = {}
    for model in ["student", "lucidmine"]:
        ckpt = "/workspace/LUCID/weights/Student.pth" if model == "student" else ""
        cmd = [
            "python", "/workspace/LUCID/tools/benchmark_latency.py",
            "--model", model,
            "--height", "1080",
            "--width", "1920",
            "--device", "cuda",
        ]
        if ckpt:
            cmd += ["--checkpoint", ckpt]
        out = subprocess.check_output(cmd, cwd="/workspace/LUCID", text=True)
        print(out)
        results[model] = out
    return results


@app.local_entrypoint()
def main():
    print("Running latency benchmark on Modal A10G ...")
    results = run_benchmark.remote()
    for model, out in results.items():
        print(f"\n=== {model} ===")
        print(out)
