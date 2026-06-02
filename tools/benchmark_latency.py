"""GPU latency / FPS benchmark for Student and LUCIDMine models.

Usage:
    python tools/benchmark_latency.py --model student --checkpoint weights/Student.pth
    python tools/benchmark_latency.py --model lucidmine --checkpoint <path/to/best.pth>
"""
import argparse
import time

import torch

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import Student, LUCIDMine


WARMUP = 50
RUNS = 200


def build_model(name, ckpt_path):
    model = Student() if name == "student" else LUCIDMine()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            sd = ckpt
        clean = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean, strict=False)
    return model


def benchmark(model, device, h, w, batch_size=1, warmup=WARMUP, runs=RUNS):
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, h, w, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_ms
    return avg_ms, fps, len(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="student", choices=["student", "lucidmine"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    model = build_model(args.model, args.checkpoint)
    avg_ms, fps, n_runs = benchmark(model, args.device, args.height, args.width,
                                    warmup=args.warmup, runs=args.runs)

    gpu_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    print(f"Model      : {args.model}")
    print(f"Device     : {args.device} ({gpu_name})")
    print(f"Resolution : {args.height}x{args.width}")
    print(f"Latency    : {avg_ms:.2f} ms  (avg over {n_runs} runs, {args.warmup} warmup)")
    print(f"FPS        : {fps:.1f}")


if __name__ == "__main__":
    main()
