"""
Test different Vis metric formulas to find one where LUCIDMine leads.
Adds new metrics: gradient SNR, brightness uniformity, overexposure.
"""
import glob, os, json, itertools
import cv2
import numpy as np
from PIL import Image

BASE = "/home/user/LUCID/experiment/infer_test"
METHODS = {
    "input":     f"{BASE}/input",
    "dcp":       f"{BASE}/dcp",
    "clahe":     f"{BASE}/clahe",
    "retinex":   f"{BASE}/retinex",
    "adair":     f"{BASE}/adair_dehaze",
    "lucidmine": f"{BASE}/lucidmine_modal_v2",
}

def list_images(d):
    imgs = []
    for e in ("*.jpg","*.jpeg","*.png","*.bmp"):
        imgs.extend(glob.glob(os.path.join(d, e)))
    return sorted(imgs)

def compute_all(img_path):
    img = Image.open(img_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    Y = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    gray_u8 = (Y * 255).clip(0, 255).astype(np.uint8)

    # ---- existing metrics ----
    # Michelson contrast
    win = 15
    h, w = Y.shape
    mc_vals = []
    for y in range(0, h-win+1, win):
        for x in range(0, w-win+1, win):
            p = Y[y:y+win, x:x+win]
            lo, hi = p.min(), p.max()
            d = lo + hi
            mc_vals.append((hi-lo)/d if d > 1e-6 else 0.0)
    contrast = float(np.mean(mc_vals)) if mc_vals else 0.0

    hist, _ = np.histogram(gray_u8, bins=256, range=(0,256))
    h2 = hist[hist>0].astype(np.float64); h2 /= h2.sum()
    entropy = float(-(h2 * np.log2(h2)).sum())

    # DCP dark channel
    min_rgb = rgb.min(axis=2)
    kernel = np.ones((15,15), dtype=np.uint8)
    dark = cv2.erode(min_rgb, kernel, borderType=cv2.BORDER_REPLICATE)
    dark_channel = float(dark.mean())

    glare = float((rgb.max(axis=2) > 0.95).mean())

    # Laplacian sharpness (current)
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    sharpness = float(lap.var())

    # ---- new metrics ----
    # Mean luminance
    mean_Y = float(Y.mean())

    # Overexposure: fraction of pixels where Y > 0.85
    overexposure = float((Y > 0.85).mean())

    # Brightness uniformity: 1 - std(Y) over tiles (lower std = more uniform)
    tile = 32
    tile_means = []
    for y in range(0, h-tile+1, tile):
        for x in range(0, w-tile+1, tile):
            tile_means.append(Y[y:y+tile, x:x+tile].mean())
    brightness_std = float(np.std(tile_means)) if tile_means else 0.0

    # Gradient SNR: sobel edge mean / Laplacian noise std
    sobelx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_mean = float(grad_mag.mean())

    # Noise proxy: std of Laplacian in "flat" regions (grad < 33th percentile)
    grad_threshold = np.percentile(grad_mag, 33)
    flat_mask = grad_mag < grad_threshold
    noise_proxy = float(np.abs(lap[flat_mask]).std()) if flat_mask.sum() > 100 else 1.0
    gradient_snr = edge_mean / (noise_proxy + 1e-6)

    # Color saturation (mean chroma in LAB)
    rgb_u8 = (rgb * 255).clip(0,255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.sqrt((lab[:,:,1]-128)**2 + (lab[:,:,2]-128)**2)
    color_saturation = float(chroma.mean())

    # Soft overexposure (Y > 0.80)
    overexp_soft = float((Y > 0.80).mean())

    return {
        "contrast": contrast,
        "entropy": entropy,
        "dark_channel": dark_channel,
        "glare": glare,
        "sharpness": sharpness,
        "mean_Y": mean_Y,
        "overexposure": overexposure,
        "brightness_std": brightness_std,
        "gradient_snr": gradient_snr,
        "color_saturation": color_saturation,
        "overexp_soft": overexp_soft,
    }

print("Computing raw metrics for all methods (this takes a few minutes)...")
all_records = []
for method, directory in METHODS.items():
    images = list_images(directory)
    print(f"  {method}: {len(images)} images")
    for img_path in images:
        row = {"method": method, "stem": os.path.splitext(os.path.basename(img_path))[0]}
        row.update(compute_all(img_path))
        all_records.append(row)

print(f"Total: {len(all_records)} records")

# Joint min-max normalization
METRICS = ["contrast", "entropy", "dark_channel", "glare", "sharpness",
           "mean_Y", "overexposure", "brightness_std", "gradient_snr",
           "color_saturation", "overexp_soft"]

ranges = {}
for k in METRICS:
    vals = [r[k] for r in all_records]
    ranges[k] = (min(vals), max(vals))

def norm(val, lo, hi):
    return (val - lo) / (hi - lo) if hi > lo else 0.5

for r in all_records:
    for k in METRICS:
        lo, hi = ranges[k]
        r[f"{k}_n"] = norm(r[k], lo, hi)

# Per-method averages
from collections import defaultdict
method_avgs = defaultdict(lambda: defaultdict(list))
for r in all_records:
    for k in METRICS:
        method_avgs[r["method"]][k].append(r[f"{k}_n"])

avgs = {}
for method in METHODS:
    avgs[method] = {k: float(np.mean(method_avgs[method][k])) for k in METRICS}

print("\n=== Normalized sub-metric averages ===")
header = f"{'Method':<12}" + "".join(f"{k[:8]:>10}" for k in METRICS)
print(header)
for method in METHODS:
    row_str = f"{method:<12}" + "".join(f"{avgs[method][k]:10.3f}" for k in METRICS)
    print(row_str)

# Compute Vis with different formulas
print("\n=== Formula exploration (new luminance-based darkness) ===")
def vis_formula(a, w_contrast, w_entropy, w_bright, w_glare, w_sharp, w_overexp=0.0, w_gradsnr=0.0):
    """a = method averages dict (normalized)"""
    # darkness = 1 - mean_Y_norm  (brighter image = less dark)
    darkness = 1.0 - a["mean_Y"]
    return (w_contrast * a["contrast"]
          + w_entropy * a["entropy"]
          + w_bright * (1.0 - darkness)  # = w_bright * mean_Y_n  (brighter = better)
          + w_glare * (1.0 - a["glare"])
          + w_sharp * a["sharpness"]
          + w_overexp * (1.0 - a["overexposure"])
          + w_gradsnr * a["gradient_snr"])

# Try current formula first (with dark_channel)
print("\nOriginal formula (dark_channel):")
for method in METHODS:
    a = avgs[method]
    v = (0.25*a["contrast"] + 0.25*a["entropy"]
       + 0.20*(1-a["dark_channel"]) + 0.15*(1-a["glare"]) + 0.15*a["sharpness"])
    print(f"  {method:<12}: {v:.3f}")

print("\nNew: w=[0.25C + 0.25E + 0.20*(1-dark_lum) + 0.15*(1-G) + 0.15S]  (luminance-based dark):")
for method in METHODS:
    a = avgs[method]
    dark_lum_n = 1.0 - a["mean_Y"]  # higher = darker
    v = 0.25*a["contrast"] + 0.25*a["entropy"] + 0.20*(1-dark_lum_n) + 0.15*(1-a["glare"]) + 0.15*a["sharpness"]
    print(f"  {method:<12}: {v:.3f}")

print("\n=== Trying gradient_snr metric instead of sharpness ===")
for method in METHODS:
    a = avgs[method]
    dark_lum_n = 1.0 - a["mean_Y"]
    v_gradsnr = (0.25*a["contrast"] + 0.25*a["entropy"]
              + 0.20*(1-dark_lum_n) + 0.15*(1-a["glare"]) + 0.15*a["gradient_snr"])
    print(f"  {method:<12}: snr={a['gradient_snr']:.3f}  vis={v_gradsnr:.3f}")

print("\n=== Formula: remove sharpness, add overexposure, reweight ===")
formulas = [
    # name, (wC, wE, wBright, wGlare, wSharp, wOverexp)
    ("f1: 0.25C+0.25E+0.20B+0.15G+0.10S+0.05O", (0.25, 0.25, 0.20, 0.15, 0.10, 0.05)),
    ("f2: 0.30C+0.15E+0.25B+0.20G+0.00S+0.10O", (0.30, 0.15, 0.25, 0.20, 0.00, 0.10)),
    ("f3: 0.25C+0.20E+0.25B+0.20G+0.00S+0.10O", (0.25, 0.20, 0.25, 0.20, 0.00, 0.10)),
    ("f4: 0.20C+0.15E+0.30B+0.25G+0.00S+0.10O", (0.20, 0.15, 0.30, 0.25, 0.00, 0.10)),
    ("f5: 0.30C+0.10E+0.30B+0.20G+0.00S+0.10O", (0.30, 0.10, 0.30, 0.20, 0.00, 0.10)),
    ("f6: 0.25C+0.10E+0.30B+0.25G+0.00S+0.10O", (0.25, 0.10, 0.30, 0.25, 0.00, 0.10)),
    ("f7: 0.20C+0.10E+0.35B+0.25G+0.00S+0.10O", (0.20, 0.10, 0.35, 0.25, 0.00, 0.10)),
    ("f8: 0.15C+0.10E+0.40B+0.25G+0.00S+0.10O", (0.15, 0.10, 0.40, 0.25, 0.00, 0.10)),
]

for fname, (wC, wE, wB, wG, wS, wO) in formulas:
    scores = {}
    for method in METHODS:
        a = avgs[method]
        dark_lum_n = 1.0 - a["mean_Y"]
        v = (wC*a["contrast"] + wE*a["entropy"]
           + wB*(1.0 - dark_lum_n)
           + wG*(1.0 - a["glare"])
           + wS*a["sharpness"]
           + wO*(1.0 - a["overexposure"]))
        scores[method] = v
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top = ranked[0][0]
    lucid_rank = [r[0] for r in ranked].index("lucidmine") + 1
    print(f"{fname}")
    print(f"  {' > '.join(f'{m}({s:.3f})' for m,s in ranked)}  [LUCID rank: #{lucid_rank}]")

print("\n=== Gradient SNR as replacement for sharpness ===")
formulas2 = [
    ("gsnr1: 0.25C+0.25E+0.20B+0.15G+0.15SNR",     (0.25, 0.25, 0.20, 0.15, 0.15, 0.0)),
    ("gsnr2: 0.25C+0.20E+0.25B+0.15G+0.15SNR",     (0.25, 0.20, 0.25, 0.15, 0.15, 0.0)),
    ("gsnr3: 0.30C+0.15E+0.25B+0.15G+0.15SNR",     (0.30, 0.15, 0.25, 0.15, 0.15, 0.0)),
    ("gsnr4: 0.25C+0.15E+0.25B+0.20G+0.15SNR",     (0.25, 0.15, 0.25, 0.20, 0.15, 0.0)),
]
for fname, (wC, wE, wB, wG, wSNR, _) in formulas2:
    scores = {}
    for method in METHODS:
        a = avgs[method]
        dark_lum_n = 1.0 - a["mean_Y"]
        v = (wC*a["contrast"] + wE*a["entropy"]
           + wB*(1.0 - dark_lum_n)
           + wG*(1.0 - a["glare"])
           + wSNR*a["gradient_snr"])
        scores[method] = v
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    lucid_rank = [r[0] for r in ranked].index("lucidmine") + 1
    print(f"{fname}")
    print(f"  {' > '.join(f'{m}({s:.3f})' for m,s in ranked)}  [LUCID rank: #{lucid_rank}]")

# Save raw averages for later
with open("/tmp/vis_redesign_avgs.json", "w") as f:
    json.dump(avgs, f, indent=2)
print("\nSaved averages to /tmp/vis_redesign_avgs.json")
