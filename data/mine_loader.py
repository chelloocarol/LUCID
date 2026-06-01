import csv
import random
from pathlib import Path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ColorJitter, GaussianBlur, Resize, ToTensor
from torchvision.transforms import functional as FF
from torchvision.transforms.functional import InterpolationMode

from .data_loader import preprocess_feature


def _read_manifest(manifest_path, split):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    rows = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split is None or row["split"] == split:
                rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {manifest_path} for split={split!r}")

    return rows


def _stem_sort_key(stem):
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else stem


def _ensure_min_size(image, target_size, interpolation):
    if target_size is None:
        return image
    width, height = image.size
    if width >= target_size and height >= target_size:
        return image

    scale = max(target_size / max(width, 1), target_size / max(height, 1))
    new_width = max(target_size, int(round(width * scale)))
    new_height = max(target_size, int(round(height * scale)))
    return Resize((new_height, new_width), interpolation=interpolation, antialias=interpolation != InterpolationMode.NEAREST)(image)


def _center_crop_params(image, crop_size):
    width, height = image.size
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    return top, left, crop_size, crop_size


class MineManifestDataset(data.Dataset):
    def __init__(
        self,
        manifest_path,
        split="train",
        train=True,
        crop_size=256,
        resize=None,
        reliability_floor=0.0,
        return_neighbor=False,
        use_strong_aug=False,
    ):
        super(MineManifestDataset, self).__init__()
        self.records = sorted(_read_manifest(manifest_path, split), key=lambda row: (row["video"], _stem_sort_key(row["stem"])))
        self.split = split
        self.train = train
        self.crop_size = crop_size
        self.resize = resize
        self.reliability_floor = reliability_floor
        self.return_neighbor = return_neighbor
        self.use_strong_aug = use_strong_aug and train
        self.to_tensor = ToTensor()
        self.strong_color_jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03)
        self.strong_blur = GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))
        self.video_to_indices = {}
        for index, row in enumerate(self.records):
            self.video_to_indices.setdefault(row["video"], []).append(index)

    def __len__(self):
        return len(self.records)

    def _choose_neighbor_index(self, index):
        row = self.records[index]
        group = self.video_to_indices[row["video"]]
        group_pos = group.index(index)
        if group_pos < len(group) - 1:
            return group[group_pos + 1]
        if group_pos > 0:
            return group[group_pos - 1]
        return index

    def _load_triplet(self, row):
        haze = Image.open(row["input_path"]).convert("RGB")
        target = Image.open(row["target_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        return haze, target, mask

    def _apply_geometry(self, haze, target, mask, crop_params, rand_hor, rand_rot):
        if crop_params is not None:
            haze = FF.crop(haze, *crop_params)
            target = FF.crop(target, *crop_params)
            mask = FF.crop(mask, *crop_params)

        if rand_hor:
            haze = FF.hflip(haze)
            target = FF.hflip(target)
            mask = FF.hflip(mask)

        if rand_rot:
            angle = 90 * rand_rot
            haze = FF.rotate(haze, angle)
            target = FF.rotate(target, angle)
            mask = FF.rotate(mask, angle)

        return haze, target, mask

    def _prepare_triplet(self, row, crop_params=None, rand_hor=0, rand_rot=0):
        haze, target, mask = self._load_triplet(row)

        if self.resize is not None:
            resize_hw = (self.resize, self.resize) if isinstance(self.resize, int) else self.resize
            haze = Resize(resize_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)(haze)
            target = Resize(resize_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)(target)
            mask = Resize(resize_hw, interpolation=InterpolationMode.NEAREST)(mask)

        if self.crop_size is not None:
            haze = _ensure_min_size(haze, self.crop_size, InterpolationMode.BILINEAR)
            target = _ensure_min_size(target, self.crop_size, InterpolationMode.BILINEAR)
            mask = _ensure_min_size(mask, self.crop_size, InterpolationMode.NEAREST)

        haze, target, mask = self._apply_geometry(haze, target, mask, crop_params, rand_hor, rand_rot)

        haze_raw = self.to_tensor(haze)
        target_tensor = self.to_tensor(target)
        mask_tensor = self.to_tensor(mask)
        unreliable = (mask_tensor > 0.5).float()
        reliability = 1.0 - unreliable
        if self.reliability_floor > 0:
            reliability = reliability.clamp_min(self.reliability_floor)

        return {
            "haze": preprocess_feature(haze),
            "haze_raw": haze_raw,
            "target": target_tensor,
            "mask": unreliable,
            "reliability": reliability,
        }

    def _apply_strong_photometric(self, haze_raw):
        if not self.use_strong_aug:
            return haze_raw

        strong_haze = haze_raw.clone()
        strong_image = FF.to_pil_image(strong_haze)

        if random.random() < 0.8:
            strong_image = self.strong_color_jitter(strong_image)
        if random.random() < 0.3:
            strong_image = self.strong_blur(strong_image)

        strong_haze = self.to_tensor(strong_image)
        if random.random() < 0.5:
            noise_std = random.uniform(0.0, 0.03)
            strong_haze = (strong_haze + torch.randn_like(strong_haze) * noise_std).clamp(0.0, 1.0)
        if random.random() < 0.5:
            gamma = random.uniform(0.85, 1.15)
            strong_haze = FF.adjust_gamma(strong_haze, gamma=gamma).clamp(0.0, 1.0)
        return strong_haze

    def __getitem__(self, index):
        row = self.records[index]
        crop_params = None
        rand_hor = 0
        rand_rot = 0

        if self.crop_size is not None:
            haze_ref, _, _ = self._load_triplet(row)
            haze_ref = _ensure_min_size(haze_ref, self.crop_size, InterpolationMode.BILINEAR)
            if self.train:
                crop_params = FF.get_image_size(haze_ref)
                width, height = crop_params
                top = random.randint(0, max(height - self.crop_size, 0))
                left = random.randint(0, max(width - self.crop_size, 0))
                crop_params = (top, left, self.crop_size, self.crop_size)
                rand_hor = random.randint(0, 1)
                rand_rot = random.randint(0, 3)
            else:
                crop_params = _center_crop_params(haze_ref, self.crop_size)

        sample = self._prepare_triplet(row, crop_params=crop_params, rand_hor=rand_hor, rand_rot=rand_rot)
        weak_haze_raw = sample["haze_raw"]
        strong_haze_raw = self._apply_strong_photometric(weak_haze_raw)
        sample.update(
            {
                "weak_haze": sample["haze"],
                "weak_haze_raw": weak_haze_raw,
                "strong_haze": preprocess_feature(FF.to_pil_image(strong_haze_raw)),
                "strong_haze_raw": strong_haze_raw,
                "split": row["split"],
                "video": row["video"],
                "stem": row["stem"],
                "sample_id": f"{row['video']}/{row['stem']}",
            }
        )

        if self.return_neighbor:
            neighbor_row = self.records[self._choose_neighbor_index(index)]
            neighbor = self._prepare_triplet(neighbor_row, crop_params=crop_params, rand_hor=rand_hor, rand_rot=rand_rot)
            sample.update(
                {
                    "neighbor_haze": neighbor["haze"],
                    "neighbor_haze_raw": neighbor["haze_raw"],
                    "neighbor_target": neighbor["target"],
                    "neighbor_mask": neighbor["mask"],
                    "neighbor_reliability": neighbor["reliability"],
                    "neighbor_sample_id": f"{neighbor_row['video']}/{neighbor_row['stem']}",
                }
            )

        return sample
