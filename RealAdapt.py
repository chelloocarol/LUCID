import csv
import json
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from data import MineManifestDataset
from loss import BlendIgnoreSSIMLoss, MaskedL1Loss, SSIM, masked_psnr
from metric import ssim
from model import LUCIDMine, Student, Student_x, Teacher
from option.RealAdapt import opt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORUN_ROOT = PROJECT_ROOT / "external" / "CORUN-Colabator"
CORUN_OPEN_CLIP_ROOT = CORUN_ROOT / "corun_colabator" / "archs"
if str(CORUN_OPEN_CLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(CORUN_OPEN_CLIP_ROOT))


def lr_schedule_cosdecay(t, total_steps, init_lr=opt.start_lr, end_lr=opt.end_lr):
    return end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / total_steps))


def set_seed_torch(seed=2026):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_model(model_arch):
    if model_arch == "student":
        return Student()
    if model_arch == "student_x":
        return Student_x()
    if model_arch == "teacher":
        return Teacher()
    if model_arch == "lucidmine":
        return LUCIDMine()
    raise ValueError(f"Unsupported model arch: {model_arch}")


def configure_trainable_parameters(model):
    scope_to_prefixes = {
        "full": None,
        "decoder_strict": (
            "convd4x",
            "dense_2",
            "conv_2",
            "fusion_2",
            "convd2x",
            "dense_1",
            "conv_1",
            "fusion_1",
            "conv_output",
        ),
        "decoder_relaxed": (
            "convd8x",
            "dense_3",
            "conv_3",
            "fusion_3",
            "convd4x",
            "dense_2",
            "conv_2",
            "fusion_2",
            "convd2x",
            "dense_1",
            "conv_1",
            "fusion_1",
            "conv_output",
        ),
        "decoder": (
            "dehaze",
            "convd16x",
            "dense_4",
            "conv_4",
            "fusion_4",
            "convd8x",
            "dense_3",
            "conv_3",
            "fusion_3",
            "convd4x",
            "dense_2",
            "conv_2",
            "fusion_2",
            "convd2x",
            "dense_1",
            "conv_1",
            "fusion_1",
            "conv_output",
        ),
        "decoder_plus_bottleneck": (
            "conv16x",
            "dense3",
            "conv4",
            "fusion4",
            "dehaze",
            "convd16x",
            "dense_4",
            "conv_4",
            "fusion_4",
            "convd8x",
            "dense_3",
            "conv_3",
            "fusion_3",
            "convd4x",
            "dense_2",
            "conv_2",
            "fusion_2",
            "convd2x",
            "dense_1",
            "conv_1",
            "fusion_1",
            "conv_output",
        ),
        "highlevel_semantic": (
            "dense3",
            "conv16x",
            "conv4",
            "fusion4",
            "dehaze",
        ),
        "lucidmine_modules": (
            "mine_prior",
            "visibility_adapter",
            "glare_calibrator",
        ),
    }
    allowed_prefixes = scope_to_prefixes[opt.adapt_scope]
    for _, param in model.named_parameters():
        param.requires_grad = False
    if allowed_prefixes is None:
        for _, param in model.named_parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in allowed_prefixes):
                param.requires_grad = True
    return [name for name, param in model.named_parameters() if param.requires_grad]


def load_checkpoint(model, checkpoint_path, strict=False):
    if not checkpoint_path:
        return {"loaded": False, "missing": [], "unexpected": []}

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    clean_state = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=strict)
    return {"loaded": True, "missing": missing, "unexpected": unexpected}


def update_ema(student_model, ema_model, decay):
    with torch.no_grad():
        student_state = student_model.state_dict()
        ema_state = ema_model.state_dict()
        for key, value in student_state.items():
            if not torch.is_floating_point(value):
                ema_state[key].copy_(value)
                continue
            ema_state[key].mul_(decay).add_(value, alpha=1.0 - decay)


def _normalize_kernel_size(kernel_size):
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _reflect_avg_pool(tensor, kernel_size):
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size <= 1:
        return tensor
    pad = kernel_size // 2
    tensor = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(tensor, kernel_size=kernel_size, stride=1)


def rgb_to_ycbcr(image):
    r = image[:, 0:1, :, :]
    g = image[:, 1:2, :, :]
    b = image[:, 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def compute_soft_confidence(target, ema_pred, patch_size, tau, floor):
    tau = max(float(tau), 1e-6)
    error_map = (target - ema_pred).abs().mean(dim=1, keepdim=True)
    confidence = torch.exp(-_reflect_avg_pool(error_map, patch_size) / tau).clamp(0.0, 1.0)
    if floor > 0:
        confidence = confidence * (1.0 - floor) + floor
    return confidence


def compute_quality_confidence(target):
    target = target.clamp(0, 1)
    chroma = rgb_to_ycbcr(target)[:, 1:, :, :]
    chroma_low = _reflect_avg_pool(chroma, opt.conf_quality_pool_kernel)
    chroma_penalty = (chroma - chroma_low).abs().mean(dim=1, keepdim=True)

    clip_margin = max(float(opt.conf_quality_clip_margin), 1e-4)
    clip_penalty = ((target <= clip_margin) | (target >= 1.0 - clip_margin)).float().mean(dim=1, keepdim=True)
    clip_penalty = _reflect_avg_pool(clip_penalty, opt.conf_quality_pool_kernel)

    quality_penalty = (
        opt.conf_quality_chroma_weight * _reflect_avg_pool(chroma_penalty, opt.conf_quality_pool_kernel)
        + opt.conf_quality_clip_weight * clip_penalty
    )
    tau = max(float(opt.conf_quality_tau), 1e-6)
    return torch.exp(-quality_penalty / tau).clamp(0.0, 1.0)


def compute_neighbor_similarity(haze_raw, neighbor_haze_raw, kernel_size, tau):
    tau = max(float(tau), 1e-6)
    error_map = (haze_raw - neighbor_haze_raw).abs().mean(dim=1, keepdim=True)
    return torch.exp(-_reflect_avg_pool(error_map, kernel_size) / tau).clamp(0.0, 1.0)


def build_training_reliability(base_reliability, target, ema_pred, epoch):
    if not opt.use_soft_conf:
        return base_reliability, None
    if ema_pred is None or epoch <= opt.conf_warmup_epochs:
        return base_reliability, None
    confidence = compute_soft_confidence(
        target=target,
        ema_pred=ema_pred,
        patch_size=opt.conf_patch_size,
        tau=opt.conf_tau,
        floor=0.0,
    )
    if opt.use_mixed_conf:
        quality_conf = compute_quality_confidence(target)
        mix_weight = float(np.clip(opt.conf_quality_weight, 0.0, 1.0))
        confidence = (1.0 - mix_weight) * confidence + mix_weight * quality_conf
    confidence = confidence.clamp(0.0, 1.0)
    if opt.conf_floor > 0:
        confidence = confidence * (1.0 - opt.conf_floor) + opt.conf_floor
    return (base_reliability * confidence).clamp(0.0, 1.0), confidence


def build_teacher_pool_target(target, ema_pred, epoch, haze_raw=None, neighbor_target=None, neighbor_haze_raw=None):
    use_ema_teacher = opt.use_teacher_pool and ema_pred is not None and epoch > opt.teacher_pool_warmup_epochs
    use_neighbor_teacher = (
        opt.use_neighbor_teacher
        and neighbor_target is not None
        and haze_raw is not None
        and neighbor_haze_raw is not None
        and epoch > opt.neighbor_teacher_warmup_epochs
    )
    if not use_ema_teacher and not use_neighbor_teacher:
        return target, None, {"teacher_ratio": 0.0, "teacher_gain": 0.0, "ema_ratio": 0.0, "neighbor_ratio": 0.0, "neighbor_similarity": 0.0}

    block_size = max(int(opt.teacher_pool_block_size), 1)
    target_score = compute_quality_confidence(target)
    if block_size > 1:
        target_score = _reflect_avg_pool(target_score, block_size)

    candidate_images = [target]
    candidate_scores = [target_score]
    candidate_names = ["target"]
    neighbor_similarity = None

    if use_ema_teacher:
        ema_score = compute_quality_confidence(ema_pred)
        if block_size > 1:
            ema_score = _reflect_avg_pool(ema_score, block_size)
        ema_blend = float(np.clip(opt.teacher_pool_blend, 0.0, 1.0))
        ema_candidate = target * (1.0 - ema_blend) + ema_pred * ema_blend
        ema_adjusted_score = target_score * (1.0 - ema_blend) + ema_score * ema_blend - float(opt.teacher_pool_margin)
        candidate_images.append(ema_candidate.clamp(0.0, 1.0))
        candidate_scores.append(ema_adjusted_score)
        candidate_names.append("ema")

    if use_neighbor_teacher:
        neighbor_similarity = compute_neighbor_similarity(
            haze_raw=haze_raw,
            neighbor_haze_raw=neighbor_haze_raw,
            kernel_size=opt.neighbor_teacher_pool_kernel,
            tau=opt.neighbor_teacher_similarity_tau,
        )
        neighbor_quality = compute_quality_confidence(neighbor_target)
        if block_size > 1:
            neighbor_quality = _reflect_avg_pool(neighbor_quality, block_size)
            neighbor_similarity = _reflect_avg_pool(neighbor_similarity, block_size)
        similarity_gate = (neighbor_similarity >= float(opt.neighbor_teacher_min_similarity)).float()
        neighbor_alpha = (neighbor_similarity * float(np.clip(opt.neighbor_teacher_blend, 0.0, 1.0)) * similarity_gate).clamp(0.0, 1.0)
        neighbor_candidate = target * (1.0 - neighbor_alpha) + neighbor_target * neighbor_alpha
        neighbor_adjusted_score = (
            target_score * (1.0 - neighbor_alpha)
            + neighbor_quality * neighbor_alpha
            - float(opt.neighbor_teacher_margin)
        )
        candidate_images.append(neighbor_candidate.clamp(0.0, 1.0))
        candidate_scores.append(neighbor_adjusted_score)
        candidate_names.append("neighbor")

    score_stack = torch.cat(candidate_scores, dim=1)
    winner = score_stack.argmax(dim=1, keepdim=True)

    masks = {}
    mixed_target = torch.zeros_like(target)
    for index, image in enumerate(candidate_images):
        mask = (winner == index).float()
        masks[candidate_names[index]] = mask
        mixed_target = mixed_target + image * mask

    diagnostics = {
        "teacher_ratio": 1.0 - masks["target"].mean().item(),
        "teacher_gain": (score_stack.max(dim=1, keepdim=True).values - score_stack[:, :1, :, :]).clamp_min(0.0).mean().item(),
        "ema_ratio": masks.get("ema", torch.zeros_like(target[:, :1, :, :])).mean().item() if "ema" in masks else 0.0,
        "neighbor_ratio": masks.get("neighbor", torch.zeros_like(target[:, :1, :, :])).mean().item() if "neighbor" in masks else 0.0,
        "neighbor_similarity": neighbor_similarity.mean().item() if neighbor_similarity is not None else 0.0,
    }
    return mixed_target.clamp(0.0, 1.0), masks, diagnostics


def low_frequency_tone_loss(pred, target, reliability, masked_l1_loss):
    pred_low = _reflect_avg_pool(rgb_to_ycbcr(pred.clamp(0, 1)), opt.tone_pool_kernel)
    target_low = _reflect_avg_pool(rgb_to_ycbcr(target.clamp(0, 1)), opt.tone_pool_kernel)
    chroma_loss = masked_l1_loss(pred_low[:, 1:, :, :], target_low[:, 1:, :, :], reliability)
    luma_loss = masked_l1_loss(pred_low[:, :1, :, :], target_low[:, :1, :, :], reliability)
    return opt.tone_chroma_weight * chroma_loss + opt.tone_luma_weight * luma_loss


def estimate_airlight(haze_raw):
    haze_low = _reflect_avg_pool(haze_raw.clamp(0, 1), opt.transmission_pool_kernel)
    return haze_low.flatten(2).amax(dim=2).view(haze_raw.shape[0], haze_raw.shape[1], 1, 1).clamp(0.6, 1.0)


def compute_transmission_proxy(haze_raw, clear_target):
    haze_luma = rgb_to_ycbcr(haze_raw.clamp(0, 1))[:, :1, :, :]
    clear_luma = rgb_to_ycbcr(clear_target.clamp(0, 1))[:, :1, :, :]
    airlight = rgb_to_ycbcr(estimate_airlight(haze_raw))[:, :1, :, :]
    denominator = (airlight - clear_luma).clamp_min(0.05)
    proxy = ((airlight - haze_luma) / denominator).clamp(0.0, 1.0)
    return _reflect_avg_pool(proxy, opt.transmission_pool_kernel)


def reconstruct_haze(clear_pred, transmission, haze_raw):
    airlight = estimate_airlight(haze_raw)
    return (clear_pred * transmission + airlight * (1.0 - transmission)).clamp(0.0, 1.0)


def edge_aware_transmission_smoothness(transmission, haze_raw):
    grad_tx = transmission[:, :, :, 1:] - transmission[:, :, :, :-1]
    grad_ty = transmission[:, :, 1:, :] - transmission[:, :, :-1, :]
    ref_grad_x = haze_raw[:, :, :, 1:] - haze_raw[:, :, :, :-1]
    ref_grad_y = haze_raw[:, :, 1:, :] - haze_raw[:, :, :-1, :]
    weight_x = torch.exp(-10.0 * ref_grad_x.abs().mean(dim=1, keepdim=True))
    weight_y = torch.exp(-10.0 * ref_grad_y.abs().mean(dim=1, keepdim=True))
    return (grad_tx.abs() * weight_x).mean() + (grad_ty.abs() * weight_y).mean()


class TransmissionLiteHead(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=32):
        super().__init__()
        hidden_channels = max(int(hidden_channels), 8)
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, feature, output_size):
        transmission = self.net(feature)
        transmission = F.interpolate(transmission, size=output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(transmission)


class ClipClearPrior(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        from CLIP.clip import encode_text_with_prompt_ensemble, load

        self.device = device
        self.model, _ = load(opt.clip_model_name, device=device, download_root=opt.clip_download_root)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        text_features = encode_text_with_prompt_ensemble(
            self.model,
            [opt.clear_prompts, opt.hazy_prompts],
            device=device,
        )
        self.register_buffer("text_features", text_features.detach())
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def encode_image(self, image):
        image = image.clamp(0, 1)
        image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
        image = (image - self.clip_mean) / self.clip_std
        image_features = self.model.encode_image(image)
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        if image_features.dim() == 3:
            image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, reference=None):
        image_features = self.encode_image(image)
        if opt.clip_loss_mode == "directional":
            if reference is None:
                raise ValueError("Directional CLIP loss requires a reference image.")
            reference_features = self.encode_image(reference)
            delta_image = F.normalize(image_features - reference_features, dim=-1)
            delta_text = F.normalize(self.text_features[0:1] - self.text_features[1:2], dim=-1)
            return (1.0 - (delta_image * delta_text).sum(dim=-1)).mean()
        logits = self.model.logit_scale.exp() * image_features @ self.text_features.t()
        labels = torch.zeros(image.shape[0], dtype=torch.long, device=image.device)
        return F.cross_entropy(logits, labels)


def _parse_float_range(raw_text):
    values = [float(item.strip()) for item in str(raw_text).split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected two comma-separated values, got: {raw_text!r}")
    return values[0], values[1]


def _parse_degradation_ids(raw_text):
    ids = [int(item.strip()) for item in str(raw_text).split(",") if item.strip()]
    if not ids:
        raise ValueError("quality_clip_degradation_ids must contain at least one id.")
    return ids


class OnlineQualityMemoryBank:
    def __init__(self, update_rule="both_better"):
        if update_rule != "both_better":
            raise ValueError(f"Unsupported memory bank update rule: {update_rule}")
        self.update_rule = update_rule
        self.storage = {}

    def select(self, sample_ids, teachers, nr_iqa_scores, clip_scores, masks, device):
        selected_teachers = []
        selected_masks = []
        selected_nr_iqa = []
        selected_clip = []
        update_count = 0
        reuse_count = 0

        for index, sample_id in enumerate(sample_ids):
            current_teacher = teachers[index].detach()
            current_nr_iqa = nr_iqa_scores[index].detach()
            current_clip = clip_scores[index].detach()
            current_mask = masks[index].detach()
            stored = self.storage.get(sample_id)
            should_update = False

            if stored is None:
                should_update = True
            else:
                should_update = bool(stored["nr_iqa"] <= current_nr_iqa.cpu() and stored["clip"] <= current_clip.cpu())

            if should_update:
                self.storage[sample_id] = {
                    "teacher": current_teacher.cpu().clone(),
                    "nr_iqa": current_nr_iqa.cpu().clone(),
                    "clip": current_clip.cpu().clone(),
                    "mask": current_mask.cpu().clone(),
                }
                selected_teachers.append(current_teacher)
                selected_masks.append(current_mask)
                selected_nr_iqa.append(current_nr_iqa)
                selected_clip.append(current_clip)
                update_count += 1
            else:
                selected_teachers.append(stored["teacher"].to(device))
                selected_masks.append(stored["mask"].to(device))
                selected_nr_iqa.append(stored["nr_iqa"].to(device))
                selected_clip.append(stored["clip"].to(device))
                reuse_count += 1

        batch = max(len(sample_ids), 1)
        return {
            "teacher": torch.stack(selected_teachers, dim=0).detach(),
            "mask": torch.stack(selected_masks, dim=0).detach(),
            "nr_iqa": torch.stack(selected_nr_iqa, dim=0).detach(),
            "clip": torch.stack(selected_clip, dim=0).detach(),
            "update_ratio": update_count / batch,
            "reuse_ratio": reuse_count / batch,
        }

    def save(self, path):
        payload = {}
        for sample_id, item in self.storage.items():
            payload[sample_id] = {
                "teacher": item["teacher"].cpu(),
                "nr_iqa": item["nr_iqa"].cpu(),
                "clip": item["clip"].cpu(),
                "mask": item["mask"].cpu(),
            }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path):
        with open(path, "rb") as handle:
            self.storage = pickle.load(handle)


class CorunQualityScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        import open_clip as corun_open_clip
        import pyiqa

        self.device = device
        self.block_size = max(int(opt.quality_block_size), 1)
        self.fusion = opt.quality_fusion
        self.nr_iqa_better = opt.quality_nr_iqa_better
        self.nr_iqa_scale = _parse_float_range(opt.quality_nr_iqa_scale)
        self.degradation_ids = _parse_degradation_ids(opt.quality_clip_degradation_ids)
        self.degradation_count = len(self.degradation_ids)

        self.clip_model, self.clip_preprocess = corun_open_clip.create_model_from_pretrained(
            opt.quality_clip_model_type,
            pretrained=opt.quality_clip_weight_path,
        )
        self.clip_model = self.clip_model.to(device).eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        tokenizer = corun_open_clip.get_tokenizer(opt.quality_clip_tokenizer_type)
        degradations = [
            "motion-blurry",
            "hazy",
            "jpeg-compressed",
            "low-light",
            "noisy",
            "raindrop",
            "rainy",
            "shadowed",
            "snowy",
            "uncompleted",
        ]
        text = tokenizer(degradations).to(device)
        with torch.no_grad(), self._autocast():
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("text_features", text_features.detach())

        self.nr_iqa = pyiqa.create_metric(opt.quality_nr_iqa_type).to(device).eval()
        for param in self.nr_iqa.parameters():
            param.requires_grad = False
        self.block_eval_chunk = 32

    def _autocast(self):
        if isinstance(self.device, str):
            is_cuda = self.device == "cuda"
        else:
            is_cuda = self.device.type == "cuda"
        return torch.cuda.amp.autocast() if is_cuda else nullcontext()

    def block_image(self, image):
        batch, channels, height, width = image.shape
        block_h = self.block_size
        block_w = self.block_size
        num_h = height // block_h
        num_w = width // block_w
        blocked = image.view(batch, channels, num_h, block_h, num_w, block_w)
        blocked = blocked.permute(2, 4, 0, 1, 3, 5).contiguous()
        return blocked.view(num_h * num_w * batch, channels, block_h, block_w)

    def unblock_mask(self, blocked_image, original_shape):
        batch, _, height, width = original_shape
        num_h = height // self.block_size
        num_w = width // self.block_size
        blocked_image = blocked_image.view(num_h, num_w, batch, 1, 1)
        blocked_image = blocked_image.permute(2, 0, 3, 1, 4).contiguous()
        blocked_image = blocked_image.view(batch, 1, num_h, num_w)
        return F.interpolate(blocked_image, (height, width), mode="bilinear", align_corners=False)

    def get_clip_degrad_rate(self, image):
        processed = self.clip_preprocess(image)
        sum_probs = 0
        with torch.no_grad(), self._autocast():
            _, degra_features = self.clip_model.encode_image(processed, control=True)
            degra_features = degra_features / degra_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * degra_features @ self.text_features.T).softmax(dim=-1)
            for degradation in self.degradation_ids:
                sum_probs = sum_probs + text_probs[:, degradation]
        return sum_probs

    def _run_metric_in_chunks(self, image, fn):
        outputs = []
        for chunk in image.split(self.block_eval_chunk, dim=0):
            outputs.append(fn(chunk))
        return torch.cat(outputs, dim=0)

    def score(self, teacher):
        teacher = teacher.detach().clamp(0.0, 1.0)
        original_shape = teacher.shape
        teacher_blocks = self.block_image(teacher)

        with torch.no_grad():
            local_nr_iqa = self._run_metric_in_chunks(teacher_blocks, self.nr_iqa)
            global_nr_iqa = self.nr_iqa(teacher)
            nr_low, nr_high = self.nr_iqa_scale
            local_nr_iqa_mask = self.unblock_mask(local_nr_iqa, original_shape)
            local_nr_iqa_mask = (local_nr_iqa_mask - nr_low) / max(nr_high - nr_low, 1e-6)
            global_nr_iqa = (global_nr_iqa - nr_low) / max(nr_high - nr_low, 1e-6)
            if self.nr_iqa_better != "higher":
                local_nr_iqa_mask = 1.0 - local_nr_iqa_mask
                global_nr_iqa = 1.0 - global_nr_iqa
            local_nr_iqa_mask = local_nr_iqa_mask.clamp(0.0, 1.0)
            global_nr_iqa = global_nr_iqa.clamp(0.0, 1.0)

            local_clip = self._run_metric_in_chunks(teacher_blocks, self.get_clip_degrad_rate)
            global_clip = self.get_clip_degrad_rate(teacher)
            local_clip_mask = self.degradation_count - self.unblock_mask(local_clip, original_shape)
            global_clip = self.degradation_count - global_clip
            local_clip_mask = (local_clip_mask / max(float(self.degradation_count), 1.0)).clamp(0.0, 1.0)
            global_clip = (global_clip / max(float(self.degradation_count), 1.0)).clamp(0.0, 1.0)

            if self.fusion == "multiplication":
                teacher_mask = local_nr_iqa_mask * local_clip_mask
            else:
                teacher_mask = (local_nr_iqa_mask + local_clip_mask) / 2.0

        return {
            "mask": teacher_mask.detach().clamp(0.0, 1.0),
            "global_nr_iqa": global_nr_iqa.detach(),
            "global_clip": global_clip.detach(),
        }


def select_corun_teacher(teacher_current, sample_ids, quality_scorer, memory_bank, device):
    score = quality_scorer.score(teacher_current)
    teacher_mask = score["mask"]
    global_nr_iqa = score["global_nr_iqa"]
    global_clip = score["global_clip"]

    if memory_bank is None:
        return {
            "teacher": teacher_current.detach(),
            "mask": teacher_mask.detach(),
            "nr_iqa": global_nr_iqa.detach(),
            "clip": global_clip.detach(),
            "update_ratio": 0.0,
            "reuse_ratio": 0.0,
        }

    bank_output = memory_bank.select(sample_ids, teacher_current, global_nr_iqa, global_clip, teacher_mask, device)
    return bank_output


def save_checkpoint(model, ema_model, epoch, metrics, is_best=False, aux_states=None):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "metrics": metrics,
        "args": vars(opt),
    }
    if aux_states:
        state["aux_modules"] = aux_states
    torch.save(state, os.path.join(opt.saved_model_dir, "latest.pth"))
    if is_best:
        torch.save(state, os.path.join(opt.saved_model_dir, "best.pth"))


def save_memory_bank(memory_bank, is_best=False):
    if memory_bank is None:
        return
    latest_path = os.path.join(opt.saved_data_dir, "memory_bank_latest.pkl")
    best_path = os.path.join(opt.saved_data_dir, "memory_bank_best.pkl")
    memory_bank.save(latest_path)
    if is_best:
        memory_bank.save(best_path)


def append_history(row):
    history_path = os.path.join(opt.saved_data_dir, "history.jsonl")
    with open(history_path, "a", encoding="utf-8-sig") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_fieldnames():
    return [
        "epoch",
        "train_loss",
        "train_l1",
        "train_ssim",
        "train_ema",
        "train_tone",
        "train_conf",
        "train_clip",
        "train_teacher_ratio",
        "train_teacher_gain",
        "train_ema_teacher_ratio",
        "train_neighbor_teacher_ratio",
        "train_neighbor_similarity",
        "train_transmission_proxy",
        "train_haze_reconstruction",
        "train_transmission_smooth",
        "train_teacher_mask_mean",
        "train_bank_reuse_ratio",
        "train_bank_update_ratio",
        "train_teacher_global_musiq",
        "train_teacher_global_daclip",
        "train_pseudo_aux_loss",
        "val_masked_psnr",
        "val_masked_ssim",
        "val_masked_l1",
        "val_full_psnr",
        "val_full_ssim",
        "val_reliability",
    ]


def init_csv_logger():
    csv_path = os.path.join(opt.saved_data_dir, "metrics.csv")
    if os.path.exists(csv_path):
        return csv_path
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames())
        writer.writeheader()
    return csv_path


def append_csv(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames())
        writer.writerow(row)


def evaluate(model, loader, masked_l1_loss, masked_ssim_loss, device):
    model.eval()
    metrics = {
        "masked_psnr": [],
        "masked_ssim": [],
        "masked_l1": [],
        "full_psnr": [],
        "full_ssim": [],
        "reliability": [],
    }

    with torch.no_grad():
        for batch in loader:
            haze = batch["haze"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            reliability = batch["reliability"].to(device, non_blocking=True)

            pred = model(haze)[0].clamp(0, 1)
            metrics["masked_psnr"].append(masked_psnr(pred, target, reliability).item())
            metrics["masked_ssim"].append((1.0 - masked_ssim_loss(pred, target, reliability)).item())
            metrics["masked_l1"].append(masked_l1_loss(pred, target, reliability).item())
            metrics["full_psnr"].append(masked_psnr(pred, target, None).item())
            metrics["full_ssim"].append(ssim(pred, target).item())
            metrics["reliability"].append(reliability.mean().item())

    return {key: float(np.mean(values)) for key, values in metrics.items()}


def train_legacy_epoch(model, ema_model, batch, epoch, optimizer, device, masked_l1_loss, masked_ssim_loss, clip_prior, transmission_head):
    haze = batch["haze"].to(device, non_blocking=True)
    haze_raw = batch["haze_raw"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    reliability = batch["reliability"].to(device, non_blocking=True)
    neighbor_target = batch["neighbor_target"].to(device, non_blocking=True) if "neighbor_target" in batch else None
    neighbor_haze_raw = batch["neighbor_haze_raw"].to(device, non_blocking=True) if "neighbor_haze_raw" in batch else None

    optimizer.zero_grad(set_to_none=True)
    pred, features = model(haze)
    pred = pred.clamp(0, 1)
    ema_pred = None
    train_reliability = reliability
    confidence = None
    teacher_stats = {
        "teacher_ratio": 0.0,
        "teacher_gain": 0.0,
        "ema_ratio": 0.0,
        "neighbor_ratio": 0.0,
        "neighbor_similarity": 0.0,
    }

    if opt.w_loss_EMA > 0 or opt.use_soft_conf or opt.use_teacher_pool:
        with torch.no_grad():
            ema_pred = ema_model(haze)[0].clamp(0, 1)
    target_for_loss, _, teacher_stats = build_teacher_pool_target(
        target,
        ema_pred,
        epoch,
        haze_raw=haze_raw,
        neighbor_target=neighbor_target,
        neighbor_haze_raw=neighbor_haze_raw,
    )
    train_reliability, confidence = build_training_reliability(reliability, target_for_loss, ema_pred, epoch)

    losses = {}
    if opt.w_loss_L1 > 0:
        losses["l1"] = masked_l1_loss(pred, target_for_loss, train_reliability)
    if opt.w_loss_SSIM > 0:
        losses["ssim"] = masked_ssim_loss(pred, target_for_loss, train_reliability)
    if opt.w_loss_EMA > 0:
        losses["ema"] = F.l1_loss(pred, ema_pred)
    if opt.w_loss_tone > 0:
        losses["tone"] = low_frequency_tone_loss(pred, target_for_loss, train_reliability, masked_l1_loss)
    if clip_prior is not None and epoch > opt.clip_warmup_epochs:
        losses["clip"] = clip_prior(pred, reference=haze if opt.clip_loss_mode == "directional" else None)
    if transmission_head is not None and epoch > opt.transmission_warmup_epochs:
        transmission_pred = transmission_head(features[-1], pred.shape[2:])
        transmission_proxy = compute_transmission_proxy(haze_raw, target_for_loss)
        if opt.w_loss_transmission_proxy > 0:
            losses["transmission_proxy"] = masked_l1_loss(transmission_pred, transmission_proxy, train_reliability)
        if opt.w_loss_haze_reconstruction > 0:
            losses["haze_reconstruction"] = F.l1_loss(reconstruct_haze(pred, transmission_pred, haze_raw), haze_raw)
        if opt.w_loss_transmission_smooth > 0:
            losses["transmission_smooth"] = edge_aware_transmission_smoothness(transmission_pred, haze_raw)

    total_loss = (
        opt.w_loss_L1 * losses.get("l1", torch.tensor(0.0, device=device))
        + opt.w_loss_SSIM * losses.get("ssim", torch.tensor(0.0, device=device))
        + opt.w_loss_EMA * losses.get("ema", torch.tensor(0.0, device=device))
        + opt.w_loss_tone * losses.get("tone", torch.tensor(0.0, device=device))
        + opt.w_loss_clip * losses.get("clip", torch.tensor(0.0, device=device))
        + opt.w_loss_transmission_proxy * losses.get("transmission_proxy", torch.tensor(0.0, device=device))
        + opt.w_loss_haze_reconstruction * losses.get("haze_reconstruction", torch.tensor(0.0, device=device))
        + opt.w_loss_transmission_smooth * losses.get("transmission_smooth", torch.tensor(0.0, device=device))
    )

    return {
        "total_loss": total_loss,
        "losses": losses,
        "confidence_value": confidence.mean().item() if confidence is not None else train_reliability.mean().item(),
        "teacher_ratio": teacher_stats["teacher_ratio"],
        "teacher_gain": teacher_stats["teacher_gain"],
        "ema_teacher_ratio": teacher_stats["ema_ratio"],
        "neighbor_teacher_ratio": teacher_stats["neighbor_ratio"],
        "neighbor_similarity": teacher_stats["neighbor_similarity"],
        "teacher_mask_mean": 0.0,
        "bank_reuse_ratio": 0.0,
        "bank_update_ratio": 0.0,
        "teacher_global_musiq": 0.0,
        "teacher_global_daclip": 0.0,
        "pseudo_aux_loss": 0.0,
    }


def train_corun_epoch(model, ema_model, batch, epoch, optimizer, device, masked_l1_loss, masked_ssim_loss, quality_scorer, quality_bank):
    weak_haze = batch["weak_haze"].to(device, non_blocking=True)
    strong_haze = batch["strong_haze"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    reliability = batch["reliability"].to(device, non_blocking=True)
    sample_ids = list(batch["sample_id"])

    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        teacher_current = ema_model(weak_haze)[0].clamp(0, 1)
        selected = select_corun_teacher(teacher_current, sample_ids, quality_scorer, quality_bank, device)

    selected_teacher = selected["teacher"].detach()
    teacher_mask = selected["mask"].detach()
    train_reliability = (reliability * teacher_mask).clamp(0.0, 1.0)

    pred_strong, _ = model(strong_haze)
    pred_strong = pred_strong.clamp(0, 1)

    losses = {}
    if opt.real_l1_weight > 0:
        losses["l1"] = masked_l1_loss(pred_strong, selected_teacher, train_reliability)
    if opt.real_ssim_weight > 0:
        losses["ssim"] = masked_ssim_loss(pred_strong, selected_teacher, train_reliability)

    pseudo_aux_loss = torch.tensor(0.0, device=device)
    if epoch <= opt.pseudo_warmup_epochs:
        pred_weak, _ = model(weak_haze)
        pred_weak = pred_weak.clamp(0, 1)
        if opt.pseudo_aux_l1_weight > 0:
            pseudo_aux_loss = pseudo_aux_loss + opt.pseudo_aux_l1_weight * masked_l1_loss(pred_weak, target, reliability)
        if opt.pseudo_aux_ssim_weight > 0:
            pseudo_aux_loss = pseudo_aux_loss + opt.pseudo_aux_ssim_weight * masked_ssim_loss(pred_weak, target, reliability)

    total_loss = (
        opt.real_l1_weight * losses.get("l1", torch.tensor(0.0, device=device))
        + opt.real_ssim_weight * losses.get("ssim", torch.tensor(0.0, device=device))
        + pseudo_aux_loss
    )

    return {
        "total_loss": total_loss,
        "losses": losses,
        "confidence_value": train_reliability.mean().item(),
        "teacher_ratio": 0.0,
        "teacher_gain": 0.0,
        "ema_teacher_ratio": 0.0,
        "neighbor_teacher_ratio": 0.0,
        "neighbor_similarity": 0.0,
        "teacher_mask_mean": teacher_mask.mean().item(),
        "bank_reuse_ratio": selected["reuse_ratio"],
        "bank_update_ratio": selected["update_ratio"],
        "teacher_global_musiq": selected["nr_iqa"].mean().item(),
        "teacher_global_daclip": selected["clip"].mean().item(),
        "pseudo_aux_loss": float(pseudo_aux_loss.item()),
    }


def train(model, ema_model, loader_train, loader_val, optimizer, device, aux_modules=None):
    masked_l1_loss = MaskedL1Loss().to(device)
    masked_ssim_loss = BlendIgnoreSSIMLoss(SSIM().to(device)).to(device)
    use_corun_closer = opt.train_mode == "corun_closer"
    clip_prior = ClipClearPrior(device).to(device) if (opt.w_loss_clip > 0 and not use_corun_closer) else None
    transmission_head = aux_modules.get("transmission_head") if aux_modules else None
    quality_scorer = CorunQualityScorer(device).to(device) if use_corun_closer else None
    quality_bank = OnlineQualityMemoryBank(opt.memory_bank_update_rule) if (use_corun_closer and opt.use_quality_bank) else None
    total_steps = max(len(loader_train) * opt.epochs, 1)
    global_step = 0
    best_score = -float("inf")
    csv_path = init_csv_logger()
    train_start = time.time()

    clip_params = [param for param in model.parameters() if param.requires_grad]
    for module in (aux_modules or {}).values():
        clip_params.extend(param for param in module.parameters() if param.requires_grad)

    for epoch in range(1, opt.epochs + 1):
        model.train()
        if transmission_head is not None:
            transmission_head.train()
        epoch_logs = {
            "loss": [],
            "l1": [],
            "ssim": [],
            "ema": [],
            "tone": [],
            "conf": [],
            "clip": [],
            "teacher_ratio": [],
            "teacher_gain": [],
            "ema_teacher_ratio": [],
            "neighbor_teacher_ratio": [],
            "neighbor_similarity": [],
            "transmission_proxy": [],
            "haze_reconstruction": [],
            "transmission_smooth": [],
            "teacher_mask_mean": [],
            "bank_reuse_ratio": [],
            "bank_update_ratio": [],
            "teacher_global_musiq": [],
            "teacher_global_daclip": [],
            "pseudo_aux_loss": [],
        }

        for batch in loader_train:
            global_step += 1
            if not opt.no_lr_sche:
                lr = lr_schedule_cosdecay(global_step, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            else:
                lr = optimizer.param_groups[0]["lr"]

            if use_corun_closer:
                step_output = train_corun_epoch(
                    model,
                    ema_model,
                    batch,
                    epoch,
                    optimizer,
                    device,
                    masked_l1_loss,
                    masked_ssim_loss,
                    quality_scorer,
                    quality_bank,
                )
            else:
                step_output = train_legacy_epoch(
                    model,
                    ema_model,
                    batch,
                    epoch,
                    optimizer,
                    device,
                    masked_l1_loss,
                    masked_ssim_loss,
                    clip_prior,
                    transmission_head,
                )

            total_loss = step_output["total_loss"]
            total_loss.backward()
            if opt.grad_clip > 0 and clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, opt.grad_clip)
            optimizer.step()
            update_ema(model, ema_model, opt.ema_decay)

            losses = step_output["losses"]
            epoch_logs["loss"].append(total_loss.item())
            epoch_logs["l1"].append(losses.get("l1", torch.tensor(0.0, device=device)).item())
            epoch_logs["ssim"].append(losses.get("ssim", torch.tensor(0.0, device=device)).item())
            epoch_logs["ema"].append(losses.get("ema", torch.tensor(0.0, device=device)).item())
            epoch_logs["tone"].append(losses.get("tone", torch.tensor(0.0, device=device)).item())
            epoch_logs["conf"].append(step_output["confidence_value"])
            epoch_logs["clip"].append(losses.get("clip", torch.tensor(0.0, device=device)).item())
            epoch_logs["teacher_ratio"].append(step_output["teacher_ratio"])
            epoch_logs["teacher_gain"].append(step_output["teacher_gain"])
            epoch_logs["ema_teacher_ratio"].append(step_output["ema_teacher_ratio"])
            epoch_logs["neighbor_teacher_ratio"].append(step_output["neighbor_teacher_ratio"])
            epoch_logs["neighbor_similarity"].append(step_output["neighbor_similarity"])
            epoch_logs["transmission_proxy"].append(losses.get("transmission_proxy", torch.tensor(0.0, device=device)).item())
            epoch_logs["haze_reconstruction"].append(losses.get("haze_reconstruction", torch.tensor(0.0, device=device)).item())
            epoch_logs["transmission_smooth"].append(losses.get("transmission_smooth", torch.tensor(0.0, device=device)).item())
            epoch_logs["teacher_mask_mean"].append(step_output["teacher_mask_mean"])
            epoch_logs["bank_reuse_ratio"].append(step_output["bank_reuse_ratio"])
            epoch_logs["bank_update_ratio"].append(step_output["bank_update_ratio"])
            epoch_logs["teacher_global_musiq"].append(step_output["teacher_global_musiq"])
            epoch_logs["teacher_global_daclip"].append(step_output["teacher_global_daclip"])
            epoch_logs["pseudo_aux_loss"].append(step_output["pseudo_aux_loss"])

            print(
                f"\repoch {epoch}/{opt.epochs} step {global_step}/{total_steps} "
                f"loss={total_loss.item():.4f} lr={lr:.7f} "
                f"time_used={(time.time() - train_start) / 60:.1f}m",
                end="",
                flush=True,
            )

        print()

        if epoch % opt.eval_every != 0:
            continue

        metrics = evaluate(ema_model, loader_val, masked_l1_loss, masked_ssim_loss, device)
        score = metrics["masked_psnr"] + 10.0 * metrics["masked_ssim"]
        is_best = score > best_score
        if is_best:
            best_score = score

        aux_states = {name: module.state_dict() for name, module in (aux_modules or {}).items()}
        save_checkpoint(model, ema_model, epoch, metrics, is_best=is_best, aux_states=aux_states)
        save_memory_bank(quality_bank, is_best=is_best)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_logs["loss"])) if epoch_logs["loss"] else 0.0,
            "train_l1": float(np.mean(epoch_logs["l1"])) if epoch_logs["l1"] else 0.0,
            "train_ssim": float(np.mean(epoch_logs["ssim"])) if epoch_logs["ssim"] else 0.0,
            "train_ema": float(np.mean(epoch_logs["ema"])) if epoch_logs["ema"] else 0.0,
            "train_tone": float(np.mean(epoch_logs["tone"])) if epoch_logs["tone"] else 0.0,
            "train_conf": float(np.mean(epoch_logs["conf"])) if epoch_logs["conf"] else 0.0,
            "train_clip": float(np.mean(epoch_logs["clip"])) if epoch_logs["clip"] else 0.0,
            "train_teacher_ratio": float(np.mean(epoch_logs["teacher_ratio"])) if epoch_logs["teacher_ratio"] else 0.0,
            "train_teacher_gain": float(np.mean(epoch_logs["teacher_gain"])) if epoch_logs["teacher_gain"] else 0.0,
            "train_ema_teacher_ratio": float(np.mean(epoch_logs["ema_teacher_ratio"])) if epoch_logs["ema_teacher_ratio"] else 0.0,
            "train_neighbor_teacher_ratio": float(np.mean(epoch_logs["neighbor_teacher_ratio"])) if epoch_logs["neighbor_teacher_ratio"] else 0.0,
            "train_neighbor_similarity": float(np.mean(epoch_logs["neighbor_similarity"])) if epoch_logs["neighbor_similarity"] else 0.0,
            "train_transmission_proxy": float(np.mean(epoch_logs["transmission_proxy"])) if epoch_logs["transmission_proxy"] else 0.0,
            "train_haze_reconstruction": float(np.mean(epoch_logs["haze_reconstruction"])) if epoch_logs["haze_reconstruction"] else 0.0,
            "train_transmission_smooth": float(np.mean(epoch_logs["transmission_smooth"])) if epoch_logs["transmission_smooth"] else 0.0,
            "train_teacher_mask_mean": float(np.mean(epoch_logs["teacher_mask_mean"])) if epoch_logs["teacher_mask_mean"] else 0.0,
            "train_bank_reuse_ratio": float(np.mean(epoch_logs["bank_reuse_ratio"])) if epoch_logs["bank_reuse_ratio"] else 0.0,
            "train_bank_update_ratio": float(np.mean(epoch_logs["bank_update_ratio"])) if epoch_logs["bank_update_ratio"] else 0.0,
            "train_teacher_global_musiq": float(np.mean(epoch_logs["teacher_global_musiq"])) if epoch_logs["teacher_global_musiq"] else 0.0,
            "train_teacher_global_daclip": float(np.mean(epoch_logs["teacher_global_daclip"])) if epoch_logs["teacher_global_daclip"] else 0.0,
            "train_pseudo_aux_loss": float(np.mean(epoch_logs["pseudo_aux_loss"])) if epoch_logs["pseudo_aux_loss"] else 0.0,
            "val_masked_psnr": metrics["masked_psnr"],
            "val_masked_ssim": metrics["masked_ssim"],
            "val_masked_l1": metrics["masked_l1"],
            "val_full_psnr": metrics["full_psnr"],
            "val_full_ssim": metrics["full_ssim"],
            "val_reliability": metrics["reliability"],
        }

        append_csv(csv_path, row)
        append_history({"epoch": epoch, "metrics": row, "best_score": best_score, "is_best": is_best})
        print(
            "validation "
            f"masked_psnr={metrics['masked_psnr']:.3f} "
            f"masked_ssim={metrics['masked_ssim']:.4f} "
            f"full_psnr={metrics['full_psnr']:.3f} "
            f"full_ssim={metrics['full_ssim']:.4f}"
        )


if __name__ == "__main__":
    set_seed_torch(opt.seed)

    use_corun_closer = opt.train_mode == "corun_closer"
    train_set = MineManifestDataset(
        opt.manifest_path,
        split=opt.train_split,
        train=True,
        crop_size=opt.crop_size,
        reliability_floor=opt.reliability_floor,
        return_neighbor=(opt.use_neighbor_teacher and not use_corun_closer),
        use_strong_aug=(opt.use_strong_aug and use_corun_closer),
    )
    val_set = MineManifestDataset(
        opt.manifest_path,
        split=opt.val_split,
        train=False,
        crop_size=opt.val_crop_size if opt.val_crop_size > 0 else None,
        reliability_floor=opt.reliability_floor,
    )

    loader_train = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=opt.device == "cuda",
    )
    loader_val = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, min(opt.num_workers, 2)),
        pin_memory=opt.device == "cuda",
    )

    model = build_model(opt.model_arch).to(opt.device)
    load_info = load_checkpoint(model, opt.init_ckpt, strict=opt.strict_load)
    trainable_names = configure_trainable_parameters(model)
    ema_model = deepcopy(model).to(opt.device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    if opt.device == "cuda":
        cudnn.benchmark = True

    aux_modules = {}
    transmission_enabled = (
        opt.train_mode != "corun_closer"
        and (
            opt.use_transmission_lite
            or opt.w_loss_transmission_proxy > 0
            or opt.w_loss_haze_reconstruction > 0
            or opt.w_loss_transmission_smooth > 0
        )
    )
    aux_param_count = 0
    if transmission_enabled:
        transmission_head = TransmissionLiteHead(hidden_channels=opt.transmission_hidden_channels).to(opt.device)
        aux_modules["transmission_head"] = transmission_head
        aux_param_count = sum(p.numel() for p in transmission_head.parameters() if p.requires_grad)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model_arch={opt.model_arch}")
    print(f"adapt_scope={opt.adapt_scope}")
    print(f"train_mode={opt.train_mode}")
    print(f"trainable_params={trainable_params}")
    print(f"aux_trainable_params={aux_param_count}")
    print(f"trainable_param_tensors={len(trainable_names)}")
    print(f"manifest_path={opt.manifest_path}")
    print(f"load_info={load_info}")
    if use_corun_closer:
        print(f"use_quality_bank={opt.use_quality_bank}")
        print(f"use_strong_aug={opt.use_strong_aug}")
        print(f"quality_clip_weight_path={opt.quality_clip_weight_path}")

    trainable_param_groups = [param for param in model.parameters() if param.requires_grad]
    for module in aux_modules.values():
        trainable_param_groups.extend(param for param in module.parameters() if param.requires_grad)
    optimizer = optim.Adam(
        params=trainable_param_groups,
        lr=opt.start_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    train(model, ema_model, loader_train, loader_val, optimizer, opt.device, aux_modules=aux_modules)
