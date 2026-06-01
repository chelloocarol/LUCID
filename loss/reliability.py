import torch
import torch.nn as nn


def _expand_weight(weight, like_tensor):
    if weight is None:
        return torch.ones_like(like_tensor[:, :1, :, :])
    if weight.dim() == 3:
        weight = weight.unsqueeze(1)
    if weight.size(1) != 1:
        weight = weight.mean(dim=1, keepdim=True)
    return weight


def masked_reduce(value_map, weight=None, eps=1e-6):
    weight = _expand_weight(weight, value_map)
    return (value_map * weight).sum() / weight.sum().clamp_min(eps)


def blend_with_reliability(pred, target, reliability):
    reliability = _expand_weight(reliability, pred)
    return pred * reliability + target * (1.0 - reliability)


class MaskedL1Loss(nn.Module):
    def forward(self, pred, target, reliability=None):
        diff_map = (pred - target).abs().mean(dim=1, keepdim=True)
        return masked_reduce(diff_map, reliability)


class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, reliability=None):
        diff_map = (pred - target).pow(2).mean(dim=1, keepdim=True)
        return masked_reduce(diff_map, reliability)


class BlendIgnoreSSIMLoss(nn.Module):
    def __init__(self, ssim_module):
        super(BlendIgnoreSSIMLoss, self).__init__()
        self.ssim_module = ssim_module

    def forward(self, pred, target, reliability=None):
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)
        blended_pred = blend_with_reliability(pred, target, reliability)
        return 1.0 - self.ssim_module(blended_pred, target)


def masked_psnr(pred, target, reliability=None, eps=1e-8):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse_map = (pred - target).pow(2).mean(dim=1, keepdim=True)
    mse = masked_reduce(mse_map, reliability, eps=eps).clamp_min(eps)
    return 10.0 * torch.log10(torch.tensor(1.0, device=pred.device, dtype=pred.dtype) / mse)
