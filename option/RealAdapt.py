import argparse
import json
import os

import torch


parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="Automatic detection")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--eval_every", type=int, default=1)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--val_crop_size", type=int, default=256)
parser.add_argument("--start_lr", default=1e-4, type=float, help="start learning rate")
parser.add_argument("--end_lr", default=1e-6, type=float, help="end learning rate")
parser.add_argument("--no_lr_sche", action="store_true", help="disable cosine lr schedule")
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--ema_decay", type=float, default=0.995)
parser.add_argument("--reliability_floor", type=float, default=0.0)
parser.add_argument("--train_mode", type=str, default="legacy", choices=["legacy", "corun_closer"])

parser.add_argument("--w_loss_L1", default=1.0, type=float, help="weight of masked L1 loss")
parser.add_argument("--w_loss_SSIM", default=0.2, type=float, help="weight of blended masked SSIM loss")
parser.add_argument("--w_loss_EMA", default=0.1, type=float, help="weight of EMA consistency loss")
parser.add_argument("--w_loss_tone", default=0.0, type=float, help="weight of low-frequency tone consistency loss")
parser.add_argument("--w_loss_clip", default=0.0, type=float, help="weight of CLIP clear-prior loss")
parser.add_argument("--clip_warmup_epochs", type=int, default=0)
parser.add_argument("--clip_loss_mode", type=str, default="clear_ce", choices=["clear_ce", "directional"])
parser.add_argument("--use_soft_conf", action="store_true", help="enable EMA disagreement based local confidence weighting")
parser.add_argument("--use_mixed_conf", action="store_true", help="enable quality-aware confidence mixing on top of EMA disagreement")
parser.add_argument("--conf_patch_size", type=int, default=32)
parser.add_argument("--conf_tau", type=float, default=0.08)
parser.add_argument("--conf_warmup_epochs", type=int, default=2)
parser.add_argument("--conf_floor", type=float, default=0.15)
parser.add_argument("--conf_quality_weight", type=float, default=0.5)
parser.add_argument("--conf_quality_pool_kernel", type=int, default=11)
parser.add_argument("--conf_quality_tau", type=float, default=0.08)
parser.add_argument("--conf_quality_chroma_weight", type=float, default=1.0)
parser.add_argument("--conf_quality_clip_weight", type=float, default=1.0)
parser.add_argument("--conf_quality_clip_margin", type=float, default=0.03)
parser.add_argument("--use_teacher_pool", action="store_true", help="enable blockwise quality-aware teacher selection between target and EMA")
parser.add_argument("--teacher_pool_block_size", type=int, default=32)
parser.add_argument("--teacher_pool_margin", type=float, default=0.02)
parser.add_argument("--teacher_pool_blend", type=float, default=1.0)
parser.add_argument("--teacher_pool_warmup_epochs", type=int, default=2)
parser.add_argument("--use_neighbor_teacher", action="store_true", help="enable neighbor-aware teacher-bank selection on top of the base target")
parser.add_argument("--neighbor_teacher_blend", type=float, default=0.35)
parser.add_argument("--neighbor_teacher_margin", type=float, default=0.015)
parser.add_argument("--neighbor_teacher_warmup_epochs", type=int, default=2)
parser.add_argument("--neighbor_teacher_pool_kernel", type=int, default=21)
parser.add_argument("--neighbor_teacher_similarity_tau", type=float, default=0.08)
parser.add_argument("--neighbor_teacher_min_similarity", type=float, default=0.55)
parser.add_argument("--tone_pool_kernel", type=int, default=11)
parser.add_argument("--tone_chroma_weight", type=float, default=1.0)
parser.add_argument("--tone_luma_weight", type=float, default=0.3)
parser.add_argument("--use_transmission_lite", action="store_true", help="enable a lightweight transmission head with reconstruction-aware losses")
parser.add_argument("--transmission_hidden_channels", type=int, default=32)
parser.add_argument("--transmission_pool_kernel", type=int, default=31)
parser.add_argument("--transmission_warmup_epochs", type=int, default=1)
parser.add_argument("--w_loss_transmission_proxy", type=float, default=0.0)
parser.add_argument("--w_loss_haze_reconstruction", type=float, default=0.0)
parser.add_argument("--w_loss_transmission_smooth", type=float, default=0.0)
parser.add_argument("--pseudo_warmup_epochs", type=int, default=2)
parser.add_argument("--pseudo_aux_l1_weight", type=float, default=0.3)
parser.add_argument("--pseudo_aux_ssim_weight", type=float, default=0.06)
parser.add_argument("--quality_block_size", type=int, default=32)
parser.add_argument("--quality_fusion", type=str, default="addition", choices=["addition", "multiplication"])
parser.add_argument("--quality_nr_iqa_type", type=str, default="musiq")
parser.add_argument("--quality_nr_iqa_better", type=str, default="higher", choices=["higher", "lower"])
parser.add_argument("--quality_nr_iqa_scale", type=str, default="0,100")
parser.add_argument("--quality_clip_model_type", type=str, default="daclip_ViT-B-32")
parser.add_argument("--quality_clip_tokenizer_type", type=str, default="ViT-B-32")
parser.add_argument("--quality_clip_weight_path", type=str, default=r"D:\ARIS\external\CORUN_weights\daclip\daclip_ViT-B-32.pt")
parser.add_argument("--quality_clip_degradation_ids", type=str, default="1")
parser.add_argument("--memory_bank_update_rule", type=str, default="both_better", choices=["both_better"])
parser.add_argument("--real_l1_weight", type=float, default=1.0)
parser.add_argument("--real_ssim_weight", type=float, default=0.2)
parser.add_argument("--use_quality_bank", action="store_true")
parser.add_argument("--use_strong_aug", action="store_true")

parser.add_argument("--manifest_path", type=str, default=r"D:\ARIS\COA\mine_research\data\mine_manifest.csv")
parser.add_argument("--train_split", type=str, default="train")
parser.add_argument("--val_split", type=str, default="val")
parser.add_argument("--model_arch", type=str, default="student", choices=["student", "student_x", "teacher", "lucidmine"])
parser.add_argument("--init_ckpt", type=str, default="")
parser.add_argument("--strict_load", action="store_true")
parser.add_argument(
    "--adapt_scope",
    type=str,
    default="full",
    choices=["full", "decoder_strict", "decoder_relaxed", "decoder", "decoder_plus_bottleneck", "highlevel_semantic", "lucidmine_modules"],
)

parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
parser.add_argument("--clip_download_root", type=str, default="./clip_model")
parser.add_argument("--clear_prompts", nargs="*", default=["clear mine tunnel", "clear underground roadway", "clear industrial scene"])
parser.add_argument("--hazy_prompts", nargs="*", default=["hazy mine tunnel", "dusty underground roadway", "foggy industrial scene"])

parser.add_argument("--exp_dir", type=str, default="./experiment")
parser.add_argument("--dataset", type=str, default="RealAdapt")
parser.add_argument("--model_name", type=str, default="MineReliability")
parser.add_argument("--saved_model_dir", type=str, default="saved_model")
parser.add_argument("--saved_data_dir", type=str, default="saved_data")

opt = parser.parse_args()
opt.device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
model_dir = os.path.join(dataset_dir, opt.model_name)

if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    opt.saved_model_dir = os.path.join(model_dir, "saved_model")
    opt.saved_data_dir = os.path.join(model_dir, "saved_data")
    os.mkdir(opt.saved_model_dir)
    os.mkdir(opt.saved_data_dir)
else:
    opt.saved_model_dir = os.path.join(model_dir, "saved_model")
    opt.saved_data_dir = os.path.join(model_dir, "saved_data")
    os.makedirs(opt.saved_model_dir, exist_ok=True)
    os.makedirs(opt.saved_data_dir, exist_ok=True)

with open(os.path.join(model_dir, "args.txt"), "w", encoding="utf-8-sig") as f:
    json.dump(opt.__dict__, f, indent=2, ensure_ascii=False)
