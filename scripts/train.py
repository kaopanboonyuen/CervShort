"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CervShort — Training Script                               ║
║                                                                              ║
║  Trains CervShort (or baseline) on the multi-center cervical cytology       ║
║  dataset. Logs all loss components and saves best checkpoint by val AUC.    ║
║                                                                              ║
║  Usage:                                                                      ║
║    # Full CervShort with ViT-L/16                                            ║
║    python scripts/train.py --backbone vit_l16 --use_cervshort               ║
║                                                                              ║
║    # Baseline without CervShort                                              ║
║    python scripts/train.py --backbone resnet50                              ║
║                                                                              ║
║  Author : Teerapong Panboonyuen (Kao Panboonyuen)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cervshort.model import CervShort
from cervshort.utils.metrics import CervShortEvaluator
from data.dataset import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CervShort")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Train CervShort on cervical cytology dataset"
    )
    # Data
    p.add_argument("--data_root",    type=str, default="data/cervical_cytology")
    p.add_argument("--split_file",   type=str, default=None)
    p.add_argument("--output_dir",   type=str, default="outputs/cervshort")

    # Model
    p.add_argument("--backbone",     type=str, default="vit_l16",
                   choices=["resnet18", "resnet50", "densenet121", "vit_l16"])
    p.add_argument("--use_cervshort",action="store_true",
                   help="Enable CervShort framework (disable for baseline)")
    p.add_argument("--num_classes",  type=int, default=5)
    p.add_argument("--num_domains",  type=int, default=5)
    p.add_argument("--feature_dim",  type=int, default=1024)
    p.add_argument("--proto_dim",    type=int, default=256)

    # Loss weights
    p.add_argument("--lambda_morph", type=float, default=0.5)
    p.add_argument("--lambda_deg",   type=float, default=0.3)
    p.add_argument("--lambda_proto", type=float, default=0.2)
    p.add_argument("--lambda_adv",   type=float, default=0.1)

    # Optimisation
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--num_workers",  type=int,   default=4)

    # Misc
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--amp",          action="store_true", help="Mixed precision")
    p.add_argument("--resume",       type=str,   default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler with linear warm-up + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, device, use_amp: bool):
    model.train()
    total_loss = 0.0
    loss_components = {k: 0.0 for k in
                       ["l_cls", "l_morph", "l_deg", "l_proto", "l_adv"]}
    n_batches = 0

    for batch in loader:
        images     = batch["image"].to(device, non_blocking=True)
        labels     = batch["label"].to(device, non_blocking=True)
        domain_ids = batch["domain_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            out = model(images, labels=labels, domain_ids=domain_ids,
                        return_loss=True)
            loss = out["loss"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            if k in out:
                loss_components[k] += out[k].item()
        n_batches += 1

    avg = {k: v / n_batches for k, v in loss_components.items()}
    avg["loss"] = total_loss / n_batches
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Validation pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    evaluator = CervShortEvaluator(num_classes=model.num_classes)

    for batch in loader:
        images     = batch["image"].to(device, non_blocking=True)
        labels     = batch["label"].to(device, non_blocking=True)
        domain_ids = batch["domain_id"].to(device, non_blocking=True)

        out = model(images, labels=labels, domain_ids=domain_ids,
                    return_loss=False)
        evaluator.update(out["logits"], labels)

    return evaluator.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, "train.log"), "w")

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Loading datasets …")
    loaders = build_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_file=args.split_file,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info(f"Building model: backbone={args.backbone} "
             f"CervShort={'ON' if args.use_cervshort else 'OFF'}")

    if args.use_cervshort:
        model = CervShort(
            backbone=args.backbone,
            num_classes=args.num_classes,
            num_domains=args.num_domains,
            feature_dim=args.feature_dim,
            proto_dim=args.proto_dim,
            lambda_morph=args.lambda_morph,
            lambda_deg=args.lambda_deg,
            lambda_proto=args.lambda_proto,
            lambda_adv=args.lambda_adv,
        )
    else:
        # Baseline: backbone + linear head (no CervShort modules)
        import torchvision.models as tvm
        model = _build_baseline(args)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # Resume checkpoint
    start_epoch = 0
    best_auc    = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        log.info(f"Resumed from epoch {start_epoch}, best AUC={best_auc:.2f}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if "encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = GradScaler(enabled=args.amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    log.info("Starting training …")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_epoch(
            model, loaders["train"], optimizer, scaler, device, args.amp
        )
        val_metrics = validate(model, loaders["val"], device)

        scheduler.step()
        lr_now = optimizer.param_groups[-1]["lr"]
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "lr"   : lr_now,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }
        history.append(entry)
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

        val_auc = val_metrics.get("auc", 0.0)
        log.info(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"val_AUC={val_auc:.2f}% | "
            f"lr={lr_now:.2e} | {elapsed:.0f}s"
        )

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = os.path.join(args.output_dir, "best.pth")
            torch.save({
                "epoch"    : epoch,
                "model"    : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc" : best_auc,
                "args"     : vars(args),
            }, ckpt_path)
            log.info(f"  ✓ New best checkpoint (AUC={best_auc:.2f}%)")

        # Save latest
        torch.save({
            "epoch"    : epoch,
            "model"    : model.state_dict(),
            "best_auc" : best_auc,
        }, os.path.join(args.output_dir, "latest.pth"))

    log.info(f"Training complete. Best val AUC = {best_auc:.2f}%")
    log_file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline builder (no CervShort)
# ─────────────────────────────────────────────────────────────────────────────

def _build_baseline(args):
    import torchvision.models as tvm

    class BaselineModel(nn.Module):
        def __init__(self, backbone, num_classes, feature_dim):
            super().__init__()
            self.num_classes = num_classes
            if backbone == "resnet18":
                enc = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
                in_f = enc.fc.in_features; enc.fc = nn.Linear(in_f, feature_dim)
            elif backbone == "resnet50":
                enc = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
                in_f = enc.fc.in_features; enc.fc = nn.Linear(in_f, feature_dim)
            elif backbone == "densenet121":
                enc = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
                in_f = enc.classifier.in_features
                enc.classifier = nn.Linear(in_f, feature_dim)
            else:  # vit_l16
                try:
                    import timm
                    enc = timm.create_model("vit_large_patch16_224",
                                            pretrained=True, num_classes=feature_dim)
                except ImportError:
                    enc = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.IMAGENET1K_V1)
                    enc.heads.head = nn.Linear(
                        enc.heads.head.in_features, feature_dim)
            self.encoder    = enc
            self.classifier = nn.Linear(feature_dim, num_classes)

        def forward(self, x, labels=None, domain_ids=None, return_loss=True):
            z      = self.encoder(x)
            logits = self.classifier(z)
            out    = {"logits": logits}
            if return_loss and labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
                out["loss"] = loss
                out["l_cls"] = loss.detach()
            return out

    return BaselineModel(args.backbone, args.num_classes, args.feature_dim)


if __name__ == "__main__":
    main()
