#!/usr/bin/env python3
"""Training script for the baseline (no-GAN) WavLM vocoder.

Launch with torchrun:
    torchrun --nproc_per_node=4 train.py --config config.yaml

Single GPU:
    python train.py --config config.yaml
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torchaudio
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dataset import AudioDataset, collate_fn
from losses import CombinedLoss
from model import WavLM2Audio


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def setup_logging(rank: int, out_dir: str) -> logging.Logger:
    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        f"[%(asctime)s][rank={rank}][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / f"train_rank{rank}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(rank, model, optimizer, scaler, step, epoch, cfg):
    if rank != 0:
        return
    ckpt_dir = Path(cfg["training"]["output_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    ckpt = {
        "step":                step,
        "epoch":               epoch,
        "model_state_dict":    state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":   scaler.state_dict(),
        "config":              cfg,
    }
    path = ckpt_dir / f"checkpoint_step{step}.pt"
    torch.save(ckpt, path)
    torch.save(ckpt, ckpt_dir / "checkpoint_latest.pt")


def try_resume(logger, cfg, model, optimizer, scaler):
    if not cfg["training"].get("resume", False):
        return 0, 0
    latest = Path(cfg["training"]["output_dir"]) / "checkpoints" / "checkpoint_latest.pt"
    if not latest.exists():
        logger.info("resume=true but no checkpoint_latest.pt found — starting from scratch.")
        return 0, 0
    logger.info("Resuming from %s", latest)
    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("step", 0))
    logger.info("Resumed: epoch=%d  step=%d", start_epoch, global_step)
    return start_epoch, global_step


def save_audio_sample(inp, out, step, out_dir, sr=16000):
    sample_dir = Path(out_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(sample_dir / f"step{step}_input.wav"),
                    inp.detach().float().cpu().unsqueeze(0), sr)
    torchaudio.save(str(sample_dir / f"step{step}_output.wav"),
                    out.detach().float().cpu().unsqueeze(0), sr)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(rank, cfg["training"]["output_dir"])

    logger.info("DDP: rank=%d  local_rank=%d  world_size=%d  device=%s",
                rank, local_rank, world_size, device)

    # Optional perf flags for Ampere/Hopper GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = AudioDataset(
        audio_dir      = cfg["data"]["train_dir"],
        segment_length = int(cfg["data"]["segment_length"]),
        sample_rate    = int(cfg["data"]["sample_rate"]),
        use_rms_norm   = bool(cfg["data"].get("use_rms_norm", False)),
        rms_threshold  = float(cfg["data"]["rms_threshold"]),
        peak_target    = float(cfg["data"]["peak_target"]),
    )
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True) \
              if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size  = int(cfg["training"]["batch_size"]),
        sampler     = sampler,
        shuffle     = (sampler is None),
        num_workers = int(cfg["training"]["num_workers"]),
        pin_memory  = True,
        collate_fn  = collate_fn,
        drop_last   = True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = WavLM2Audio(
        wavlm_model_name   = cfg["model"]["wavlm_model_name"],
        hidden_dim         = int(cfg["model"]["hidden_dim"]),
        num_adapter_layers = int(cfg["model"]["num_adapter_layers"]),
        kernel_size        = int(cfg["model"]["kernel_size"]),
        freeze_wavlm       = bool(cfg["model"]["freeze_wavlm"]),
        dropout            = float(cfg["model"].get("dropout", 0.0)),
    ).to(device)

    criterion = CombinedLoss(
        stft_loss_weight = float(cfg["loss"]["stft_weight"]),
        l1_loss_weight   = float(cfg["loss"]["l1_weight"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(cfg["training"]["lr"]),
        betas        = (0.9, 0.999),
        weight_decay = float(cfg["training"].get("weight_decay", 0.01)),
    )
    scaler       = GradScaler(enabled=bool(cfg["training"]["use_amp"]))
    grad_clip    = float(cfg["training"]["grad_clip"])
    save_every   = int(cfg["training"]["save_interval"])
    num_epochs   = int(cfg["training"]["num_epochs"])

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch, global_step = try_resume(logger, cfg, model, optimizer, scaler)

    # ── SIGTERM handler (Jean-Zay / SLURM safety) ────────────────────────────
    def _sigterm(signum, frame):
        if rank == 0:
            logger.info("SIGTERM received — saving emergency checkpoint")
            save_checkpoint(rank, model, optimizer, scaler,
                            global_step, start_epoch, cfg)
        if dist.is_initialized():
            dist.barrier()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm)

    # ── Wrap with DDP ────────────────────────────────────────────────────────
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        # Keep WavLM in eval mode if frozen
        m = model.module if hasattr(model, "module") else model
        model.train()
        if getattr(m, "freeze_wavlm", False):
            m.wavlm.eval()

        pbar = tqdm(loader, disable=(rank != 0),
                    desc=f"epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            batch = batch.to(device, non_blocking=True)  # (B, T)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=bool(cfg["training"]["use_amp"])):
                pred = model(batch)

            with autocast(enabled=False):
                loss, loss_dict = criterion(pred.float(), batch.float())

            if not torch.isfinite(loss):
                logger.error("Non-finite loss at step %d: %s", global_step, loss_dict)
                global_step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=False
            )

            if not torch.isfinite(grad_norm):
                logger.error("Non-finite grad norm at step %d", global_step)
                scaler.update()
                global_step += 1
                continue

            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if global_step % 100 == 0:
                    logger.info(
                        "step=%d  loss=%.5f  l1=%.5f  stft=%.5f  gnorm=%.3f",
                        global_step,
                        loss_dict["loss"], loss_dict["l1"], loss_dict["stft"],
                        grad_norm,
                    )

                if global_step > 0 and global_step % save_every == 0:
                    save_checkpoint(rank, model, optimizer, scaler,
                                    global_step, epoch, cfg)
                    save_audio_sample(batch[0], pred[0], global_step,
                                      cfg["training"]["output_dir"],
                                      sr=int(cfg["data"]["sample_rate"]))

            global_step += 1

        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
