#!/usr/bin/env python3
"""GAN training script for WavLM vocoder.

Architecture: WavLM-Base+ (frozen) → Adapter → HiFiGenerator
Supervision:  L1 + Mel + multi-STFT + MPD/MSD adversarial + Feature Matching

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

# Local imports — all from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import WavLM2AudioImproved
from discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from losses import CombinedGANLoss, GANLoss

# Dataset is shared with 01_baseline_no_gan
sys.path.insert(0, str(Path(__file__).parent.parent / "01_baseline_no_gan"))
from dataset import AudioDataset, collate_fn


# ---------------------------------------------------------------------------
# Distributed helpers (identical to baseline)
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

def save_checkpoint(rank, generator, mpd, msd,
                    optim_g, optim_d, scaler_g, scaler_d,
                    step, epoch, cfg):
    if rank != 0:
        return
    ckpt_dir = Path(cfg["training"]["output_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _state(m):
        return m.module.state_dict() if hasattr(m, "module") else m.state_dict()

    ckpt = {
        "step":                  step,
        "epoch":                 epoch,
        "generator_state_dict":  _state(generator),
        "mpd_state_dict":        _state(mpd),
        "msd_state_dict":        _state(msd),
        "optim_g_state_dict":    optim_g.state_dict(),
        "optim_d_state_dict":    optim_d.state_dict(),
        "scaler_g_state_dict":   scaler_g.state_dict(),
        "scaler_d_state_dict":   scaler_d.state_dict(),
        "config":                cfg,
    }
    torch.save(ckpt, ckpt_dir / f"checkpoint_step{step}.pt")
    torch.save(ckpt, ckpt_dir / "checkpoint_latest.pt")


def try_resume(logger, cfg, generator, mpd, msd,
               optim_g, optim_d, scaler_g, scaler_d):
    if not cfg["training"].get("resume", False):
        return 0, 0
    latest = Path(cfg["training"]["output_dir"]) / "checkpoints" / "checkpoint_latest.pt"
    if not latest.exists():
        logger.info("resume=true but no checkpoint_latest.pt — starting from scratch.")
        return 0, 0
    logger.info("Resuming from %s", latest)
    ckpt = torch.load(latest, map_location="cpu")
    generator.load_state_dict(ckpt["generator_state_dict"], strict=False)
    mpd.load_state_dict(ckpt["mpd_state_dict"], strict=False)
    msd.load_state_dict(ckpt["msd_state_dict"], strict=False)
    optim_g.load_state_dict(ckpt["optim_g_state_dict"])
    optim_d.load_state_dict(ckpt["optim_d_state_dict"])
    scaler_g.load_state_dict(ckpt["scaler_g_state_dict"])
    scaler_d.load_state_dict(ckpt["scaler_d_state_dict"])
    start_epoch = int(ckpt.get("epoch", 0))
    step        = int(ckpt.get("step", 0))
    logger.info("Resumed: epoch=%d  step=%d", start_epoch, step)
    return start_epoch, step


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
    parser.add_argument("--feature_mode",  default=None,
                        help="Override model.feature_mode in config")
    parser.add_argument("--wavlm_last_n",  type=int, default=None,
                        help="Override model.wavlm_last_n in config")
    parser.add_argument("--output_dir",    default=None,
                        help="Override training.output_dir in config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides (useful for SLURM array jobs)
    if args.feature_mode:
        cfg["model"]["feature_mode"] = args.feature_mode
    if args.wavlm_last_n is not None:
        cfg["model"]["wavlm_last_n"] = args.wavlm_last_n
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir

    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(rank, cfg["training"]["output_dir"])

    logger.info("DDP: rank=%d  local_rank=%d  world_size=%d", rank, local_rank, world_size)
    logger.info("feature_mode=%s  wavlm_last_n=%d",
                cfg["model"].get("feature_mode", "last"),
                cfg["model"].get("wavlm_last_n", 1))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = AudioDataset(
        audio_dir      = cfg["data"]["train_dir"],
        segment_length = int(cfg["data"]["segment_length"]),
        sample_rate    = int(cfg["data"]["sample_rate"]),
        use_rms_norm   = bool(cfg["data"].get("use_rms_norm", False)),
        rms_threshold  = float(cfg["data"].get("rms_threshold", 0.005)),
        peak_target    = float(cfg["data"].get("peak_target", 0.95)),
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

    # ── Models ───────────────────────────────────────────────────────────────
    mc = cfg["model"]
    generator = WavLM2AudioImproved(
        wavlm_model_name   = mc["wavlm_model_name"],
        hidden_dim         = int(mc["hidden_dim"]),
        num_adapter_layers = int(mc.get("num_adapter_layers", 6)),
        kernel_size        = int(mc.get("kernel_size", 7)),
        freeze_wavlm       = bool(mc.get("freeze_wavlm", True)),
        dropout            = float(mc.get("dropout", 0.1)),
        feature_mode       = str(mc.get("feature_mode", "last")),
        wavlm_last_n       = int(mc.get("wavlm_last_n", 1)),
        use_snake          = bool(mc.get("use_snake", False)),
    ).to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    lc = cfg["loss"]
    criterion_g = CombinedGANLoss(
        l1_weight  = float(lc.get("l1_weight", 1.0)),
        mel_weight = float(lc.get("mel_weight", 45.0)),
        stft_weight= float(lc.get("stft_weight", 1.0)),
        fm_weight  = float(lc.get("fm_weight", 2.0)),
    ).to(device)
    criterion_d = GANLoss().to(device)

    tc = cfg["training"]
    optim_g = torch.optim.AdamW(
        generator.parameters(), lr=float(tc["lr_g"]),
        betas=(0.8, 0.99), weight_decay=float(tc.get("weight_decay", 0.01)),
    )
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=float(tc["lr_d"]),
        betas=(0.8, 0.99), weight_decay=float(tc.get("weight_decay", 0.01)),
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999)

    scaler_g = GradScaler(enabled=bool(tc["use_amp"]))
    scaler_d = GradScaler(enabled=bool(tc["use_amp"]))

    grad_clip  = float(tc["grad_clip"])
    save_every = int(tc["save_interval"])
    num_epochs = int(tc["num_epochs"])
    warmup_steps = 10_000  # spectral-only warmup before adversarial loss kicks in

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch, global_step = try_resume(
        logger, cfg, generator, mpd, msd, optim_g, optim_d, scaler_g, scaler_d
    )

    # ── SIGTERM handler ───────────────────────────────────────────────────────
    def _sigterm(signum, frame):
        if rank == 0:
            logger.info("SIGTERM received — saving emergency checkpoint")
            save_checkpoint(rank, generator, mpd, msd, optim_g, optim_d,
                            scaler_g, scaler_d, global_step, start_epoch, cfg)
        if dist.is_initialized():
            dist.barrier()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm)

    # ── DDP wrap ──────────────────────────────────────────────────────────────
    if world_size > 1:
        generator = DDP(generator, device_ids=[local_rank], find_unused_parameters=False)
        mpd       = DDP(mpd,       device_ids=[local_rank], find_unused_parameters=False)
        msd       = DDP(msd,       device_ids=[local_rank], find_unused_parameters=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        m = generator.module if hasattr(generator, "module") else generator
        generator.train()
        mpd.train()
        msd.train()
        if getattr(m, "freeze_wavlm", False):
            m.wavlm.eval()

        pbar = tqdm(loader, disable=(rank != 0),
                    desc=f"epoch {epoch + 1}/{num_epochs}")

        for real in pbar:
            real = real.to(device, non_blocking=True)  # (B, T)
            use_adv = global_step >= warmup_steps

            # ── Discriminator step ────────────────────────────────────────
            optim_d.zero_grad(set_to_none=True)
            with autocast(enabled=bool(tc["use_amp"])):
                with torch.no_grad():
                    fake = generator(real)
                mpd_real, mpd_fake = mpd(real.unsqueeze(1)), mpd(fake.detach().unsqueeze(1))
                msd_real, msd_fake = msd(real.unsqueeze(1)), msd(fake.detach().unsqueeze(1))
                loss_d = criterion_d.discriminator_loss(mpd_real, mpd_fake) + \
                         criterion_d.discriminator_loss(msd_real, msd_fake)

            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(optim_d)
            torch.nn.utils.clip_grad_norm_(
                list(mpd.parameters()) + list(msd.parameters()),
                grad_clip, error_if_nonfinite=False
            )
            scaler_d.step(optim_d)
            scaler_d.update()

            # ── Generator step ────────────────────────────────────────────
            optim_g.zero_grad(set_to_none=True)
            with autocast(enabled=bool(tc["use_amp"])):
                fake = generator(real)
                if use_adv:
                    mpd_real_fm, mpd_fake_fm = mpd(real.unsqueeze(1)), mpd(fake.unsqueeze(1))
                    msd_real_fm, msd_fake_fm = msd(real.unsqueeze(1)), msd(fake.unsqueeze(1))
                    loss_g = criterion_g(
                        fake, real,
                        mpd_real_fm, mpd_fake_fm,
                        msd_real_fm, msd_fake_fm,
                    )
                else:
                    loss_g = criterion_g.spectral_only(fake, real)

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optim_g)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                generator.parameters(), grad_clip, error_if_nonfinite=False
            )
            scaler_g.step(optim_g)
            scaler_g.update()

            if rank == 0:
                pbar.set_postfix(g=f"{loss_g.item():.4f}", d=f"{loss_d.item():.4f}")
                if global_step % 100 == 0:
                    logger.info(
                        "step=%d  loss_g=%.4f  loss_d=%.4f  gnorm=%.3f  adv=%s",
                        global_step, loss_g.item(), loss_d.item(), grad_norm, use_adv,
                    )
                if global_step > 0 and global_step % save_every == 0:
                    save_checkpoint(rank, generator, mpd, msd, optim_g, optim_d,
                                    scaler_g, scaler_d, global_step, epoch, cfg)
                    save_audio_sample(real[0], fake[0], global_step,
                                      tc["output_dir"], sr=int(cfg["data"]["sample_rate"]))

            global_step += 1

        scheduler_g.step()
        scheduler_d.step()
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
