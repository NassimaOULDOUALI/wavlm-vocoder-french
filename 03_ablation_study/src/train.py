#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
from pathlib import Path
import yaml
import logging
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torchaudio

from model import WavLM2AudioImproved
from discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from losses import CombinedGANLoss, GANLoss


def set_seed(seed: int, rank: int):
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def setup_logging(rank: int, out_dir: str):
    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_rank{rank}.log"

    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        f"[%(asctime)s][rank={rank}][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def save_audio_sample(inp_1d, out_1d, step, output_dir, sr=16000):
    sample_dir = Path(output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    inp = inp_1d.detach().float().cpu().unsqueeze(0)
    out = out_1d.detach().float().cpu().unsqueeze(0)
    torchaudio.save(str(sample_dir / f"step{step}_input.wav"), inp, sample_rate=sr)
    torchaudio.save(str(sample_dir / f"step{step}_output.wav"), out, sample_rate=sr)


def save_checkpoint(rank, generator, mpd, msd, optim_g, optim_d, scaler_g, scaler_d, step, epoch, cfg):
    if rank != 0:
        return
    ckpt_dir = Path(cfg["training"]["output_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    gen_state = generator.module.state_dict() if hasattr(generator, "module") else generator.state_dict()
    mpd_state = mpd.module.state_dict() if hasattr(mpd, "module") else mpd.state_dict()
    msd_state = msd.module.state_dict() if hasattr(msd, "module") else msd.state_dict()

    ckpt = {
        "step": step,
        "epoch": epoch,
        "generator_state_dict": gen_state,
        "mpd_state_dict": mpd_state,
        "msd_state_dict": msd_state,
        "optim_g_state_dict": optim_g.state_dict(),
        "optim_d_state_dict": optim_d.state_dict(),
        "scaler_g_state_dict": scaler_g.state_dict(),
        "scaler_d_state_dict": scaler_d.state_dict(),
        "config": cfg,
    }

    torch.save(ckpt, ckpt_dir / f"checkpoint_step{step}.pt")
    torch.save(ckpt, ckpt_dir / "checkpoint_latest.pt")


def try_resume(logger, cfg, generator, mpd, msd, optim_g, optim_d, scaler_g, scaler_d):
    if not bool(cfg["training"].get("resume", False)):
        return 0, 0

    ckpt_path = Path(cfg["training"]["output_dir"]) / "checkpoints" / "checkpoint_latest.pt"
    if not ckpt_path.exists():
        logger.info("resume=true mais aucun checkpoint trouvé → training from scratch.")
        return 0, 0

    logger.info(f"Resuming from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    generator.load_state_dict(ckpt["generator_state_dict"], strict=False)
    mpd.load_state_dict(ckpt["mpd_state_dict"], strict=False)
    msd.load_state_dict(ckpt["msd_state_dict"], strict=False)
    optim_g.load_state_dict(ckpt["optim_g_state_dict"])
    optim_d.load_state_dict(ckpt["optim_d_state_dict"])

    if "scaler_g_state_dict" in ckpt:
        scaler_g.load_state_dict(ckpt["scaler_g_state_dict"])
    if "scaler_d_state_dict" in ckpt:
        scaler_d.load_state_dict(ckpt["scaler_d_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("step", 0))
    logger.info(f"✅ Resume OK: start_epoch={start_epoch} global_step={global_step}")
    return start_epoch, global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    # overrides
    parser.add_argument("--feature_mode", type=str, default=None, choices=["last", "last_n_mean", "weighted_all", "weighted_last_n"])
    parser.add_argument("--wavlm_last_n", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # apply overrides
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.feature_mode is not None:
        cfg["model"]["feature_mode"] = args.feature_mode
    if args.wavlm_last_n is not None:
        cfg["model"]["wavlm_last_n"] = int(args.wavlm_last_n)
    if args.seed is not None:
        cfg.setdefault("training", {})
        cfg["training"]["seed"] = int(args.seed)

    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(rank, cfg["training"]["output_dir"])
    logger.info(f"DDP init: rank={rank} local_rank={local_rank} world_size={world_size}")

    seed = int(cfg.get("training", {}).get("seed", 1234))
    set_seed(seed, rank)

    logger.info("=" * 80)
    logger.info("🚀 TRAINING GAN - Ablation last-N WavLM layers")
    logger.info("=" * 80)
    logger.info(f"feature_mode={cfg['model'].get('feature_mode')}, wavlm_last_n={cfg['model'].get('wavlm_last_n')}, seed={seed}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ==================== DATASET ====================
    from torch.utils.data import Dataset

    class SimpleAudioDataset(Dataset):
        def __init__(self, audio_dir, segment_length=32000, sample_rate=16000, peak_target=0.95):
            self.audio_dir = Path(audio_dir)
            self.segment_length = int(segment_length)
            self.sample_rate = int(sample_rate)
            self.peak_target = float(peak_target)

            self.files = (
                list(self.audio_dir.glob("**/*.wav"))
                + list(self.audio_dir.glob("**/*.mp3"))
                + list(self.audio_dir.glob("**/*.flac"))
            )
            if len(self.files) == 0:
                raise RuntimeError(f"No audio files found in {self.audio_dir}")

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            audio_path = self.files[idx]
            wav, sr = torchaudio.load(str(audio_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)

            wav = wav.squeeze(0)  # [T]

            if wav.shape[0] >= self.segment_length:
                start = torch.randint(0, wav.shape[0] - self.segment_length + 1, (1,)).item()
                wav = wav[start:start + self.segment_length]
            else:
                wav = F.pad(wav, (0, self.segment_length - wav.shape[0]))

            wav = wav / (wav.abs().max() + 1e-8) * self.peak_target
            return wav.float()

    dataset = SimpleAudioDataset(
        audio_dir=cfg["data"]["train_dir"],
        segment_length=cfg["data"]["segment_length"],
        sample_rate=cfg["data"]["sample_rate"],
        peak_target=cfg["data"].get("peak_target", 0.95),
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )

    # ==================== MODELS ====================
    generator = WavLM2AudioImproved(
        wavlm_model_name=cfg["model"]["wavlm_model_name"],
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_adapter_layers=int(cfg["model"]["num_adapter_layers"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        freeze_wavlm=bool(cfg["model"]["freeze_wavlm"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        feature_mode=str(cfg["model"].get("feature_mode", "last_n_mean")),
        wavlm_last_n=int(cfg["model"].get("wavlm_last_n", 1)),
        use_snake=bool(cfg["model"].get("use_snake", False)),
    ).to(device)

    if getattr(generator, "freeze_wavlm", False):
        generator.wavlm.eval()

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # ==================== LOSSES ====================
    criterion = CombinedGANLoss(
        l1_weight=float(cfg["loss"]["l1_weight"]),
        mel_weight=float(cfg["loss"]["mel_weight"]),
        stft_weight=float(cfg["loss"]["stft_weight"]),
        fm_weight=float(cfg["loss"]["fm_weight"]),
        sample_rate=int(cfg["data"]["sample_rate"]),
    ).to(device)
    gan_loss = GANLoss(loss_type="hinge")

    # ==================== OPTIMIZERS ====================
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        lr=float(cfg["training"]["lr_g"]),
        betas=(0.8, 0.99),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=float(cfg["training"]["lr_d"]),
        betas=(0.8, 0.99),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999)

    use_amp = bool(cfg["training"]["use_amp"])
    scaler_g = GradScaler(enabled=use_amp)
    scaler_d = GradScaler(enabled=use_amp)

    grad_clip = float(cfg["training"]["grad_clip"])
    save_interval = int(cfg["training"]["save_interval"])
    num_epochs = int(cfg["training"]["num_epochs"])

    # ==================== RESUME ====================
    start_epoch, global_step = try_resume(
        logger, cfg, generator, mpd, msd, optim_g, optim_d, scaler_g, scaler_d
    )

    # ==================== DDP WRAP ====================
    if world_size > 1:
        generator = DDP(generator, device_ids=[local_rank], find_unused_parameters=False)
        mpd = DDP(mpd, device_ids=[local_rank], find_unused_parameters=False)
        msd = DDP(msd, device_ids=[local_rank], find_unused_parameters=False)

    # ==================== TRAIN LOOP ====================
    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        generator.train()
        mpd.train()
        msd.train()

        gen_module = generator.module if hasattr(generator, "module") else generator
        if getattr(gen_module, "freeze_wavlm", False):
            gen_module.wavlm.eval()

        it = tqdm(loader, disable=(rank != 0), desc=f"epoch {epoch+1}/{num_epochs}")

        for batch in it:
            real_audio = batch.to(device, non_blocking=True)  # [B, T]

            # ---- Train D ----
            optim_d.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                with torch.no_grad():
                    fake_audio = generator(real_audio)
                mpd_real = mpd(real_audio)
                mpd_fake = mpd(fake_audio.detach())
                msd_real = msd(real_audio)
                msd_fake = msd(fake_audio.detach())

                loss_d = gan_loss.discriminator_loss(mpd_real, mpd_fake) + gan_loss.discriminator_loss(msd_real, msd_fake)

            if torch.isfinite(loss_d):
                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(optim_d)
                torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(msd.parameters()), max_norm=grad_clip)
                scaler_d.step(optim_d)
                scaler_d.update()

            # ---- Train G ----
            optim_g.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                fake_audio = generator(real_audio)
                mpd_real = mpd(real_audio)
                mpd_fake = mpd(fake_audio)
                msd_real = msd(real_audio)
                msd_fake = msd(fake_audio)

            with autocast(enabled=False):
                loss_g, loss_dict = criterion.generator_step(
                    fake_audio.float(),
                    real_audio.float(),
                    mpd_real=mpd_real, mpd_fake=mpd_fake,
                    msd_real=msd_real, msd_fake=msd_fake,
                )

            if torch.isfinite(loss_g):
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optim_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=grad_clip)
                scaler_g.step(optim_g)
                scaler_g.update()

            # ---- Logging & Save ----
            if rank == 0:
                it.set_postfix(loss_g=f"{loss_dict.get('loss_total', 0):.4f}", loss_d=f"{loss_d.item():.4f}")

                if global_step % 50 == 0:
                    logger.info(
                        f"step={global_step} | "
                        f"loss_g={loss_dict.get('loss_total', 0):.4f} | "
                        f"loss_d={loss_d.item():.4f} | "
                        f"l1={loss_dict.get('loss_l1', 0):.4f} | "
                        f"mel={loss_dict.get('loss_mel', 0):.4f}"
                    )

                if global_step > 0 and global_step % save_interval == 0:
                    save_checkpoint(rank, generator, mpd, msd, optim_g, optim_d, scaler_g, scaler_d, global_step, epoch, cfg)
                    save_audio_sample(real_audio[0], fake_audio[0], global_step, cfg["training"]["output_dir"], sr=int(cfg["data"]["sample_rate"]))

            global_step += 1

        scheduler_g.step()
        scheduler_d.step()

        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
