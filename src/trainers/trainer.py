"""
Trainer for WavLM Vocoder
=========================

Handles distributed training with DDP, AMP, and checkpointing.
"""

import os
import sys
import signal
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.wavlm_vocoder import WavLM2Audio
from ..models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from ..losses.combined import CombinedLoss
from ..losses.gan import DiscriminatorAdversarialLoss
from ..data.dataset import AudioDataset
from ..data.collate import collate_fn
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.audio import save_audio

logger = logging.getLogger(__name__)


class Trainer:
    """
    Distributed trainer for WavLM Vocoder.
    
    Supports:
        - DDP multi-GPU training
        - AMP (Automatic Mixed Precision)
        - GAN training (optional)
        - Checkpointing and resuming
        - TensorBoard logging
    """
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Setup distributed
        self.rank, self.local_rank, self.world_size = self._init_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        logger.info(f"Trainer initialized on rank {self.rank}/{self.world_size}")
        
        # Setup directories
        self.output_dir = Path(config.training.output_dir)
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.sample_dir = self.output_dir / 'samples'
        self.log_dir = self.output_dir / 'logs'
        
        if self.rank == 0:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.sample_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir) if self.rank == 0 else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Setup model, data, optimizers
        self._setup_model()
        self._setup_data()
        self._setup_optimizers()
        self._setup_loss()
        
        # Resume if requested
        if config.training.resume and config.training.checkpoint_path:
            self._resume()
        
        # SIGTERM handler
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        
        logger.info("Trainer setup complete")
    
    def _init_distributed(self):
        """Initialize distributed training."""
        if 'RANK' not in os.environ:
            return 0, 0, 1
        
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    
    def _setup_model(self):
        """Setup generator and discriminators."""
        logger.info("Creating models...")
        
        # Generator
        self.generator = WavLM2Audio(self.config).to(self.device)
        
        # Discriminators (if using GAN)
        if self.config.loss.use_gan:
            self.mpd = MultiPeriodDiscriminator().to(self.device)
            self.msd = MultiScaleDiscriminator().to(self.device)
        
        # Wrap with DDP
        if self.world_size > 1:
            self.generator = DDP(
                self.generator,
                device_ids=[self.local_rank],
                find_unused_parameters=False
            )
            
            if self.config.loss.use_gan:
                self.mpd = DDP(self.mpd, device_ids=[self.local_rank])
                self.msd = DDP(self.msd, device_ids=[self.local_rank])
        
        num_params = self.generator.module.get_num_params() if hasattr(self.generator, 'module') else self.generator.get_num_params()
        logger.info(f"Generator params: {num_params:,}")
    
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        logger.info("Loading datasets...")
        
        # Train dataset
        train_dataset = AudioDataset(self.config, split='train')
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        self.train_sampler = train_sampler
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
    
    def _setup_optimizers(self):
        """Setup optimizers and schedulers."""
        # Generator optimizer
        self.opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.training.lr,
            betas=self.config.training.betas,
            weight_decay=self.config.training.weight_decay
        )
        
        # Discriminator optimizer (if using GAN)
        if self.config.loss.use_gan:
            disc_params = list(self.mpd.parameters()) + list(self.msd.parameters())
            self.opt_d = torch.optim.AdamW(
                disc_params,
                lr=self.config.training.get('lr_discriminator', self.config.training.lr),
                betas=self.config.training.betas,
                weight_decay=self.config.training.weight_decay
            )
        
        # AMP scaler
        self.scaler_g = GradScaler(enabled=self.config.training.use_amp)
        if self.config.loss.use_gan:
            self.scaler_d = GradScaler(enabled=self.config.training.use_amp)
        
        # Performance optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _setup_loss(self):
        """Setup loss functions."""
        self.criterion = CombinedLoss(self.config).to(self.device)
        
        if self.config.loss.use_gan:
            self.disc_loss = DiscriminatorAdversarialLoss().to(self.device)
    
    def _resume(self):
        """Resume from checkpoint."""
        ckpt_path = self.config.training.checkpoint_path
        
        if not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return
        
        logger.info(f"Resuming from {ckpt_path}")
        
        self.current_epoch, self.global_step = load_checkpoint(
            ckpt_path,
            self.generator,
            self.opt_g,
            self.scaler_g
        )
        
        logger.info(f"Resumed: epoch={self.current_epoch}, step={self.global_step}")
    
    def _handle_sigterm(self, signum, frame):
        """Handle SIGTERM (preemption)."""
        if self.rank == 0:
            logger.warning("SIGTERM received, saving checkpoint...")
            save_checkpoint(
                self.generator,
                self.opt_g,
                self.scaler_g,
                self.current_epoch,
                self.global_step,
                self.config,
                self.ckpt_dir,
                filename='checkpoint_sigterm.pt'
            )
        
        if self.world_size > 1:
            dist.barrier()
        
        sys.exit(0)
    
    def train_epoch(self):
        """Train one epoch."""
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)
        
        self.generator.train()
        
        # Keep WavLM frozen
        gen_module = self.generator.module if hasattr(self.generator, 'module') else self.generator
        if hasattr(gen_module, 'wavlm'):
            gen_module.wavlm.eval()
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}',
            disable=(self.rank != 0)
        )
        
        for batch in pbar:
            batch = batch.to(self.device, non_blocking=True)
            
            # ========================================
            # Train Generator
            # ========================================
            
            self.opt_g.zero_grad(set_to_none=True)
            
            with autocast(enabled=self.config.training.use_amp):
                pred = self.generator(batch)
            
            # Compute generator loss
            if self.config.loss.use_gan and self.global_step >= self.config.training.get('warmup_steps', 0):
                # GAN mode
                with autocast(enabled=False):
                    # Discriminator outputs for fake
                    mpd_fake_out, mpd_fake_feat = self.mpd(pred.float())
                    msd_fake_out, msd_fake_feat = self.msd(pred.float())
                    
                    # Real features
                    with torch.no_grad():
                        mpd_real_out, mpd_real_feat = self.mpd(batch.float())
                        msd_real_out, msd_real_feat = self.msd(batch.float())
                    
                    disc_outputs = mpd_fake_out + msd_fake_out
                    disc_features = (
                        mpd_real_feat + msd_real_feat,
                        mpd_fake_feat + msd_fake_feat
                    )
                    
                    loss_g, loss_dict = self.criterion(
                        pred.float(),
                        batch.float(),
                        disc_outputs=disc_outputs,
                        disc_features=disc_features
                    )
            else:
                # No GAN mode
                with autocast(enabled=False):
                    loss_g, loss_dict = self.criterion(pred.float(), batch.float())
            
            # Skip if non-finite
            if not torch.isfinite(loss_g):
                logger.warning(f"Non-finite generator loss at step {self.global_step}")
                self.global_step += 1
                continue
            
            # Backward
            self.scaler_g.scale(loss_g).backward()
            
            # Gradient clipping
            self.scaler_g.unscale_(self.opt_g)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=self.config.training.grad_clip,
                error_if_nonfinite=False
            )
            
            if not torch.isfinite(grad_norm_g):
                logger.warning(f"Non-finite gradients at step {self.global_step}")
                self.scaler_g.update()
                self.global_step += 1
                continue
            
            # Update
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()
            
            # ========================================
            # Train Discriminator (if using GAN)
            # ========================================
            
            if self.config.loss.use_gan and self.global_step >= self.config.training.get('warmup_steps', 0):
                self.opt_d.zero_grad(set_to_none=True)
                
                with autocast(enabled=self.config.training.use_amp):
                    # Real
                    mpd_real_out, _ = self.mpd(batch)
                    msd_real_out, _ = self.msd(batch)
                    
                    # Fake (detached)
                    mpd_fake_out, _ = self.mpd(pred.detach())
                    msd_fake_out, _ = self.msd(pred.detach())
                
                with autocast(enabled=False):
                    loss_mpd = self.disc_loss(mpd_real_out, mpd_fake_out)
                    loss_msd = self.disc_loss(msd_real_out, msd_fake_out)
                    loss_d = loss_mpd + loss_msd
                
                if torch.isfinite(loss_d):
                    self.scaler_d.scale(loss_d).backward()
                    self.scaler_d.unscale_(self.opt_d)
                    
                    grad_norm_d = torch.nn.utils.clip_grad_norm_(
                        list(self.mpd.parameters()) + list(self.msd.parameters()),
                        max_norm=self.config.training.grad_clip,
                        error_if_nonfinite=False
                    )
                    
                    if torch.isfinite(grad_norm_d):
                        self.scaler_d.step(self.opt_d)
                    
                    self.scaler_d.update()
                    
                    loss_dict['disc_loss'] = loss_d.item()
            
            # ========================================
            # Logging
            # ========================================
            
            if self.rank == 0:
                pbar.set_postfix({'loss': f"{loss_g.item():.4f}"})
                
                # TensorBoard
                if self.global_step % self.config.logging.log_interval == 0:
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
                    self.writer.add_scalar('train/grad_norm_g', grad_norm_g, self.global_step)
                
                # Console
                if self.global_step % 100 == 0:
                    log_str = f"Step {self.global_step}: "
                    log_str += " ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                    logger.info(log_str)
                
                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.config.training.save_interval == 0:
                    save_checkpoint(
                        self.generator,
                        self.opt_g,
                        self.scaler_g,
                        self.current_epoch,
                        self.global_step,
                        self.config,
                        self.ckpt_dir,
                        filename=f'checkpoint_step{self.global_step}.pt'
                    )
                    
                    # Save audio sample
                    save_audio(
                        batch[0].cpu(),
                        self.sample_dir / f'step{self.global_step:06d}_input.wav',
                        self.config.data.sample_rate
                    )
                    save_audio(
                        pred[0].detach().cpu(),
                        self.sample_dir / f'step{self.global_step:06d}_output.wav',
                        self.config.data.sample_rate
                    )
            
            self.global_step += 1
        
        # Barrier
        if self.world_size > 1:
            dist.barrier()
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.training.num_epochs}")
        logger.info(f"  Batch size per GPU: {self.config.training.batch_size}")
        logger.info(f"  Total batch size: {self.config.training.batch_size * self.world_size}")
        logger.info(f"  Steps per epoch: {len(self.train_loader)}")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()
            
            # Save epoch checkpoint
            if self.rank == 0:
                save_checkpoint(
                    self.generator,
                    self.opt_g,
                    self.scaler_g,
                    epoch + 1,
                    self.global_step,
                    self.config,
                    self.ckpt_dir,
                    filename=f'checkpoint_epoch{epoch + 1}.pt'
                )
        
        logger.info("Training complete!")
        
        # Cleanup
        if self.world_size > 1:
            dist.destroy_process_group()
        
        if self.writer is not None:
            self.writer.close()
