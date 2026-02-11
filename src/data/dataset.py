"""
Audio Dataset
=============

Dataset for loading and preprocessing audio files.
"""

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """
    Audio dataset for training.
    
    Loads audio files, applies preprocessing, and extracts random segments.
    """
    
    def __init__(self, config, split='train'):
        """
        Args:
            config: Configuration object
            split: 'train' or 'val'
        """
        self.config = config
        self.split = split
        
        # Get directory
        if split == 'train':
            self.audio_dir = config.data.train_dir
        else:
            self.audio_dir = config.data.val_dir
        
        self.segment_length = config.data.segment_length
        self.sample_rate = config.data.sample_rate
        self.use_rms_norm = config.data.use_rms_norm
        self.rms_threshold = config.data.rms_threshold
        self.peak_target = config.data.peak_target
        
        # Collect audio files
        self.audio_files = []
        audio_extensions = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
        
        for root, _, files in os.walk(self.audio_dir):
            for f in files:
                if f.lower().endswith(audio_extensions):
                    self.audio_files.append(os.path.join(root, f))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {self.audio_dir}")
        
        logger.info(f"{split} dataset: {len(self.audio_files)} files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """Load and preprocess audio."""
        audio_path = self.audio_files[idx]
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Load audio
                waveform, sr = torchaudio.load(audio_path)
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                
                # Extract segment
                if self.split == 'train':
                    # Random segment for training
                    if waveform.shape[1] >= self.segment_length:
                        start = random.randint(0, waveform.shape[1] - self.segment_length)
                        waveform = waveform[:, start:start + self.segment_length]
                    else:
                        # Pad if too short
                        pad_length = self.segment_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
                else:
                    # Fixed segment for validation
                    if waveform.shape[1] >= self.segment_length:
                        waveform = waveform[:, :self.segment_length]
                    else:
                        pad_length = self.segment_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
                
                # Check for silence
                signal_energy = torch.mean(waveform ** 2)
                if signal_energy < 1e-8:
                    idx = random.randint(0, len(self) - 1)
                    retry_count += 1
                    continue
                
                # Normalize
                if self.use_rms_norm:
                    rms = torch.sqrt(torch.mean(waveform ** 2))
                    rms = torch.clamp(rms, min=self.rms_threshold)
                    waveform = waveform / (rms + 1e-8) * 0.1
                else:
                    peak = waveform.abs().max()
                    if peak > 1e-6:
                        waveform = waveform / (peak + 1e-8) * self.peak_target
                
                # Final check
                final_rms = torch.sqrt(torch.mean(waveform ** 2))
                if final_rms < self.rms_threshold:
                    idx = random.randint(0, len(self) - 1)
                    retry_count += 1
                    continue
                
                waveform = torch.clamp(waveform, -1.0, 1.0).float()
                
                return waveform.squeeze(0)
                
            except Exception as e:
                logger.error(f"Error loading {audio_path}: {e}")
                idx = random.randint(0, len(self) - 1)
                retry_count += 1
                continue
        
        raise RuntimeError(f"Failed to load valid sample after {max_retries} retries")
