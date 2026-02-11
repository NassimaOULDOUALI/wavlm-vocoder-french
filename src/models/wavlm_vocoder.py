"""
WavLM Vocoder Main Model
========================

Complete model combining WavLM encoder (frozen), adapter, and generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import logging

logger = logging.getLogger(__name__)


class WavLM2Audio(nn.Module):
    """
    Complete WavLM to Audio vocoder.
    
    Architecture:
        Audio → WavLM (frozen) → Layer Fusion → Adapter → Generator → Audio
    
    Args:
        config: Configuration dict or OmegaConf object
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Load WavLM
        logger.info(f"Loading WavLM: {config.model.wavlm_model_name}")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.model.wavlm_model_name
        )
        self.wavlm = WavLMModel.from_pretrained(
            config.model.wavlm_model_name
        )
        
        self.wavlm_dim = self.wavlm.config.hidden_size
        
        # Freeze WavLM
        if config.model.freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm.eval()
            logger.info("WavLM frozen")
        
        # Layer fusion
        from .adapter import LayerFusion
        self.layer_fusion = LayerFusion(
            num_layers=config.model.num_layers,
            fusion_type=config.model.fusion_type
        )
        
        # Adapter
        from .adapter import WavLMAdapter
        self.adapter = WavLMAdapter(
            wavlm_dim=self.wavlm_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_adapter_layers,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout
        )
        
        # Generator
        from .generator import HiFiGANGenerator
        self.generator = HiFiGANGenerator(
            hidden_dim=config.model.hidden_dim
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")
    
    def forward(self, audio):
        """
        Args:
            audio: (B, T) waveform
            
        Returns:
            (B, T) reconstructed waveform
        """
        B, T = audio.shape
        
        # Extract WavLM features
        if self.config.model.freeze_wavlm:
            with torch.no_grad():
                outputs = self.wavlm(audio, output_hidden_states=True)
        else:
            outputs = self.wavlm(audio, output_hidden_states=True)
        
        # Get all hidden states
        hidden_states = outputs.hidden_states  # Tuple of (B, T', 768)
        
        # Fuse layers
        fused = self.layer_fusion(hidden_states)  # (B, T', 768)
        
        # Adapt
        adapted = self.adapter(fused)  # (B, hidden_dim, T')
        
        # Generate
        reconstructed = self.generator(adapted)  # (B, 1, T'×320)
        reconstructed = reconstructed.squeeze(1)  # (B, T'×320)
        
        # Match original length
        if reconstructed.shape[1] != T:
            reconstructed = F.interpolate(
                reconstructed.unsqueeze(1),
                size=T,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        return reconstructed
    
    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
