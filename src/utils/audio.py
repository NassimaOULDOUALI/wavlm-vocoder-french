"""
Audio Utilities
===============
"""

import torch
import torchaudio


def load_audio(audio_path, target_sr=16000):
    """
    Load audio file and convert to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        waveform: (T,) mono waveform
        sr: Sample rate
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    return waveform.squeeze(0), sr


def save_audio(waveform, save_path, sample_rate=16000):
    """
    Save audio waveform to file.
    
    Args:
        waveform: (T,) or (1, T) waveform
        save_path: Path to save file
        sample_rate: Sample rate
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    torchaudio.save(save_path, waveform.cpu(), sample_rate)


def process_long_audio(model, audio, chunk_size=320000, overlap=80000, device='cuda'):
    """
    Process long audio with chunking and overlap.
    
    Args:
        model: WavLM2Audio model
        audio: (T,) waveform
        chunk_size: Chunk size in samples
        overlap: Overlap size in samples
        device: Device to run on
        
    Returns:
        (T,) reconstructed waveform
    """
    if len(audio) <= chunk_size:
        # Short audio, process directly
        with torch.no_grad():
            audio_batch = audio.unsqueeze(0).to(device)
            output = model(audio_batch)
            return output.squeeze(0).cpu()
    
    # Process in chunks
    step = chunk_size - overlap
    num_chunks = (len(audio) - overlap) // step + 1
    
    output_chunks = []
    
    for i in range(num_chunks):
        start = i * step
        end = min(start + chunk_size, len(audio))
        
        chunk = audio[start:end]
        
        # Pad if needed
        if len(chunk) < chunk_size:
            pad_size = chunk_size - len(chunk)
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
        
        # Process chunk
        with torch.no_grad():
            chunk_batch = chunk.unsqueeze(0).to(device)
            output_chunk = model(chunk_batch).squeeze(0).cpu()
        
        # Crop overlap
        if i == 0:
            output_chunks.append(output_chunk[:chunk_size - overlap // 2])
        elif i == num_chunks - 1:
            output_chunks.append(output_chunk[overlap // 2:end - start])
        else:
            output_chunks.append(output_chunk[overlap // 2:chunk_size - overlap // 2])
    
    # Concatenate
    output = torch.cat(output_chunks, dim=0)
    
    return output
