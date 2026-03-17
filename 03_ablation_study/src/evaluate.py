#!/usr/bin/env python3
"""
Ablation inference + robust checkpoint selection + SI-SDR time-alignment
+ extended reconstruction metrics (STOI/PESQ/MCD-DTW/F0/VUV + speaker sim + EER)

Improvements vs previous infer.py:
- SI-SDR / SNR / log-mel computed AFTER best lag alignment (xcorr) within +/- max_shift_ms.
- Two checkpoint policies:
  Option 1 (fair): same anchor step across runs (auto=min(max_step_per_run) or user-defined).
  Option 2 (best-per-run): select best checkpoint per run using DEV set, then evaluate on TEST.
- Keeps your evaluation modes:
  trainlike (default): center crop/pad to segment_length, single forward, overlap=0.
  full: chunking + overlap-add.
  center_crop / random_crop: crop to segment_length then single forward.

Added metrics:
- time-domain: rmse, mae (gain-aligned), dur_ref_s, dur_deg_s, dur_ratio
- intelligibility: stoi (requires pystoi)
- perceptual: pesq (requires pesq)
- spectral: mcd_dtw (MFCC-based DTW via torchaudio MFCC; robust and self-contained)
- prosody: f0_rmse_hz, vuv_f1 (pitch extraction via torchaudio, with fallbacks)
- speaker: spk_cosine (ECAPA-TDNN via SpeechBrain), spk_eer (EER computed on ref-vs-est pairs)

Outputs:
  <output_dir>/
    dev/ (if option2)
      refs/, decoded/ (optional), metrics_dev.csv, metrics_dev_summary.csv
      metrics_summary_by_gender.csv, metrics_summary_by_speaker.csv
      selection_dev.csv
    test/ (option2) or eval/ (option1)
      refs/, decoded/<run_name>/audio*.wav, metrics.csv, metrics_summary.csv
      metrics_summary_by_gender.csv, metrics_summary_by_speaker.csv
    chosen_checkpoints.csv
"""
import sys
from pathlib import Path
# Ajouter le dossier parent et le dossier models au path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import copy
import csv
import glob
import inspect
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

from models.models_ablation import WavLM2AudioImproved

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ablation_eval")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# Optional deps (explicitly required only if metric enabled)
try:
    from pystoi import stoi as _stoi_fn  # type: ignore
except Exception:
    _stoi_fn = None

try:
    from pesq import pesq as _pesq_fn  # type: ignore
except Exception:
    _pesq_fn = None

try:
    from speechbrain.inference import EncoderClassifier  # type: ignore
except Exception:
    EncoderClassifier = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "wavlm_model_name": "microsoft/wavlm-base-plus",
        "hidden_dim": 256,
        "num_adapter_layers": 6,
        "kernel_size": 7,
        "freeze_wavlm": True,
        "dropout": 0.1,
        "use_snake": False,
    },
    "data": {
        "sample_rate": 16000,
        "segment_length": 32000,
    }
}

# -------------------------
# Config helpers
# -------------------------
def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def merge_config(default_cfg: Dict[str, Any], ckpt_cfg: Dict[str, Any], user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(default_cfg)
    if isinstance(ckpt_cfg, dict) and ckpt_cfg:
        deep_update(cfg, ckpt_cfg)
    if isinstance(user_cfg, dict) and user_cfg:
        deep_update(cfg, user_cfg)
    return cfg

# -------------------------
# Audio discovery (recursive + prefer **/audio/**)
# -------------------------
def _is_audio_file(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() in AUDIO_EXTENSIONS)

def _has_glob_chars(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])

def _gather_from_dir(root: Path, prefer_audio_subdirs: bool = True) -> List[str]:
    """
    If prefer_audio_subdirs=True:
      1) collect files under **/audio/** with supported extensions
      2) if none, fallback to **/* with supported extensions
    """
    root = root.resolve()
    out: List[Path] = []

    if prefer_audio_subdirs:
        for p in root.rglob("*"):
            if _is_audio_file(p) and ("audio" in [x.name for x in p.parents]):
                out.append(p)
        if out:
            return sorted({str(p) for p in out})

    out = [p for p in root.rglob("*") if _is_audio_file(p)]
    return sorted({str(p) for p in out})

def get_audio_files(input_paths: List[str], prefer_audio_subdirs: bool = True) -> List[str]:
    """
    Accepts:
      - file paths (wav/mp3/...)
      - directories (recursively searched)
      - glob patterns (e.g. /voice/Aznavour_*/audio/*.wav)
    Returns a sorted unique list.
    """
    files: List[str] = []

    for input_path in input_paths:
        s = input_path.strip()
        if not s:
            continue

        if _has_glob_chars(s):
            matches = [Path(x) for x in glob.glob(s, recursive=True)]
            for m in matches:
                if m.is_file() and _is_audio_file(m):
                    files.append(str(m))
                elif m.is_dir():
                    files.extend(_gather_from_dir(m, prefer_audio_subdirs=prefer_audio_subdirs))
            continue

        p = Path(s)
        if p.is_file():
            if _is_audio_file(p):
                files.append(str(p.resolve()))
            else:
                raise ValueError(f"Unsupported extension: {p.suffix} for file {p}")
        elif p.is_dir():
            files.extend(_gather_from_dir(p, prefer_audio_subdirs=prefer_audio_subdirs))
        else:
            raise FileNotFoundError(f"Input not found: {s}")

    return sorted({f for f in files})

# -------------------------
# Speaker/gender helpers
# -------------------------
def infer_gender_from_path(path: str) -> str:
    s = path.lower()
    # conservative tokens
    if re.search(r"(^|[/_\-\s])(female|females|woman|women|f)([/_\-\s]|$)", s):
        return "female"
    if re.search(r"(^|[/_\-\s])(male|males|man|men|m)([/_\-\s]|$)", s):
        return "male"
    # also common folder names
    if "/female/" in s or "/f/" in s:
        return "female"
    if "/male/" in s or "/m/" in s:
        return "male"
    return "unknown"

def speaker_id_from_path(path: str, mode: str, regex: Optional[str] = None) -> str:
    p = Path(path)
    if mode == "none":
        return "unknown"
    if mode == "parent":
        return p.parent.name or "unknown"
    if mode == "grandparent":
        return p.parent.parent.name if p.parent.parent is not None else "unknown"
    if mode == "filename_prefix":
        stem = p.stem
        return (stem.split("_")[0] if "_" in stem else stem) or "unknown"
    if mode == "regex":
        if not regex:
            return "unknown"
        m = re.search(regex, str(p))
        if not m:
            return "unknown"
        if m.groups():
            return m.group(1)
        return m.group(0)
    return "unknown"

# -------------------------
# Checkpoint helpers
# -------------------------
def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        return state_dict
    if any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if isinstance(k, str) and k.startswith("module."):
                new_sd[k.replace("module.", "", 1)] = v
            else:
                new_sd[k] = v
        return new_sd
    return state_dict

def _select_state_dict_from_checkpoint(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "generator_state_dict" in ckpt:
            return ckpt["generator_state_dict"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if any(isinstance(k, str) and ("wavlm" in k or "adapter" in k or "generator" in k) for k in ckpt.keys()):
            return ckpt
        raise ValueError("Unrecognized checkpoint dict format.")
    return ckpt

def parse_step_from_ckpt(p: Path) -> Optional[int]:
    m = re.search(r"checkpoint_step(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else None

def list_step_checkpoints(ckpt_dir: Path) -> List[Path]:
    ckpts = sorted(ckpt_dir.glob("checkpoint_step*.pt"))

    def step_num(pp: Path) -> int:
        s = parse_step_from_ckpt(pp)
        return int(s) if s is not None else -1

    ckpts.sort(key=step_num)
    return ckpts

def pick_ckpt_at_or_before(ckpt_dir: Path, anchor_step: int, require_exact: bool = False) -> Optional[Path]:
    ckpts = list_step_checkpoints(ckpt_dir)
    if not ckpts:
        return None
    if require_exact:
        p = ckpt_dir / f"checkpoint_step{anchor_step}.pt"
        return p if p.exists() else None
    best = None
    best_step = -1
    for p in ckpts:
        s = parse_step_from_ckpt(p)
        if s is None:
            continue
        if s <= anchor_step and s > best_step:
            best = p
            best_step = s
    return best

def max_step_in_dir(ckpt_dir: Path) -> Optional[int]:
    ckpts = list_step_checkpoints(ckpt_dir)
    if not ckpts:
        return None
    s = parse_step_from_ckpt(ckpts[-1])
    return s

def wait_for_condition(desc: str, cond_fn, timeout_s: int, poll_s: int) -> bool:
    t0 = time.time()
    while True:
        ok = bool(cond_fn())
        if ok:
            return True
        elapsed = int(time.time() - t0)
        if elapsed >= timeout_s:
            logger.warning(f"[WAIT] timeout reached ({timeout_s}s) for: {desc}")
            return False
        logger.info(f"[WAIT] {desc} ... elapsed={elapsed}s, next_check_in={poll_s}s")
        time.sleep(poll_s)

# -------------------------
# Audio IO
# -------------------------
def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.dim() != 2:
        raise ValueError(f"Unexpected dims: {wav.shape}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav  # (1, T)

def save_audio(waveform: torch.Tensor, output_path: str, sample_rate: int = 16000):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.clamp(-1.0, 1.0)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_p), waveform.cpu(), sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)

# -------------------------
# Crop/pad like training
# -------------------------
def crop_or_pad_1ch(audio: torch.Tensor, seglen: int, mode: str, rng: random.Random) -> Tuple[torch.Tensor, Dict[str, Any]]:
    assert audio.dim() == 2 and audio.shape[0] == 1
    T = int(audio.shape[1])
    info = {"orig_T": T, "seglen": seglen, "mode": mode, "start": 0, "padded": False}

    if mode == "full":
        return audio, info

    if T >= seglen:
        if mode in ["center_crop", "trainlike"]:
            start = (T - seglen) // 2
        elif mode == "random_crop":
            start = rng.randint(0, T - seglen)
        else:
            start = 0
        info["start"] = int(start)
        return audio[:, start:start + seglen], info

    pad = seglen - T
    info["padded"] = True
    out = F.pad(audio, (0, pad))
    return out, info

def peak_normalize(x: torch.Tensor, target_peak: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    peak = x.abs().max()
    return x / (peak + eps) * target_peak

# -------------------------
# Chunking / resynthesis
# -------------------------
def crop_pad_to_len_1d(x_1d: torch.Tensor, L: int) -> torch.Tensor:
    T = int(x_1d.numel())
    if T >= L:
        return x_1d[:L]
    return F.pad(x_1d, (0, L - T))

@torch.no_grad()
def resynthesize_audio(
    model: torch.nn.Module,
    audio: torch.Tensor,   # (1, T)
    device: str,
    chunk_size: int,
    overlap: int,
    do_peak_norm_per_chunk: bool = True,
) -> torch.Tensor:
    if audio.dim() != 2 or audio.shape[0] != 1:
        raise ValueError(f"Expected shape (1, T), got {audio.shape}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap/chunk_size.")

    audio_1d = audio.squeeze(0)
    total_length = int(audio_1d.numel())

    def prep_chunk(x_raw_1d: torch.Tensor) -> torch.Tensor:
        x = crop_pad_to_len_1d(x_raw_1d, chunk_size)
        if do_peak_norm_per_chunk:
            x = peak_normalize(x)
        return x

    if total_length <= chunk_size:
        x = prep_chunk(audio_1d)
        y = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
        y = y[:total_length]
        if y.numel() < total_length:
            y = F.pad(y, (0, total_length - y.numel()))
        return y.unsqueeze(0)

    step = chunk_size - overlap
    positions = list(range(0, max(1, total_length - chunk_size + 1), step))
    if positions[-1] + chunk_size < total_length:
        positions.append(total_length - chunk_size)

    result = torch.zeros(total_length)
    weights = torch.zeros(total_length)

    for idx, start in enumerate(positions):
        end = min(start + chunk_size, total_length)
        chunk_raw = audio_1d[start:end]
        chunk = prep_chunk(chunk_raw)

        y = model(chunk.unsqueeze(0).to(device)).squeeze(0).cpu()

        chunk_len = end - start
        y = y[:chunk_len]
        if y.numel() < chunk_len:
            y = F.pad(y, (0, chunk_len - y.numel()))

        window = torch.ones(chunk_len)
        if overlap > 0:
            fade_len = min(overlap, chunk_len // 2)
            if fade_len > 0:
                is_first = (idx == 0)
                is_last = (idx == len(positions) - 1)
                if not is_first:
                    window[:fade_len] = torch.linspace(0.0, 1.0, fade_len)
                if not is_last:
                    window[-fade_len:] = torch.linspace(1.0, 0.0, fade_len)

        result[start:end] += y * window
        weights[start:end] += window

    result = result / (weights + 1e-8)
    return result.unsqueeze(0)

# -------------------------
# Core metrics + alignment
# -------------------------
def _match_length_1d(ref: torch.Tensor, est: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    T = min(ref.numel(), est.numel())
    return ref[:T], est[:T]

def _optimal_gain_align(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.dot(est, ref)
    den = torch.dot(est, est) + eps
    a = num / den
    return est * a

def si_sdr_db(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    ref, est = _match_length_1d(ref, est)
    ref = ref - ref.mean()
    est = est - est.mean()
    s_energy = torch.dot(ref, ref) + eps
    proj = torch.dot(est, ref) / s_energy
    s_target = proj * ref
    e_noise = est - s_target
    ratio = (torch.dot(s_target, s_target) + eps) / (torch.dot(e_noise, e_noise) + eps)
    return float(10.0 * torch.log10(ratio))

def snr_db_gain_aligned(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    ref, est = _match_length_1d(ref, est)
    est_a = _optimal_gain_align(ref, est, eps=eps)
    err = ref - est_a
    ratio = (torch.dot(ref, ref) + eps) / (torch.dot(err, err) + eps)
    return float(10.0 * torch.log10(ratio))

def build_mel(sr: int) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=80, f_min=0.0, f_max=sr / 2.0, power=1.0, center=True, pad_mode="reflect"
    )

def build_mfcc(sr: int, n_mfcc: int) -> torchaudio.transforms.MFCC:
    # MFCC via torchaudio; used for a robust MCD-DTW proxy (self-contained)
    return torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            center=True,
            pad_mode="reflect",
            power=2.0,
        ),
    )

def log_mel_l1(ref: torch.Tensor, est: torch.Tensor, mel_fn, eps: float = 1e-5) -> float:
    ref, est = _match_length_1d(ref, est)
    est_a = _optimal_gain_align(ref, est)
    ref_m = mel_fn(ref.unsqueeze(0))
    est_m = mel_fn(est_a.unsqueeze(0))
    ref_l = torch.log(ref_m.clamp_min(eps))
    est_l = torch.log(est_m.clamp_min(eps))
    return float(torch.mean(torch.abs(ref_l - est_l)))

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p

def best_lag_xcorr(ref: torch.Tensor, est: torch.Tensor, max_shift: int) -> int:
    """
    Returns lag (in samples) maximizing cross-correlation in [-max_shift, +max_shift].
    lag > 0 means est is delayed (needs to be shifted left by lag to align to ref).
    Implemented as linear xcorr via FFT: xcorr(ref, est) = conv(flip(ref), est)
    """
    ref, est = _match_length_1d(ref, est)
    ref0 = (ref - ref.mean()).to(torch.float32)
    est0 = (est - est.mean()).to(torch.float32)

    L = int(ref0.numel())
    N = 2 * L - 1
    nfft = _next_pow2(N)

    x_rev = torch.flip(ref0, dims=[0])
    X = torch.fft.rfft(x_rev, n=nfft)
    Y = torch.fft.rfft(est0, n=nfft)
    corr_full = torch.fft.irfft(X * Y, n=nfft)[:N]

    center = L - 1
    lo = max(0, center - max_shift)
    hi = min(N, center + max_shift + 1)

    window = corr_full[lo:hi]
    rel = int(torch.argmax(window).item())
    idx = lo + rel
    lag = idx - center
    return int(lag)

def apply_lag(est: torch.Tensor, lag: int) -> torch.Tensor:
    """
    est: 1D
    If lag > 0: aligned_est[t] = est[t + lag] (shift left), pad tail with zeros.
    If lag < 0: shift right, pad head with zeros.
    """
    T = int(est.numel())
    if lag == 0:
        return est
    if lag > 0:
        core = est[lag:]
        return F.pad(core, (0, lag))[:T]
    else:
        k = -lag
        return F.pad(est, (k, 0))[:T]

# -------------------------
# Extra metrics: DTW / MCD, pitch, STOI/PESQ
# -------------------------
def _as_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)

def _finite(x: float) -> bool:
    return isinstance(x, (float, int)) and math.isfinite(float(x))

def _mcd_const() -> float:
    # 10 / ln(10) * sqrt(2)
    return float((10.0 / math.log(10.0)) * math.sqrt(2.0))

def dtw_path_cost(dist_mat: np.ndarray, window: int) -> Tuple[float, int]:
    """
    Classic DTW with Sakoe-Chiba band window (in frames).
    Returns (total_cost, path_len) for the min-cost path.
    dist_mat: [N, M]
    """
    N, M = dist_mat.shape
    w = max(int(window), abs(N - M))

    dp = np.full((N + 1, M + 1), np.inf, dtype=np.float64)
    bt = np.zeros((N + 1, M + 1), dtype=np.int8)  # 1=up,2=left,3=diag
    dp[0, 0] = 0.0

    for i in range(1, N + 1):
        j_start = max(1, i - w)
        j_end = min(M, i + w)
        for j in range(j_start, j_end + 1):
            cost = float(dist_mat[i - 1, j - 1])
            a = dp[i - 1, j]
            b = dp[i, j - 1]
            c = dp[i - 1, j - 1]
            if c <= a and c <= b:
                dp[i, j] = cost + c
                bt[i, j] = 3
            elif a <= b:
                dp[i, j] = cost + a
                bt[i, j] = 1
            else:
                dp[i, j] = cost + b
                bt[i, j] = 2

    # backtrack length
    i, j = N, M
    if not math.isfinite(dp[i, j]):
        return float("nan"), 0

    path_len = 0
    while i > 0 or j > 0:
        step = bt[i, j]
        if step == 3:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            # unreachable if dp is valid, but guard anyway
            break
        path_len += 1

    return float(dp[N, M]), int(path_len)

def mcd_dtw_mfcc(ref: torch.Tensor, est: torch.Tensor, mfcc_fn, dtw_window: int, n_mfcc: int) -> float:
    """
    MCD-DTW proxy using MFCCs (torchaudio MFCC) instead of true mel-cepstrum.
    Returns mean MCD along DTW path.
    """
    # MFCC expects (batch, time)
    ref_m = mfcc_fn(ref.unsqueeze(0)).squeeze(0)  # [C, T]
    est_m = mfcc_fn(est.unsqueeze(0)).squeeze(0)

    # drop c0 to match common MCD convention
    if ref_m.shape[0] >= 2:
        ref_m = ref_m[1:min(n_mfcc, ref_m.shape[0]), :]
    if est_m.shape[0] >= 2:
        est_m = est_m[1:min(n_mfcc, est_m.shape[0]), :]

    # [frames, dims]
    A = ref_m.transpose(0, 1).contiguous()
    B = est_m.transpose(0, 1).contiguous()

    # safety downsample if extremely long
    max_frames = 2000
    if A.shape[0] > max_frames:
        idx = torch.linspace(0, A.shape[0] - 1, max_frames).long()
        A = A[idx]
    if B.shape[0] > max_frames:
        idx = torch.linspace(0, B.shape[0] - 1, max_frames).long()
        B = B[idx]

    A_np = _as_np(A)
    B_np = _as_np(B)
    # dist matrix: euclidean per-frame
    # memory ok for small frames (you typically use a few seconds)
    dist = np.sqrt(((A_np[:, None, :] - B_np[None, :, :]) ** 2).sum(axis=2) + 1e-12)
    total, plen = dtw_path_cost(dist, window=dtw_window)
    if plen <= 0 or not math.isfinite(total):
        return float("nan")
    mean_l2 = float(total / plen)
    return float(_mcd_const() * mean_l2)

def _extract_pitch_torchaudio(wav_1d: torch.Tensor, sr: int, frame_ms: float, win_ms: float, fmin: float, fmax: float) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns (f0_hz, vuv_mask) over frames, or (None, None) if unavailable.
    Tries torchaudio.functional.compute_kaldi_pitch if present, else detect_pitch_frequency.
    """
    wav = wav_1d.unsqueeze(0)  # (1,T)
    hop_samp = int(round(sr * frame_ms / 1000.0))
    win_samp = int(round(sr * win_ms / 1000.0))

    # 1) compute_kaldi_pitch (if available)
    if hasattr(torchaudio.functional, "compute_kaldi_pitch"):
        try:
            # returns (pitch, nccf) in some versions, or just pitch
            out = torchaudio.functional.compute_kaldi_pitch(
                wav,
                sample_rate=sr,
                frame_length=win_ms,
                frame_shift=frame_ms,
                min_f0=fmin,
                max_f0=fmax,
            )
            if isinstance(out, tuple):
                pitch = out[0]
            else:
                pitch = out
            f0 = pitch.squeeze(0).to(torch.float32).clamp_min(0.0)
            vuv = (f0 > 0.0).to(torch.float32)
            return f0, vuv
        except Exception:
            pass

    # 2) detect_pitch_frequency (YIN-like)
    if hasattr(torchaudio.functional, "detect_pitch_frequency"):
        try:
            f0 = torchaudio.functional.detect_pitch_frequency(
                wav,
                sample_rate=sr,
                frame_time=frame_ms / 1000.0,
                win_length=win_samp,
                freq_low=fmin,
                freq_high=fmax,
            )
            f0 = f0.squeeze(0).to(torch.float32).clamp_min(0.0)
            vuv = (f0 > 0.0).to(torch.float32)
            return f0, vuv
        except Exception:
            pass

    return None, None

def f0_metrics(ref: torch.Tensor, est: torch.Tensor, sr: int, frame_ms: float, win_ms: float, fmin: float, fmax: float) -> Tuple[float, float]:
    """
    Returns (f0_rmse_hz, vuv_f1). NaN if cannot compute.
    """
    f0_r, vuv_r = _extract_pitch_torchaudio(ref, sr, frame_ms, win_ms, fmin, fmax)
    f0_e, vuv_e = _extract_pitch_torchaudio(est, sr, frame_ms, win_ms, fmin, fmax)
    if f0_r is None or f0_e is None or vuv_r is None or vuv_e is None:
        return float("nan"), float("nan")

    L = min(f0_r.numel(), f0_e.numel())
    f0_r = f0_r[:L]
    f0_e = f0_e[:L]
    vuv_r = vuv_r[:L]
    vuv_e = vuv_e[:L]

    # F0 RMSE only on frames voiced in BOTH
    both_voiced = (vuv_r > 0.5) & (vuv_e > 0.5)
    if both_voiced.any():
        diff = f0_r[both_voiced] - f0_e[both_voiced]
        rmse = float(torch.mean(diff ** 2).sqrt().item())
    else:
        rmse = float("nan")

    # VUV F1: voiced as positive, ref as ground truth
    y_true = (vuv_r > 0.5)
    y_pred = (vuv_e > 0.5)
    tp = int((y_true & y_pred).sum().item())
    fp = int((~y_true & y_pred).sum().item())
    fn = int((y_true & ~y_pred).sum().item())
    denom = (2 * tp + fp + fn)
    f1 = float(0.0 if denom == 0 else (2 * tp) / denom)
    return rmse, f1

def stoi_metric(ref: torch.Tensor, est: torch.Tensor, sr: int) -> float:
    if _stoi_fn is None:
        raise RuntimeError("STOI requested but pystoi is not installed. Install with: pip install pystoi")
    r = _as_np(ref)
    e = _as_np(est)
    # pystoi expects 1D arrays
    return float(_stoi_fn(r, e, sr, extended=False))

def pesq_metric(ref: torch.Tensor, est: torch.Tensor, sr: int) -> float:
    if _pesq_fn is None:
        raise RuntimeError("PESQ requested but pesq is not installed. Install with: pip install pesq")
    # PESQ library expects > ~0.25s usually
    if ref.numel() < int(0.25 * sr) or est.numel() < int(0.25 * sr):
        return float("nan")
    r = _as_np(ref)
    e = _as_np(est)
    mode = "wb" if sr >= 16000 else "nb"
    try:
        return float(_pesq_fn(sr, r, e, mode))
    except Exception:
        return float("nan")

# -------------------------
# Speaker metrics (SpeechBrain ECAPA)
# -------------------------
def build_speaker_encoder(source: str, savedir: Optional[str], device: str):
    if EncoderClassifier is None:
        raise RuntimeError("Speaker metrics requested but speechbrain is not installed. Install with: pip install speechbrain")
    try:
        run_opts = {"device": device}
        enc = EncoderClassifier.from_hparams(source=source, savedir=savedir, run_opts=run_opts)
        enc.eval()
        return enc
    except Exception as e:
        raise RuntimeError(
            "Failed to load SpeechBrain speaker encoder. "
            "If running offline, pre-download the model and pass --spk-ecapa-savedir to a local directory.\n"
            f"Underlying error: {repr(e)}"
        )

@torch.no_grad()
def spk_embed(enc, wav_1d: torch.Tensor, device: str) -> torch.Tensor:
    """
    Returns L2-normalized embedding on CPU.
    wav_1d: (T,) float tensor in [-1,1]
    """
    x = wav_1d.unsqueeze(0).to(device)  # [1,T]
    lens = torch.tensor([1.0], device=device)
    emb = enc.encode_batch(x, lens)
    # speechbrain sometimes returns [B,1,D] or [B,D]
    while emb.dim() > 2:
        emb = emb.squeeze(1)
    emb = emb.squeeze(0)
    emb = F.normalize(emb, dim=-1).detach().cpu()
    return emb

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).clamp(-1.0, 1.0).item())

def compute_eer(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """
    EER where FAR == FRR, using linear interpolation on sorted thresholds.
    Scores assumed: higher => more similar (target).
    """
    pos_scores = pos_scores.astype(np.float64)
    neg_scores = neg_scores.astype(np.float64)

    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])

    # sort descending by score
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    P = float((labels == 1).sum())
    N = float((labels == 0).sum())
    if P == 0 or N == 0:
        return float("nan")

    fp = 0.0
    tp = 0.0
    fn = P
    tn = N

    fars = []
    frrs = []

    for y in labels:
        if y == 1:
            tp += 1.0
            fn -= 1.0
        else:
            fp += 1.0
            tn -= 1.0
        far = fp / N
        frr = fn / P
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars, dtype=np.float64)
    frrs = np.array(frrs, dtype=np.float64)
    diff = fars - frrs

    # find crossing
    idx = np.where(diff >= 0)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float((fars[i] + frrs[i]) / 2.0)

    d1 = diff[i - 1]
    d2 = diff[i]
    if d2 == d1:
        return float((fars[i] + frrs[i]) / 2.0)

    # linear interpolation factor
    alpha = d1 / (d1 - d2)
    far = fars[i - 1] + alpha * (fars[i] - fars[i - 1])
    frr = frrs[i - 1] + alpha * (frrs[i] - frrs[i - 1])
    return float((far + frr) / 2.0)

# -------------------------
# Metrics wrapper
# -------------------------
def compute_metrics(
    ref_1d: torch.Tensor,
    est_1d: torch.Tensor,
    sr: int,
    mel_fn,
    max_shift: int,
    want_stoi: bool,
    want_pesq: bool,
    want_mcd: bool,
    want_f0: bool,
    mfcc_fn=None,
    mfcc_n: int = 13,
    dtw_window: int = 80,
    pitch_frame_ms: float = 10.0,
    pitch_win_ms: float = 40.0,
    pitch_fmin: float = 50.0,
    pitch_fmax: float = 600.0,
) -> Dict[str, Any]:
    dur_ref_s = float(ref_1d.numel() / sr)
    dur_deg_s = float(est_1d.numel() / sr)
    dur_ratio = float(dur_deg_s / dur_ref_s) if dur_ref_s > 0 else float("nan")

    ref_1d, est_1d = _match_length_1d(ref_1d, est_1d)

    finite_ok = bool(torch.isfinite(est_1d).all().item())
    est_rms = float(est_1d.pow(2).mean().sqrt().item())
    ref_rms = float(ref_1d.pow(2).mean().sqrt().item())
    silence_ratio = float((est_1d.abs() < 1e-4).float().mean().item())

    # raw (no time alignment)
    si_raw = si_sdr_db(ref_1d, est_1d)
    snr_raw = snr_db_gain_aligned(ref_1d, est_1d)
    mel_raw = log_mel_l1(ref_1d, est_1d, mel_fn)

    # time alignment (xcorr) + gain align for time-domain error
    lag = best_lag_xcorr(ref_1d, est_1d, max_shift=max_shift) if max_shift > 0 else 0
    est_al = apply_lag(est_1d, lag)
    est_al_g = _optimal_gain_align(ref_1d, est_al)

    si_al = si_sdr_db(ref_1d, est_al)
    snr_al = snr_db_gain_aligned(ref_1d, est_al)
    mel_al = log_mel_l1(ref_1d, est_al, mel_fn)

    err = ref_1d - est_al_g
    rmse = float(torch.mean(err ** 2).sqrt().item())
    mae = float(torch.mean(torch.abs(err)).item())

    stoi_v = float("nan")
    pesq_v = float("nan")
    mcd_v = float("nan")
    f0_rmse = float("nan")
    vuv_f1 = float("nan")

    if want_stoi:
        stoi_v = stoi_metric(ref_1d, est_al_g, sr=sr)

    if want_pesq:
        pesq_v = pesq_metric(ref_1d, est_al_g, sr=sr)

    if want_mcd:
        if mfcc_fn is None:
            raise RuntimeError("MCD requested but mfcc_fn is None (unexpected).")
        mcd_v = mcd_dtw_mfcc(ref_1d, est_al_g, mfcc_fn=mfcc_fn, dtw_window=dtw_window, n_mfcc=mfcc_n)

    if want_f0:
        f0_rmse, vuv_f1 = f0_metrics(
            ref_1d, est_al_g, sr=sr,
            frame_ms=pitch_frame_ms, win_ms=pitch_win_ms,
            fmin=pitch_fmin, fmax=pitch_fmax
        )

    return {
        "finite_ok": int(finite_ok),
        "dur_ref_s": dur_ref_s,
        "dur_deg_s": dur_deg_s,
        "dur_ratio": dur_ratio,

        "ref_rms": ref_rms,
        "est_rms": est_rms,
        "silence_ratio": silence_ratio,
        "lag_samples": int(lag),
        "lag_ms": float(1000.0 * lag / sr),

        "rmse": rmse,
        "mae": mae,

        "si_sdr_db_raw": float(si_raw),
        "snr_db_raw": float(snr_raw),
        "log_mel_l1_raw": float(mel_raw),

        "si_sdr_db_aligned": float(si_al),
        "snr_db_aligned": float(snr_al),
        "log_mel_l1_aligned": float(mel_al),

        "stoi": float(stoi_v),
        "pesq": float(pesq_v),
        "mcd_dtw": float(mcd_v),
        "f0_rmse_hz": float(f0_rmse),
        "vuv_f1": float(vuv_f1),
    }

# -------------------------
# Model loading
# -------------------------
def build_model_from_config(config: Dict[str, Any], feature_mode: str, wavlm_last_n: int) -> torch.nn.Module:
    mcfg = config["model"]
    sig = inspect.signature(WavLM2AudioImproved.__init__)
    kwargs = dict(
        wavlm_model_name=mcfg["wavlm_model_name"],
        hidden_dim=int(mcfg.get("hidden_dim", 256)),
        num_adapter_layers=int(mcfg.get("num_adapter_layers", 6)),
        kernel_size=int(mcfg.get("kernel_size", 7)),
        freeze_wavlm=True,
        dropout=0.0,
        use_snake=bool(mcfg.get("use_snake", False)),
    )
    if "feature_mode" in sig.parameters:
        kwargs["feature_mode"] = feature_mode
    if "wavlm_last_n" in sig.parameters:
        kwargs["wavlm_last_n"] = int(wavlm_last_n)
    return WavLM2AudioImproved(**kwargs)

def load_model(ckpt_path: Path, config: Dict[str, Any], device: str, feature_mode: str, wavlm_last_n: int) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = strip_module_prefix(_select_state_dict_from_checkpoint(ckpt))
    model = build_model_from_config(config, feature_mode=feature_mode, wavlm_last_n=wavlm_last_n)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    if hasattr(model, "remove_weight_norm"):
        try:
            model.remove_weight_norm()
        except Exception:
            pass
    return model

# -------------------------
# Runs discovery
# -------------------------
def discover_runs(runs_root: Path) -> List[Tuple[int, Path]]:
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")
    out: List[Tuple[int, Path]] = []
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^N(\d+)_", d.name)
        if m:
            out.append((int(m.group(1)), d))
    out.sort(key=lambda x: x[0])
    return out

# -------------------------
# CSV helpers (robust to missing/nonfinite)
# -------------------------
def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def summarize(
    rows: List[Dict[str, Any]],
    group_keys: Union[str, List[str]],
    metric_keys_mean: List[str],
    extra_keys: List[str],
) -> List[Dict[str, Any]]:
    if isinstance(group_keys, str):
        group_keys = [group_keys]

    agg: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for r in rows:
        g = tuple(str(r.get(k, "NA")) for k in group_keys)
        if g not in agg:
            base = {k: g[i] for i, k in enumerate(group_keys)}
            base["n"] = 0
            for mk in metric_keys_mean:
                base[mk + "_mean"] = 0.0
                base[mk + "_n_valid"] = 0
            for ek in extra_keys:
                base[ek] = r.get(ek, None)
            agg[g] = base

        agg[g]["n"] += 1
        for mk in metric_keys_mean:
            v = r.get(mk, None)
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                agg[g][mk + "_mean"] += fv
                agg[g][mk + "_n_valid"] += 1

    out: List[Dict[str, Any]] = []
    for g, a in agg.items():
        for mk in metric_keys_mean:
            c = int(a.get(mk + "_n_valid", 0))
            a[mk + "_mean"] = float(a[mk + "_mean"] / c) if c > 0 else float("nan")
        out.append(a)
    return out

# -------------------------
# Evaluation core
# -------------------------
def prepare_refs(
    audio_paths: List[str],
    sr: int,
    seglen: int,
    eval_mode: str,
    seed: int,
    num_audios: int,
    reuse_selection_file: Optional[Path],
    out_dir: Path,
    do_peak_norm: bool,
    speaker_id_mode: str,
    speaker_regex: Optional[str],
    infer_gender: bool,
) -> Tuple[List[torch.Tensor], List[str], List[str], List[str]]:
    all_audio = audio_paths
    if len(all_audio) == 0:
        raise RuntimeError("No audio files found.")

    sel_file = reuse_selection_file if reuse_selection_file is not None else (out_dir / "selected_files.txt")
    if sel_file.exists():
        selected = [l.strip() for l in sel_file.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
        logger.info(f"[SELECT] Reusing selection from {sel_file} ({len(selected)} files)")
    else:
        if len(all_audio) < num_audios:
            raise ValueError(f"Requested num_audios={num_audios} but only found {len(all_audio)} files.")
        rng = random.Random(seed)
        selected = rng.sample(all_audio, num_audios)
        sel_file.write_text("\n".join(selected) + "\n")
        logger.info(f"[SELECT] Sampled {len(selected)} files (seed={seed}) -> {sel_file}")

    crop_rng = random.Random(seed)

    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    refs: List[torch.Tensor] = []
    spk_ids: List[str] = []
    genders: List[str] = []
    crop_meta_rows: List[Dict[str, Any]] = []

    force_single_chunk = eval_mode in ["trainlike", "center_crop", "random_crop"]

    for j, ap in enumerate(selected, start=1):
        a = load_audio(ap, target_sr=sr)  # (1,T)
        a_used, info = crop_or_pad_1ch(a, seglen, eval_mode, crop_rng)

        if do_peak_norm and force_single_chunk:
            a_used = peak_normalize(a_used)

        refs.append(a_used)
        save_audio(a_used, str(refs_dir / f"audio{j}_ref.wav"), sample_rate=sr)

        sid = speaker_id_from_path(ap, mode=speaker_id_mode, regex=speaker_regex)
        gen = infer_gender_from_path(ap) if infer_gender else "unknown"
        spk_ids.append(sid)
        genders.append(gen)

        info.update({
            "audio_idx": j,
            "audio_path": ap,
            "speaker_id": sid,
            "gender": gen,
            "duration_s": float(a_used.shape[1] / sr),
            "sr": sr,
        })
        crop_meta_rows.append(info)

        logger.info(
            f"[REF] audio{j}: {ap} orig={a.shape[1]/sr:.2f}s -> used={a_used.shape[1]/sr:.2f}s "
            f"(mode={eval_mode}, start={info['start']}, padded={info['padded']} | speaker_id={sid}, gender={gen})"
        )

    crop_meta_csv = out_dir / "crop_meta.csv"
    write_csv(crop_meta_csv, crop_meta_rows, fieldnames=list(crop_meta_rows[0].keys()) if crop_meta_rows else [])
    logger.info(f"[SAVE] {crop_meta_csv}")

    return refs, selected, spk_ids, genders

def eval_one_run_one_ckpt(
    run_name: str,
    n_layers: int,
    ckpt_path: Path,
    user_cfg: Dict[str, Any],
    device: str,
    feature_mode: str,
    refs: List[torch.Tensor],
    selected_paths: List[str],
    spk_ids: List[str],
    genders: List[str],
    sr: int,
    seglen: int,
    eval_mode: str,
    overlap_default: int,
    out_dir: Path,
    save_decoded: bool,
    max_shift_samples: int,
    # metric flags
    want_stoi: bool,
    want_pesq: bool,
    want_mcd: bool,
    want_f0: bool,
    mfcc_n: int,
    dtw_window: int,
    pitch_frame_ms: float,
    pitch_win_ms: float,
    pitch_fmin: float,
    pitch_fmax: float,
    # speaker
    spk_enc=None,
    spk_device: str = "cpu",
    ref_spk_embs: Optional[List[torch.Tensor]] = None,
    want_eer: bool = False,
    eer_neg_per_pos: int = 5,
    eer_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_cfg = {}
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("config", None), dict):
        ckpt_cfg = ckpt_obj["config"]
    cfg = merge_config(DEFAULT_CONFIG, ckpt_cfg, user_cfg)

    force_single_chunk = eval_mode in ["trainlike", "center_crop", "random_crop"]
    chunk_size = seglen if force_single_chunk else int(cfg["data"].get("segment_length", seglen))
    overlap = 0 if force_single_chunk else int(overlap_default)
    do_peak_norm_per_chunk = True and (not force_single_chunk)

    mel_fn = build_mel(sr)
    mfcc_fn = build_mfcc(sr, n_mfcc=mfcc_n) if want_mcd else None

    model = load_model(ckpt_path=ckpt_path, config=cfg, device=device, feature_mode=feature_mode, wavlm_last_n=n_layers)

    decoded_root = out_dir / "decoded" / run_name
    if save_decoded:
        decoded_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    step_num = parse_step_from_ckpt(ckpt_path)

    # speaker embeddings collection for EER
    est_spk_embs: List[torch.Tensor] = []

    for j, ref in enumerate(refs, start=1):
        out_wav = resynthesize_audio(
            model=model,
            audio=ref,
            device=device,
            chunk_size=chunk_size,
            overlap=overlap,
            do_peak_norm_per_chunk=do_peak_norm_per_chunk,
        )

        out_path = decoded_root / f"audio{j}.wav" if save_decoded else Path("")
        if save_decoded:
            save_audio(out_wav, str(out_path), sample_rate=sr)

        ref_1d = ref.squeeze(0).cpu().to(torch.float32)
        est_1d = out_wav.squeeze(0).cpu().to(torch.float32)

        m = compute_metrics(
            ref_1d, est_1d, sr=sr, mel_fn=mel_fn, max_shift=max_shift_samples,
            want_stoi=want_stoi, want_pesq=want_pesq, want_mcd=want_mcd, want_f0=want_f0,
            mfcc_fn=mfcc_fn, mfcc_n=mfcc_n, dtw_window=dtw_window,
            pitch_frame_ms=pitch_frame_ms, pitch_win_ms=pitch_win_ms,
            pitch_fmin=pitch_fmin, pitch_fmax=pitch_fmax,
        )

        # Speaker cosine per utterance (ref vs est), using aligned+gain-aligned audio
        spk_cos = float("nan")
        if spk_enc is not None:
            # reproduce the same alignment used in compute_metrics for stable embeddings
            # (lag alignment + gain alignment)
            ref_m, est_m = _match_length_1d(ref_1d, est_1d)
            lag = best_lag_xcorr(ref_m, est_m, max_shift=max_shift_samples) if max_shift_samples > 0 else 0
            est_al = apply_lag(est_m, lag)
            est_al_g = _optimal_gain_align(ref_m, est_al)

            if ref_spk_embs is None:
                ref_emb = spk_embed(spk_enc, ref_m, device=spk_device)
            else:
                ref_emb = ref_spk_embs[j - 1]
            est_emb = spk_embed(spk_enc, est_al_g, device=spk_device)
            spk_cos = cosine_sim(ref_emb, est_emb)
            est_spk_embs.append(est_emb)

        rows.append({
            "run_name": run_name,
            "N": int(n_layers),
            "step": int(step_num) if step_num is not None else -1,
            "checkpoint_path": str(ckpt_path),
            "audio_idx": j,
            "audio_path": selected_paths[j - 1],
            "speaker_id": spk_ids[j - 1] if j - 1 < len(spk_ids) else "unknown",
            "gender": genders[j - 1] if j - 1 < len(genders) else "unknown",
            "output_path": str(out_path) if save_decoded else "",
            **m,
            "spk_cosine": float(spk_cos),
        })

    del model
    if "cuda" in device:
        torch.cuda.empty_cache()

    metric_keys = [
        "rmse", "mae",
        "si_sdr_db_raw", "snr_db_raw", "log_mel_l1_raw",
        "si_sdr_db_aligned", "snr_db_aligned", "log_mel_l1_aligned",
        "stoi", "pesq", "mcd_dtw", "f0_rmse_hz", "vuv_f1",
        "spk_cosine",
        "lag_ms", "silence_ratio", "est_rms", "dur_ratio",
    ]
    summary_rows = summarize(
        rows,
        group_keys="run_name",
        metric_keys_mean=metric_keys,
        extra_keys=["N", "step", "checkpoint_path"]
    )
    summary = summary_rows[0] if summary_rows else {}

    if rows:
        finite_ok_rate = sum(int(r["finite_ok"]) for r in rows) / max(1, len(rows))
        summary["finite_ok_rate"] = float(finite_ok_rate)

    # EER per run (needs >=2 distinct speakers and spk_cosine available)
    if want_eer and spk_enc is not None:
        # Build positives: (ref_i, est_i)
        # Negatives: sample mismatched speakers (ref_i vs est_j where speaker differs)
        # Note: we need ref_embs for each i
        if ref_spk_embs is None:
            # compute on the fly (rare)
            ref_embs = [spk_embed(spk_enc, refs[i].squeeze(0).cpu().to(torch.float32), device=spk_device) for i in range(len(refs))]
        else:
            ref_embs = ref_spk_embs

        # map indices by speaker
        uniq_spk = sorted(set(spk_ids))
        uniq_spk = [s for s in uniq_spk if s != "unknown"]
        if len(uniq_spk) < 2:
            summary["spk_eer"] = float("nan")
        else:
            rng = random.Random(eer_seed)
            pos = []
            neg = []
            K = len(refs)
            for i in range(K):
                if i >= len(est_spk_embs):
                    continue
                sid_i = spk_ids[i] if i < len(spk_ids) else "unknown"
                if sid_i == "unknown":
                    continue
                pos.append(cosine_sim(ref_embs[i], est_spk_embs[i]))

                # sample negatives
                candidates = [j for j in range(K) if j != i and j < len(est_spk_embs) and (spk_ids[j] != sid_i)]
                if not candidates:
                    continue
                for _ in range(max(1, int(eer_neg_per_pos))):
                    j = rng.choice(candidates)
                    neg.append(cosine_sim(ref_embs[i], est_spk_embs[j]))

            pos_np = np.array(pos, dtype=np.float64)
            neg_np = np.array(neg, dtype=np.float64)
            summary["spk_eer"] = float(compute_eer(pos_np, neg_np))

    return rows, summary

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runs-root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--feature-mode", type=str, default="last_n_mean")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--eval-mode", type=str, default="trainlike",
                        choices=["trainlike", "full", "center_crop", "random_crop"])
    parser.add_argument("--overlap", type=int, default=1600)

    parser.add_argument("--max-shift-ms", type=float, default=50.0,
                        help="Time alignment window for SI-SDR etc (ms). 0 disables alignment.")
    parser.add_argument("--no-peak-norm", action="store_true")

    parser.add_argument("--ckpt-policy", type=str, required=True, choices=["option1", "option2"],
                        help="option1: fixed anchor step across runs. option2: pick best ckpt per run on dev, eval on test.")
    parser.add_argument("--anchor-step", type=str, default="auto",
                        help="Option1: 'auto' -> min(max_step_per_run). Or an int step like 60000.")
    parser.add_argument("--require-exact-anchor", action="store_true",
                        help="If set, requires checkpoint_step<anchor>.pt to exist; otherwise uses best <= anchor.")

    parser.add_argument("--wait-timeout-s", type=int, default=0,
                        help="If >0: wait up to this many seconds for required checkpoints to appear.")
    parser.add_argument("--wait-poll-s", type=int, default=600)

    parser.add_argument("--dev-audio", type=str, default=None,
                        help="Comma-separated list of dev audio dirs/files/globs (required for option2).")
    parser.add_argument("--test-audio", type=str, required=True,
                        help="Comma-separated list of test audio dirs/files/globs.")
    parser.add_argument("--dev-num-audios", type=int, default=3)
    parser.add_argument("--test-num-audios", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--save-dev-decoded", action="store_true",
                        help="Option2: also save decoded wavs for dev (default: no, to save space).")

    # Speaker/gender parsing
    parser.add_argument("--speaker-id-mode", type=str, default="parent",
                        choices=["parent", "grandparent", "filename_prefix", "regex", "none"],
                        help="How to derive speaker_id from audio_path.")
    parser.add_argument("--speaker-regex", type=str, default=None,
                        help="If speaker-id-mode=regex, first capture group is used as speaker_id.")
    parser.add_argument("--infer-gender", action="store_true",
                        help="Infer gender label from audio path tokens (male/female).")

    # Metrics control
    parser.add_argument("--metrics", type=str, default="full", choices=["basic", "full"],
                        help="basic: keep SI-SDR/SNR/log-mel + rmse/mae/duration. full: adds STOI/PESQ/MCD/F0 + speaker.")
    parser.add_argument("--no-stoi", action="store_true")
    parser.add_argument("--no-pesq", action="store_true")
    parser.add_argument("--no-mcd", action="store_true")
    parser.add_argument("--no-f0", action="store_true")
    parser.add_argument("--no-spk", action="store_true")
    parser.add_argument("--no-eer", action="store_true")

    # MCD options
    parser.add_argument("--mfcc-n", type=int, default=13)
    parser.add_argument("--dtw-window", type=int, default=80)

    # Pitch options
    parser.add_argument("--pitch-frame-ms", type=float, default=10.0)
    parser.add_argument("--pitch-win-ms", type=float, default=40.0)
    parser.add_argument("--pitch-fmin", type=float, default=50.0)
    parser.add_argument("--pitch-fmax", type=float, default=600.0)

    # Speaker encoder options
    parser.add_argument("--spk-ecapa-source", type=str, default="speechbrain/spkrec-ecapa-voxceleb",
                        help="SpeechBrain ECAPA source or local path.")
    parser.add_argument("--spk-ecapa-savedir", type=str, default=None,
                        help="SpeechBrain cache dir / local model dir (important offline).")
    parser.add_argument("--spk-device", type=str, default=None,
                        help="Device for speaker encoder (cpu/cuda). Default: same as --device.")
    parser.add_argument("--eer-neg-per-pos", type=int, default=5)
    parser.add_argument("--eer-seed", type=int, default=0)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spk_device = args.spk_device or device

    runs_root = Path(args.runs_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    user_cfg = load_yaml_config(args.config)
    cfg_global = merge_config(DEFAULT_CONFIG, {}, user_cfg)
    sr = int(cfg_global["data"].get("sample_rate", 16000))
    seglen = int(cfg_global["data"].get("segment_length", 32000))

    do_peak_norm = not args.no_peak_norm
    max_shift_samples = int(round((args.max_shift_ms / 1000.0) * sr))

    want_full = (args.metrics == "full")
    want_stoi = bool(want_full and (not args.no_stoi))
    want_pesq = bool(want_full and (not args.no_pesq))
    want_mcd = bool(want_full and (not args.no_mcd))
    want_f0 = bool(want_full and (not args.no_f0))
    want_spk = bool(want_full and (not args.no_spk))
    want_eer = bool(want_spk and (not args.no_eer))

    # dependency checks (explicit + early)
    if want_stoi and _stoi_fn is None:
        raise RuntimeError("Requested STOI but pystoi is not installed. Install with: pip install pystoi")
    if want_pesq and _pesq_fn is None:
        raise RuntimeError("Requested PESQ but pesq is not installed. Install with: pip install pesq")
    if want_spk and EncoderClassifier is None:
        raise RuntimeError("Requested speaker metrics but speechbrain is not installed. Install with: pip install speechbrain")

    logger.info(f"[RUN] device={device} | spk_device={spk_device}")
    logger.info(f"[CFG] sr={sr}, seglen={seglen}, eval_mode={args.eval_mode}, max_shift={max_shift_samples} samples")
    logger.info(f"[METRICS] mode={args.metrics} stoi={want_stoi} pesq={want_pesq} mcd={want_mcd} f0={want_f0} spk={want_spk} eer={want_eer}")

    runs = discover_runs(runs_root)
    if not runs:
        raise RuntimeError(f"No runs found under {runs_root}")

    test_inputs = [p.strip() for p in args.test_audio.split(",") if p.strip()]
    test_files = get_audio_files(test_inputs, prefer_audio_subdirs=True)
    logger.info(f"[AUDIO] TEST: found {len(test_files)} files from inputs={test_inputs[:3]}{'...' if len(test_inputs)>3 else ''}")

    if args.ckpt_policy == "option2":
        if not args.dev_audio:
            raise ValueError("option2 requires --dev-audio (separate dev set).")
        dev_inputs = [p.strip() for p in args.dev_audio.split(",") if p.strip()]
        dev_files = get_audio_files(dev_inputs, prefer_audio_subdirs=True)
        logger.info(f"[AUDIO] DEV:  found {len(dev_files)} files from inputs={dev_inputs[:3]}{'...' if len(dev_inputs)>3 else ''}")
    else:
        dev_files = []

    def each_run_has_ckpt() -> bool:
        for _, rd in runs:
            ckpt_dir = rd / "checkpoints"
            if not list_step_checkpoints(ckpt_dir):
                return False
        return True

    if args.wait_timeout_s > 0:
        wait_for_condition("each run has at least one checkpoint_step*.pt", each_run_has_ckpt,
                           timeout_s=args.wait_timeout_s, poll_s=args.wait_poll_s)

    # speaker encoder (global, reused)
    spk_enc = None
    if want_spk:
        spk_enc = build_speaker_encoder(args.spk_ecapa_source, args.spk_ecapa_savedir, device=spk_device)

    chosen: List[Dict[str, Any]] = []

    # -------------------------
    # OPTION 1
    # -------------------------
    if args.ckpt_policy == "option1":
        if args.anchor_step == "auto":
            max_steps = []
            for _, rd in runs:
                ms = max_step_in_dir(rd / "checkpoints")
                if ms is None:
                    max_steps.append(None)
                else:
                    max_steps.append(int(ms))
            if any(m is None for m in max_steps):
                raise RuntimeError("Some runs have no checkpoints; cannot use option1 auto anchor.")
            anchor = int(min(max_steps))
        else:
            anchor = int(args.anchor_step)

        if args.wait_timeout_s > 0:
            def can_pick_all() -> bool:
                for _, rd in runs:
                    ck = pick_ckpt_at_or_before(rd / "checkpoints", anchor, require_exact=args.require_exact_anchor)
                    if ck is None:
                        return False
                return True
            wait_for_condition(f"all runs have ckpt {'exact' if args.require_exact_anchor else '<='} anchor={anchor}",
                               can_pick_all, timeout_s=args.wait_timeout_s, poll_s=args.wait_poll_s)

        eval_dir = out_root / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        refs, selected, spk_ids, genders = prepare_refs(
            audio_paths=test_files, sr=sr, seglen=seglen, eval_mode=args.eval_mode,
            seed=args.seed, num_audios=args.test_num_audios,
            reuse_selection_file=None, out_dir=eval_dir, do_peak_norm=do_peak_norm,
            speaker_id_mode=args.speaker_id_mode, speaker_regex=args.speaker_regex,
            infer_gender=args.infer_gender
        )

        # precompute ref speaker embeddings once (for option1 eval set)
        ref_spk_embs = None
        if spk_enc is not None:
            ref_spk_embs = [spk_embed(spk_enc, refs[i].squeeze(0).cpu().to(torch.float32), device=spk_device) for i in range(len(refs))]

        all_rows: List[Dict[str, Any]] = []
        all_summaries: List[Dict[str, Any]] = []

        for n, rd in tqdm(runs, desc="Runs(option1)"):
            run_name = rd.name
            ckpt_dir = rd / "checkpoints"
            ckpt = pick_ckpt_at_or_before(ckpt_dir, anchor, require_exact=args.require_exact_anchor)
            if ckpt is None:
                logger.warning(f"[SKIP] {run_name}: no ckpt found for anchor={anchor}")
                continue

            rows, summ = eval_one_run_one_ckpt(
                run_name=run_name, n_layers=n, ckpt_path=ckpt,
                user_cfg=user_cfg, device=device, feature_mode=args.feature_mode,
                refs=refs, selected_paths=selected, spk_ids=spk_ids, genders=genders,
                sr=sr, seglen=seglen, eval_mode=args.eval_mode, overlap_default=args.overlap,
                out_dir=eval_dir, save_decoded=True,
                max_shift_samples=max_shift_samples,
                want_stoi=want_stoi, want_pesq=want_pesq, want_mcd=want_mcd, want_f0=want_f0,
                mfcc_n=args.mfcc_n, dtw_window=args.dtw_window,
                pitch_frame_ms=args.pitch_frame_ms, pitch_win_ms=args.pitch_win_ms,
                pitch_fmin=args.pitch_fmin, pitch_fmax=args.pitch_fmax,
                spk_enc=spk_enc, spk_device=spk_device, ref_spk_embs=ref_spk_embs,
                want_eer=want_eer, eer_neg_per_pos=args.eer_neg_per_pos, eer_seed=args.eer_seed
            )
            all_rows.extend(rows)
            all_summaries.append(summ)

            chosen.append({
                "run_name": run_name, "N": n, "policy": "option1",
                "anchor_step": anchor, "chosen_step": summ.get("step", -1),
                "checkpoint_path": str(ckpt)
            })

        metrics_csv = eval_dir / "metrics.csv"
        fields = list(all_rows[0].keys()) if all_rows else []
        write_csv(metrics_csv, all_rows, fields)
        logger.info(f"[SAVE] {metrics_csv}")

        summ_csv = eval_dir / "metrics_summary.csv"
        summ_fields = list(all_summaries[0].keys()) if all_summaries else []
        write_csv(summ_csv, all_summaries, summ_fields)
        logger.info(f"[SAVE] {summ_csv}")

        # summaries by gender / speaker
        if all_rows:
            metric_keys = [k for k in all_rows[0].keys() if k in [
                "rmse","mae","si_sdr_db_aligned","snr_db_aligned","log_mel_l1_aligned","stoi","pesq","mcd_dtw","f0_rmse_hz","vuv_f1","spk_cosine","dur_ratio"
            ]]
            by_gender = summarize(all_rows, group_keys=["run_name", "gender"], metric_keys_mean=metric_keys, extra_keys=["N","step","checkpoint_path"])
            write_csv(eval_dir / "metrics_summary_by_gender.csv", by_gender, list(by_gender[0].keys()) if by_gender else [])
            by_spk = summarize(all_rows, group_keys=["run_name", "speaker_id"], metric_keys_mean=metric_keys, extra_keys=["N","step","checkpoint_path"])
            write_csv(eval_dir / "metrics_summary_by_speaker.csv", by_spk, list(by_spk[0].keys()) if by_spk else [])

    # -------------------------
    # OPTION 2
    # -------------------------
    else:
        dev_dir = out_root / "dev"
        test_dir = out_root / "test"
        dev_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        dev_refs, dev_selected, dev_spk_ids, dev_genders = prepare_refs(
            audio_paths=dev_files, sr=sr, seglen=seglen, eval_mode=args.eval_mode,
            seed=args.seed, num_audios=args.dev_num_audios,
            reuse_selection_file=None, out_dir=dev_dir, do_peak_norm=do_peak_norm,
            speaker_id_mode=args.speaker_id_mode, speaker_regex=args.speaker_regex,
            infer_gender=args.infer_gender
        )

        test_refs, test_selected, test_spk_ids, test_genders = prepare_refs(
            audio_paths=test_files, sr=sr, seglen=seglen, eval_mode=args.eval_mode,
            seed=args.seed + 999, num_audios=args.test_num_audios,
            reuse_selection_file=None, out_dir=test_dir, do_peak_norm=do_peak_norm,
            speaker_id_mode=args.speaker_id_mode, speaker_regex=args.speaker_regex,
            infer_gender=args.infer_gender
        )

        # precompute ref speaker embeddings once per split (dev/test)
        dev_ref_spk_embs = None
        test_ref_spk_embs = None
        if spk_enc is not None:
            dev_ref_spk_embs = [spk_embed(spk_enc, dev_refs[i].squeeze(0).cpu().to(torch.float32), device=spk_device) for i in range(len(dev_refs))]
            test_ref_spk_embs = [spk_embed(spk_enc, test_refs[i].squeeze(0).cpu().to(torch.float32), device=spk_device) for i in range(len(test_refs))]

        selection_rows: List[Dict[str, Any]] = []
        chosen_ckpts: Dict[str, Path] = {}

        for n, rd in tqdm(runs, desc="Select(option2)"):
            run_name = rd.name
            ckpt_dir = rd / "checkpoints"
            ckpts = list_step_checkpoints(ckpt_dir)
            if not ckpts:
                logger.warning(f"[SKIP] {run_name}: no checkpoints")
                continue

            best = None
            best_key = None

            for ckpt in ckpts:
                rows, summ = eval_one_run_one_ckpt(
                    run_name=run_name, n_layers=n, ckpt_path=ckpt,
                    user_cfg=user_cfg, device=device, feature_mode=args.feature_mode,
                    refs=dev_refs, selected_paths=dev_selected, spk_ids=dev_spk_ids, genders=dev_genders,
                    sr=sr, seglen=seglen, eval_mode=args.eval_mode, overlap_default=args.overlap,
                    out_dir=dev_dir, save_decoded=args.save_dev_decoded,
                    max_shift_samples=max_shift_samples,
                    want_stoi=want_stoi, want_pesq=want_pesq, want_mcd=want_mcd, want_f0=want_f0,
                    mfcc_n=args.mfcc_n, dtw_window=args.dtw_window,
                    pitch_frame_ms=args.pitch_frame_ms, pitch_win_ms=args.pitch_win_ms,
                    pitch_fmin=args.pitch_fmin, pitch_fmax=args.pitch_fmax,
                    spk_enc=spk_enc, spk_device=spk_device, ref_spk_embs=dev_ref_spk_embs,
                    want_eer=False,  # EER not needed for selection loop
                )

                # Selection criterion (keep your original stable policy)
                k1 = float(summ.get("log_mel_l1_aligned_mean", 1e9))
                k2 = float(summ.get("si_sdr_db_aligned_mean", -1e9))
                key = (k1, -k2)

                selection_rows.append({
                    "run_name": run_name,
                    "N": n,
                    "step": int(summ.get("step", -1)),
                    "checkpoint_path": str(ckpt),
                    "log_mel_l1_aligned_mean": float(summ.get("log_mel_l1_aligned_mean", 1e9)),
                    "si_sdr_db_aligned_mean": float(summ.get("si_sdr_db_aligned_mean", -1e9)),
                    "finite_ok_rate": float(summ.get("finite_ok_rate", 0.0)),
                })

                if best is None or key < best_key:
                    best = ckpt
                    best_key = key

            if best is not None:
                chosen_ckpts[run_name] = best
                chosen.append({
                    "run_name": run_name, "N": n, "policy": "option2",
                    "anchor_step": -1, "chosen_step": parse_step_from_ckpt(best) or -1,
                    "checkpoint_path": str(best)
                })
                logger.info(f"[CHOSEN] {run_name} -> {best.name}")

        sel_csv = dev_dir / "selection_dev.csv"
        if selection_rows:
            write_csv(sel_csv, selection_rows, fieldnames=list(selection_rows[0].keys()))
            logger.info(f"[SAVE] {sel_csv}")

        all_rows: List[Dict[str, Any]] = []
        all_summaries: List[Dict[str, Any]] = []

        for n, rd in tqdm(runs, desc="Runs(test option2)"):
            run_name = rd.name
            if run_name not in chosen_ckpts:
                logger.warning(f"[SKIP] {run_name}: no chosen checkpoint")
                continue
            ckpt = chosen_ckpts[run_name]
            rows, summ = eval_one_run_one_ckpt(
                run_name=run_name, n_layers=n, ckpt_path=ckpt,
                user_cfg=user_cfg, device=device, feature_mode=args.feature_mode,
                refs=test_refs, selected_paths=test_selected, spk_ids=test_spk_ids, genders=test_genders,
                sr=sr, seglen=seglen, eval_mode=args.eval_mode, overlap_default=args.overlap,
                out_dir=test_dir, save_decoded=True,
                max_shift_samples=max_shift_samples,
                want_stoi=want_stoi, want_pesq=want_pesq, want_mcd=want_mcd, want_f0=want_f0,
                mfcc_n=args.mfcc_n, dtw_window=args.dtw_window,
                pitch_frame_ms=args.pitch_frame_ms, pitch_win_ms=args.pitch_win_ms,
                pitch_fmin=args.pitch_fmin, pitch_fmax=args.pitch_fmax,
                spk_enc=spk_enc, spk_device=spk_device, ref_spk_embs=test_ref_spk_embs,
                want_eer=want_eer, eer_neg_per_pos=args.eer_neg_per_pos, eer_seed=args.eer_seed
            )
            all_rows.extend(rows)
            all_summaries.append(summ)

        metrics_csv = test_dir / "metrics.csv"
        fields = list(all_rows[0].keys()) if all_rows else []
        write_csv(metrics_csv, all_rows, fields)
        logger.info(f"[SAVE] {metrics_csv}")

        summ_csv = test_dir / "metrics_summary.csv"
        summ_fields = list(all_summaries[0].keys()) if all_summaries else []
        write_csv(summ_csv, all_summaries, summ_fields)
        logger.info(f"[SAVE] {summ_csv}")

        # summaries by gender / speaker
        if all_rows:
            metric_keys = [k for k in all_rows[0].keys() if k in [
                "rmse","mae","si_sdr_db_aligned","snr_db_aligned","log_mel_l1_aligned","stoi","pesq","mcd_dtw","f0_rmse_hz","vuv_f1","spk_cosine","dur_ratio"
            ]]
            by_gender = summarize(all_rows, group_keys=["run_name", "gender"], metric_keys_mean=metric_keys, extra_keys=["N","step","checkpoint_path"])
            write_csv(test_dir / "metrics_summary_by_gender.csv", by_gender, list(by_gender[0].keys()) if by_gender else [])
            by_spk = summarize(all_rows, group_keys=["run_name", "speaker_id"], metric_keys_mean=metric_keys, extra_keys=["N","step","checkpoint_path"])
            write_csv(test_dir / "metrics_summary_by_speaker.csv", by_spk, list(by_spk[0].keys()) if by_spk else [])

    chosen_csv = out_root / "chosen_checkpoints.csv"
    if chosen:
        write_csv(chosen_csv, chosen, fieldnames=list(chosen[0].keys()))
        logger.info(f"[SAVE] {chosen_csv}")

    logger.info("[DONE]")

if __name__ == "__main__":
    main()
