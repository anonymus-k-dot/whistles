"""
stage2_stft1.py -- Stage 2: STFT #1 on 192 kHz High-Pass Filtered Audio.

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Section 1-3, D4, D20)
  config_paper_test.yaml (block: stft1)

Lineage Categories:
  - Fs = 192,000 Hz: [PAPER-SPECIFIED] Sec. II.C
  - Window = Hamming, 2048 points: [PAPER-SPECIFIED] Sec. II.C
  - Overlap = 50% (hop = 1024 points): [DERIVED FROM PAPER] Sec. II.C
  - Frequency resolution = 93.75 Hz: [DERIVED FROM PAPER] 192000 / 2048 = 93.75 Hz
  - Time resolution ≈ 5.333 ms: [DERIVED FROM PAPER] 1024 / 192000 = 5.3333 ms
  - Frame center time: t_k = (k*hop + nfft/2)/fs_out: [IMPLEMENTATION REQUIRED] (D20)
  - All 6 channels share identical frame grid and time origin: [IMPLEMENTATION REQUIRED] (D20)
"""

import os
import sys
import numpy as np

def get_hamming_window(nfft=2048):
    return np.hamming(nfft).astype(np.float64)

def compute_stft1(channel_audio, fs=192000, nfft=2048, hop=1024):
    """
    Compute STFT #1 on single channel 1D audio.
    
    Parameters:
      channel_audio: 1D array of float32/float64 samples at fs=192 kHz
      fs: Sample rate (192000 Hz)
      nfft: Window length and FFT size (2048)
      hop: Step size (1024)
      
    Returns:
      stft_complex: 2D array of shape (n_bins, n_frames), complex64
      times_s: 1D array of frame center times in seconds from file start
      freqs_hz: 1D array of bin center frequencies in Hz
    """
    audio = np.asarray(channel_audio, dtype=np.float64)
    n_samples = len(audio)
    
    if n_samples < nfft:
        raise ValueError(f"Audio length {n_samples} is shorter than nfft {nfft}")
        
    n_frames = (n_samples - nfft) // hop + 1
    win = get_hamming_window(nfft)
    
    # Strided framing
    shape = (n_frames, nfft)
    strides = (audio.strides[0] * hop, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    
    # Window and rfft
    windowed_frames = frames * win
    spec = np.fft.rfft(windowed_frames, n=nfft, axis=1)
    stft_complex = spec.T.astype(np.complex64)
    
    # Common time coordinate
    frame_indices = np.arange(n_frames, dtype=np.float64)
    times_s = (frame_indices * hop + (nfft / 2.0)) / fs
    
    n_bins = nfft // 2 + 1
    freqs_hz = np.arange(n_bins, dtype=np.float64) * (fs / nfft)
    
    return stft_complex, times_s, freqs_hz

def compute_spectrogram_db(stft_complex, eps=1e-12):
    """Compute log magnitude (dB) spectrogram from complex STFT."""
    mag = np.abs(stft_complex)
    return 20.0 * np.log10(mag + eps)
