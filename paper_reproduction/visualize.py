"""
visualize.py -- Diagnostic Visualizations for the 10-File Paper-Reproduction Preprocessing Test.

Generates representative visualizations for:
  1. raw six-channel audio
  2. STFT #1 spectrogram
  3. whistle detector output
  4. postprocessed contours
  5. time-frequency vectors
  6. NFM waveform/baseband
  7. STFT #2 spectrogram
  8. final 128x128 image
  9. round-trip validation contour overlay
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_raw_six_channel_audio(raw_audio_int16, fs=500000, out_path="visualizations/01_raw_six_channel_audio.png", filename=""):
    """Plot raw 6-channel audio waveform."""
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    n_samples = raw_audio_int16.shape[0]
    t = np.arange(n_samples) / fs
    step = 50
    t_sub = t[::step]
    
    for ch in range(6):
        ax = axes[ch]
        sig_sub = raw_audio_int16[::step, ch]
        ax.plot(t_sub, sig_sub, color="#1f77b4", lw=0.6)
        ax.set_ylabel(f"Ch {ch}\n(int16)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-33000, 33000)
        
    axes[-1].set_xlabel("Time (seconds from file start)", fontsize=11)
    fig.suptitle(f"Raw 6-Channel Hydrophone Waveforms (500 kHz, ~60s)\nFile: {filename}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_stft1_spectrogram(S_db, times_s, freqs_hz, out_path="visualizations/02_stft1_spectrogram.png", filename="", ch=0):
    """Plot STFT #1 log-magnitude spectrogram with 2 kHz edge."""
    fig, ax = plt.subplots(figsize=(12, 6))
    f_mask = freqs_hz <= 25000
    extent = [times_s[0], times_s[-1], freqs_hz[0] / 1000.0, freqs_hz[f_mask][-1] / 1000.0]
    
    im = ax.imshow(
        S_db[f_mask, :],
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=np.percentile(S_db[f_mask, :], 15),
        vmax=np.percentile(S_db[f_mask, :], 99.8)
    )
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.axhline(2.0, color="cyan", linestyle="--", lw=1.2, label="High-pass Passband Edge (2 kHz)")
    ax.set_ylabel("Frequency (kHz)", fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title(f"STFT #1 Spectrogram (192 kHz, 2048-pt Hamming, 50% Overlap)\nFile: {filename}, Channel: {ch}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_whistle_detector_output(S_db, times_s, freqs_hz, raw_contours, out_path="visualizations/03_whistle_detector_output.png", filename="", ch=0):
    """Plot detected whistle contours overlaid on spectrogram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    f_mask = (freqs_hz >= 1500) & (freqs_hz <= 22000)
    extent = [times_s[0], times_s[-1], freqs_hz[f_mask][0] / 1000.0, freqs_hz[f_mask][-1] / 1000.0]
    
    im = ax.imshow(
        S_db[f_mask, :],
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=np.percentile(S_db[f_mask, :], 20),
        vmax=np.percentile(S_db[f_mask, :], 99.5)
    )
    
    colors = plt.cm.tab20.colors
    for i, c in enumerate(raw_contours):
        color = colors[i % len(colors)]
        t = np.array(c["times_s"])
        f_khz = np.array(c["frequencies_hz"]) / 1000.0
        ax.plot(t, f_khz, color=color, lw=1.8, alpha=0.9)
        
    ax.set_ylabel("Frequency (kHz)", fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title(f"Stage 3: Whistle Detector Output (Stand-in Substitution)\nFile: {filename}, Channel: {ch} ({len(raw_contours)} candidates)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_postprocessed_contours(accepted, rejected, times_range, out_path="visualizations/04_postprocessed_contours.png", filename="", ch=0):
    """Plot postprocessed contours (accepted vs rejected)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for r in rejected:
        t = np.array(r["times_s"])
        f_khz = np.array(r["frequencies_hz"]) / 1000.0
        ax.plot(t, f_khz, color="red", linestyle=":", lw=1.2, alpha=0.5)
        
    for i, a in enumerate(accepted):
        t = np.array(a["times_s"])
        f_khz = np.array(a["frequencies_hz"]) / 1000.0
        ax.plot(t, f_khz, color="#0055ff", lw=2.0, alpha=0.9)
        
    ax.set_ylabel("Frequency (kHz)", fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylim(2.0, 20.0)
    ax.set_xlim(times_range[0], times_range[1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Stage 4: Postprocessed Contours (Accepted: {len(accepted)}, Short Rejected: {len(rejected)})\nFile: {filename}, Channel: {ch}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_time_frequency_vectors(channel_contours_map, out_path="visualizations/05_time_frequency_vectors.png", filename=""):
    """Plot whistle time-frequency vectors across all 6 hydrophones."""
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True, sharey=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for ch in range(6):
        ax = axes[ch]
        c_list = channel_contours_map.get(ch, [])
        for c in c_list:
            t = np.array(c["times_s"])
            f_khz = np.array(c["frequencies_hz"]) / 1000.0
            ax.plot(t, f_khz, color=colors[ch], lw=1.8)
        ax.set_ylabel(f"Ch {ch}\n(kHz)", fontsize=9)
        ax.set_ylim(2.0, 20.0)
        ax.grid(True, linestyle="--", alpha=0.4)
        
    axes[-1].set_xlabel("Time (seconds from file start)", fontsize=11)
    fig.suptitle(f"Stage 5: Whistle Time-Frequency Vectors (All 6 Channels, No Dedup)\nFile: {filename}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_nfm_waveform(s_bb, fs_bb=12000, out_path="visualizations/06_nfm_waveform_baseband.png", cid=""):
    """Plot constant-amplitude NFM baseband waveform."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(s_bb)) / fs_bb * 1000.0 # ms
    
    ax1.plot(t, np.real(s_bb), color="blue", lw=1.0, label="Real (I)")
    ax1.plot(t, np.imag(s_bb), color="orange", lw=1.0, alpha=0.7, label="Imag (Q)")
    ax1.plot(t, np.abs(s_bb), color="red", linestyle="--", lw=1.5, label="Constant Envelope (|s(t)| = 1.0)")
    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.set_ylim(-1.3, 1.3)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")
    ax1.set_title(f"Stage 6: Constant-Amplitude NFM Baseband Waveform (Fs = 12 kHz, A = 1.0)\nContour ID: {cid}", fontsize=11, fontweight="bold")
    
    phase = np.unwrap(np.angle(s_bb))
    inst_f = np.diff(phase) / (2.0 * np.pi) * fs_bb
    t_f = t[:-1] + (t[1] - t[0]) / 2.0
    ax2.plot(t_f, inst_f, color="purple", lw=1.5)
    ax2.set_ylabel("Baseband Freq (Hz)", fontsize=11)
    ax2.set_xlabel("Time (ms)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_stft2_spectrogram(img, out_path="visualizations/07_stft2_spectrogram.png", cid=""):
    """Plot STFT #2 baseband spectrogram."""
    fig, ax = plt.subplots(figsize=(7, 7))
    extent = [-0.3413, 0.3413, -6.0, 6.0]
    
    im = ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0
    )
    plt.colorbar(im, ax=ax, label="Normalized Magnitude [0, 1]")
    ax.axhline(0.0, color="red", linestyle=":", lw=1.2, label="DC Center (0 Hz)")
    ax.set_ylabel("Baseband Frequency (kHz)", fontsize=11)
    ax.set_xlabel("Time relative to center (seconds)", fontsize=11)
    ax.set_title(f"Stage 7: STFT #2 Baseband Spectrogram (12 kHz, NFFT 128, Hop 64)\nContour ID: {cid}", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_final_image(img, out_path="visualizations/08_final_128x128_image.png", cid=""):
    """Plot final 128x128 continuous grayscale image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(
        img,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Continuous Grayscale [0, 1]")
    ax.set_xlabel("Time Frame Column (0 - 127)", fontsize=11)
    ax.set_ylabel("Frequency Row (0 - 127)", fontsize=11)
    ax.axhline(64, color="cyan", linestyle=":", lw=1.0, alpha=0.7, label="Center Row 64")
    ax.set_title(f"Stage 7: Final 128x128 Continuous Grayscale Image\n{cid}", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_validation_roundtrip(orig_rows, dec_rows_matched, out_path="visualizations/09_roundtrip_validation_comparison.png", cid=""):
    """Plot original vs 0.05-decoded contour."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    frames = np.arange(len(orig_rows))
    
    ax1.plot(frames, orig_rows, color="blue", lw=2.2, label="Original Contour Row")
    ax1.plot(frames, dec_rows_matched, color="red", linestyle="--", lw=1.8, label="Decoded Centroid Row (0.05 threshold)")
    ax1.set_ylabel("Image Row (0 - 127)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    ax1.set_title(f"Internal Validation: Original vs Decoded Contour (Threshold 0.05)\nContour: {cid}", fontsize=11, fontweight="bold")
    
    residuals = np.array(dec_rows_matched) - np.array(orig_rows)
    ax2.plot(frames, residuals, color="purple", lw=1.5)
    ax2.axhline(0.0, color="black", linestyle="-", lw=0.8)
    ax2.axhline(1.0, color="red", linestyle=":", lw=1.0, label="+1 Bin Tolerance")
    ax2.axhline(-1.0, color="red", linestyle=":", lw=1.0, label="-1 Bin Tolerance")
    ax2.set_ylabel("Residual (Bins)", fontsize=11)
    ax2.set_xlabel("Contour Frame Index", fontsize=11)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
