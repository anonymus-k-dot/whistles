import os
import csv
import soundfile as sf
import numpy as np
import scipy.signal as signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

AUDIO_DIR = "encounter_191_audio"

def analyze_file(fname):
    filepath = os.path.join(AUDIO_DIR, fname)
    try:
        info = sf.info(filepath)
        sr = info.samplerate
        
        # Read two 5-second windows (seconds 10-15 and 35-40) to be fast
        windows = [int(10 * sr), int(35 * sr)]
        frame_len = int(5 * sr)
        
        max_whistle_snr = 0.0
        max_crest_factor = 0.0
        max_rms = 0.0
        
        for w_start in windows:
            audio, _ = sf.read(filepath, start=w_start, frames=frame_len)
            ch = audio[:, 0]  # Channel 0
            
            # Decimate to 48kHz (500kHz -> 48kHz: ratio 12/125)
            ch_48k = signal.resample_poly(ch, 12, 125)
            
            # Compute Spectrogram
            f, t, Sxx = signal.spectrogram(ch_48k, fs=48000, nperseg=1024, noverlap=512)
            
            # False Killer Whale Whistle Band: 3 kHz - 18 kHz
            band_mask = (f >= 3000) & (f <= 18000)
            band_Sxx = Sxx[band_mask, :]
            
            # Peak to median ratio in whistle band
            peak_to_med = np.max(band_Sxx) / (np.median(band_Sxx) + 1e-12)
            whistle_snr = 10 * np.log10(peak_to_med)
            
            # Click / transient crest factor (Peak / RMS)
            crest = np.max(np.abs(ch)) / (np.std(ch) + 1e-12)
            rms = float(np.std(ch))
            
            if whistle_snr > max_whistle_snr:
                max_whistle_snr = whistle_snr
            if crest > max_crest_factor:
                max_crest_factor = crest
            if rms > max_rms:
                max_rms = rms

        # Activity classification
        # Whistle SNR > 30 dB indicates strong tonal whistle components
        # Crest Factor > 10 indicates distinct echolocation clicks
        if max_whistle_snr >= 32.0 or max_crest_factor >= 15.0:
            activity = "HIGH"
        elif max_whistle_snr >= 22.0 or max_crest_factor >= 8.0:
            activity = "MEDIUM"
        else:
            activity = "LOW (Ambient)"

        return {
            "filename": fname,
            "activity": activity,
            "whistle_snr_db": round(max_whistle_snr, 1),
            "click_crest_factor": round(max_crest_factor, 1),
            "rms_energy": round(max_rms, 4)
        }
    except Exception as e:
        return {
            "filename": fname,
            "activity": f"ERROR: {str(e)}",
            "whistle_snr_db": 0,
            "click_crest_factor": 0,
            "rms_energy": 0
        }

def main():
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: {AUDIO_DIR} does not exist.")
        return

    files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".flac")])
    print(f"Scanning {len(files)} files in '{AUDIO_DIR}' for whale sounds (whistles & clicks)...")

    results = []
    # Use ProcessPoolExecutor for CPU-bound signal processing
    workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyze_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Scanning"):
            results.append(future.result())

    # Sort by Whistle SNR descending
    results.sort(key=lambda x: (x["whistle_snr_db"], x["click_crest_factor"]), reverse=True)

    # Save to CSV
    csv_out = "audio_activity_scores.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    high_count = sum(1 for r in results if r["activity"] == "HIGH")
    med_count = sum(1 for r in results if r["activity"] == "MEDIUM")
    low_count = sum(1 for r in results if r["activity"] == "LOW (Ambient)")

    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY:")
    print(f"  • HIGH Vocal Activity (Strong whistles/clicks):   {high_count} files")
    print(f"  • MEDIUM Vocal Activity (Moderate whistles/clicks): {med_count} files")
    print(f"  • LOW / Ambient (Mostly water/ship background):     {low_count} files")
    print("=" * 70)

    print("\nTOP 15 MOST ACTIVE AUDIO FILES:")
    print(f"{'Filename':<32} | {'Activity':<8} | {'Whistle SNR':<12} | {'Click Crest'}")
    print("-" * 70)
    for r in results[:15]:
        print(f"{r['filename']:<32} | {r['activity']:<8} | {r['whistle_snr_db']:>5.1f} dB    | {r['click_crest_factor']:>5.1f}")
    
    print(f"\nComplete rankings saved to: {os.path.abspath(csv_out)}")

if __name__ == "__main__":
    main()
