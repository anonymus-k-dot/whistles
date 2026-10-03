import os
import sys
import argparse
import soundfile as sf
import numpy as np
import scipy.signal as signal

AUDIO_DIR = "encounter_191_audio"

def main():
    parser = argparse.ArgumentParser(description="Convert 500kHz bioacoustic FLAC into audible 48kHz WAV")
    parser.add_argument("filename", type=str, help="Filename (e.g. 1705_20170930_014847_062.flac)")
    parser.add_argument("--channel", type=int, default=0, help="Channel index 0-5 (default: 0)")
    parser.add_argument("--duration", type=int, default=30, help="Seconds to preview (default: 30)")
    parser.add_argument("--slow", type=float, default=1.0, help="Slow down factor (e.g. 5.0 to hear ultrasound clicks in human range)")
    args = parser.parse_args()

    # Find file
    fname = os.path.basename(args.filename)
    path = os.path.join(AUDIO_DIR, fname)
    if not os.path.exists(path):
        path = args.filename
    if not os.path.exists(path):
        print(f"Error: File '{fname}' not found.")
        return

    info = sf.info(path)
    sr = info.samplerate
    read_frames = min(int(args.duration * sr), info.frames)
    
    print(f"Reading {args.duration}s from {fname} (Channel {args.channel}, original SR: {sr} Hz)...")
    audio, _ = sf.read(path, frames=read_frames)
    
    if audio.ndim > 1:
        ch_data = audio[:, args.channel]
    else:
        ch_data = audio

    # Resample from 500,000 Hz to 48,000 Hz
    # Ratio: 48 / 500 = 12 / 125
    target_sr = 48000
    print("Resampling to standard 48 kHz and normalizing volume...")
    ch_48k = signal.resample_poly(ch_data, 12, 125)

    # Normalize volume
    peak = np.max(np.abs(ch_48k))
    if peak > 0:
        ch_norm = (ch_48k / peak) * 0.95
    else:
        ch_norm = ch_48k

    out_name = f"audible_{os.path.splitext(fname)[0]}.wav"
    out_sr = int(target_sr / args.slow)
    sf.write(out_name, ch_norm, out_sr)
    
    print(f"\nCreated: {os.path.abspath(out_name)}")
    print(f"Sample Rate: {out_sr} Hz")
    
    # Auto-play on Windows
    if sys.platform == "win32":
        try:
            os.startfile(out_name)
            print("Opening in your default audio player...")
        except Exception:
            pass

if __name__ == "__main__":
    main()
