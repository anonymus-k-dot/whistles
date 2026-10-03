# Whale Whistle Processing Pipeline

This folder organizes all the metadata, documentation, and tooling required for acoustic analysis and preprocessing of false killer whale and odontocete whistle encounters.

## Directory Structure

```
d:\ALL\whistles\
├── encounter_191_audio/          # Raw 500 kHz FLAC audio recordings (363 files, ~68 GB)
└── whistle_pipeline/
    ├── docs/                     # Dataset documentation and reference papers
    │   ├── Dataset_DCLDE_Oahu.pdf
    │   └── VAE-GMM Cetacean Wistles.pdf
    ├── metadata/                 # Ground truth logs and activity scores
    │   ├── DCLDE2020_DetectionData.xlsx
    │   ├── whale_encounters_1705.csv
    │   └── audio_activity_scores.csv
    └── tools/                    # Utility scripts for downloading and scanning audio
        ├── download_audio.py
        ├── inspect_encounters.py
        ├── play_file.py
        └── scan_audio_activity.py
```

## Folder Details

- **`docs/`**: Reference paper on VAE-GMM cetacean whistle extraction and NOAA Oahu DCLDE dataset technical documentation.
- **`metadata/`**: Original DCLDE Excel encounter log, extracted encounter time windows (`whale_encounters_1705.csv`), and acoustic whistle SNR & click crest factor scores (`audio_activity_scores.csv`).
- **`tools/`**:
  - `download_audio.py`: Downloads specific encounters or time windows from NOAA GCS bucket.
  - `inspect_encounters.py`: Parses the DCLDE dataset spreadsheet to extract target odontocete encounters.
  - `scan_audio_activity.py`: Multi-process whistle SNR & click detector to prioritize active files.
  - `play_file.py`: Resamples high-frequency 500 kHz FLAC files down to 48 kHz WAV for human listening.
