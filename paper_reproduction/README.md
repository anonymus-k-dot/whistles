# Paper-Reproduction Preprocessing Pipeline (363-File Full Dataset Ready)

**Target Paper:** G.-H. Park et al., *"A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of Gaussians,"* **IEEE Access**, vol. 14, pp. 27877–27896, 2026.  
**Dataset Reference:** NOAA DCLDE 2022 Cruise 1705, Encounter 191 (*Pseudorca crassidens* / False killer whale), continuous 6.05 hours  
**Raw Source Dataset:** `d:\ALL\whistles\encounter_191_audio\` (363 FLAC files, 6 channels, 500 kHz, 60.0 s each)

---

## 1. Package Architecture

```
paper_reproduction/
├── config_paper_reproduction.yaml   # Authoritative YAML configuration
├── manifest_363.csv                 # Cryptographic input inventory (SHA-256 for all 363 files)
├── PAPER_PARAMETER_LINEAGE.md       # Exhaustive 27-item epistemic classification matrix
├── PAPER_REPRODUCTION_AUDIT.md      # Detailed audit report and known limitations
├── README.md                        # Package documentation and CLI reference
├── h1_image.py                      # NFM synthesis, STFT #2 derivation, and staleness guard
├── stage1_audio.py                  # Polyphase resampling (48/125 -> 192 kHz) and zero-phase HPF > 2 kHz
├── stage2_stft1.py                  # STFT #1: 2048 Hamming, 50% overlap, 93.75 Hz, 5.33 ms
├── stage3_detector.py               # Stand-in ridge tracker ([ENGINEERING SUBSTITUTION])
├── stage4_postprocess.py            # Duration filtering (>= 10 frames / 53.3 ms)
├── stage5_dedup.py                  # Cross-channel deduplication module (DISABLED by default)
├── stage6_vectors.py                # Dual-representation time-frequency vectors
├── stage7_nfm.py                    # Center frequency removal and NFM baseband synthesis
├── stage8_image.py                  # STFT #2 and 128x128 continuous grayscale image construction
├── stage9_qc.py                     # Internal implementation validation (0.05 decode check)
├── visualize.py                     # Diagnostic plotting module
├── run_pipeline.py                  # Master runner with --dry-run and --full-run CLI flags
├── outputs/                         # Output target directories (staged)
├── logs/                            # Execution logs
└── tests/
    ├── test_preflight.py            # 12 static and pre-flight validation unit tests
    └── run_all_tests.py             # Test suite runner
```

---

## 2. CLI Execution Commands

### A. Pre-flight Validation Dry Run (Zero Audio Processed)
```powershell
python "paper_reproduction/run_pipeline.py" --config "paper_reproduction/config_paper_reproduction.yaml" --dry-run
```
- Discovers and validates all 363 files.
- Verifies SHA-256 and formatting against `manifest_363.csv`.
- Confirms H1 parameter derivation freshness.
- Reports:
  - `files discovered = 363`
  - `files that WOULD be processed = 363`
  - `files ACTUALLY processed = 0`

### B. Run Pre-flight Unit Tests
```powershell
python "paper_reproduction/tests/run_all_tests.py"
```

### C. Full 363-File Preprocessing Execution (Authorized Execution Only)
```powershell
python "paper_reproduction/run_pipeline.py" --config "paper_reproduction/config_paper_reproduction.yaml" --full-run
```
*(NOTE: Requires explicit user authorization before execution).*

---

## 3. Scientific Claim Boundaries

> **This pipeline is a paper-faithful reconstruction to the extent supported by the available paper and source evidence.**  
> **Exact detector-level reproduction is not currently possible because the paper's detector implementation and neural network weights are unavailable.**
