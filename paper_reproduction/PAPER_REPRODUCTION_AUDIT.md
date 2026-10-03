# Paper-Reproduction Preprocessing Audit & Scientific Provenance Report

**Target Research Paper:**  
G.-H. Park, H.-S. Jeon, J.-H. Baek, S. Huh, and H.-S. Cho, *"A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of Gaussians,"* **IEEE Access**, vol. 14, pp. 27877–27896, 2026.

**Document Purpose:**  
This document records the rigorous scientific audit conducted following the 10-file paper-reproduction preprocessing test. It details the empirical observations, identifies discrepancies where previous implementation choices were improperly conflated with paper specifications, documents the corrective actions taken, and establishes the formal boundaries of what remains unresolved.

---

## 1. What Was Tested in the 10-File Experiment

The 10-file paper-reproduction preprocessing test executed an end-to-end pipeline across a sample of 10 raw acoustic recordings copied from the local 363-file NOAA Cruise 1705 Encounter 191 false killer whale dataset (`encounter_191_audio/`).

### Test Execution Scope:
- **Input Data:** Exactly 10 raw FLAC recordings (6 channels, 500 kHz, 60.0 s each, totaling 600.0 s of 6-channel audio).
- **Sampling Strategy:** Deterministic timeline sampling across the 6.05-hour encounter timeline with zero overlap against previous Phase-1 copies.
- **Pipeline Stages Evaluated:**
  1. Audio format validation and 48/125 polyphase resampling ($500\text{ kHz} \to 192\text{ kHz}$).
  2. 4th-order zero-phase Butterworth high-pass filtering ($> 2\text{ kHz}$, design cutoff $1791.439\text{ Hz}$).
  3. STFT #1 ($N_{\text{FFT}}=2048$, Hamming, $50\%$ overlap, $93.75\text{ Hz}$ frequency spacing, $5.333\text{ ms}$ hop).
  4. Whistle contour detection using a classical ridge tracker stand-in across $[2000, 20000]\text{ Hz}$.
  5. Minimum duration filtering ($\ge 10\text{ frames} / 53.33\text{ ms}$).
  6. Time-frequency vector extraction with cross-channel deduplication **disabled** (preserving all 6 hydrophones).
  7. Center-frequency removal using midrange $(f_{\min} + f_{\max})/2$.
  8. Constant-amplitude NFM baseband synthesis at $12\text{ kHz}$ sampling rate.
  9. STFT #2 ($N_{\text{FFT}}=128$, hop 64, Hamming-128) and $128 \times 128$ continuous grayscale image construction.
  10. Internal round-trip contour decoding validation at threshold 0.05.

### Quantitative Yield:
- **Candidate Tracks Detected:** 792,101 across 60 channels.
- **After Post-Processing Filter ($\ge 10$ frames):** 29,116 accepted contours.
- **Whistle Time-Frequency Vectors:** 29,116 vectors.
- **Overlong Whistles Dropped ($> 128$ frames / $0.683\text{ s}$):** 302 contours ($1.04\%$).
- **Supported NFM Waveforms Synthesized:** 28,814.
- **Final $128 \times 128$ Continuous Grayscale Images:** 28,814 float32 images in $[0.0, 1.0]$.
- **Internal Validation Pass Rate:** $97.81\%$ ($28,183$ passed, $631$ quarantined for high-sweep window smearing, $0$ integrity failures).

---

## 2. Identified Implementation Problems & Lineage Misclassifications

The post-test audit revealed several critical epistemic errors in the previous 10-file reporting where implementation choices and engineering hypotheses were erroneously classified as "derived from paper":

### Error 1: STFT #2 and Baseband Parameters Were Falsely Labeled `[DERIVED FROM PAPER]`
- **Previous Finding:** The 10-file summary classified the baseband sampling rate ($12\text{ kHz}$), STFT #2 FFT size ($N_{\text{FFT}} = 128$), hop size ($64\text{ samples}$), Hamming window, DC center row (row 64), and temporal centering formula as `[DERIVED FROM PAPER]`.
- **Authoritative Text Audit:** A literal, exhaustive text search of the 18-page research paper revealed:
  - The term `12 kHz` appears **0 times** in the paper.
  - The number `12,000` appears only as the number of GMM mixture components $K=12,000$ in Section IV.
  - The term `NFFT` appears **0 times**.
  - The term `hop` appears only inside "Workshop" and "Ganhacks".
  - The term `Hamming` appears **exactly once**, describing STFT #1 only.
  - The term `64` appears only as the VAE latent dimension $L=64$.
- **Corrected Epistemic Reality:** The paper states only:
  > *"Each vector was then used to generate a constant-amplitude NFM baseband signal and its spectrogram, which was further transformed into a 128 × 128 grayscale image (See Figs. 1 and 3)."*
  The entire $12\text{ kHz}$ / $N_{\text{FFT}}=128$ / $\text{hop}=64$ formulation is **Hypothesis H1**—an unresolved engineering hypothesis consistent with Figure 2's $\pm 6\text{ kHz}$ axis, but **NOT** a paper-derived mathematical certainty.
- **Correction Applied:** Reclassified all STFT #2 / H1 parameters from `[DERIVED FROM PAPER]` to `[NOT SPECIFIED BY PAPER] / [IMPLEMENTATION REQUIRED] (Hypothesis H1)`.

### Error 2: High-Pass Design Cutoff ($1791.4\text{ Hz}$) Conflated with Paper Specification
- **Previous Finding:** The design cutoff frequency $1791.439\text{ Hz}$ was previously listed alongside the passband edge.
- **Authoritative Text Audit:** The paper states only: *"a high-pass filter with a passband above 2 kHz was applied"*. It does not specify the filter family, order, or cutoff.
- **Corrected Epistemic Reality:** $1791.439\text{ Hz}$ is purely an implementation realization of a 4th-order Butterworth filter to ensure $-3.01\text{ dB}$ attenuation at $2000\text{ Hz}$ under forward-backward zero-phase squaring.
- **Correction Applied:** Labeled strictly as `[IMPLEMENTATION REQUIRED]`.

### Error 3: Midrange Center Frequency Calculation Was Unverified
- **Previous Finding:** $(f_{\min} + f_{\max})/2$ was applied as default center frequency.
- **Authoritative Text Audit:** The paper states in Section II.B and Appendix C: *"with the center frequency removed as in Fig. 1"* and *"The center frequency was excluded from the conditional generation results, as it was normalized during the preprocessing"*. The exact formula (midrange vs mean vs median vs peak-energy frequency) is not defined.
- **Correction Applied:** Midrange calculation is retained because it centers the frequency trajectory symmetrically within the baseband canvas, but it is explicitly tagged as `[NOT SPECIFIED BY PAPER]`.

### Error 4: Image Intensity Normalization
- **Previous Finding:** Constant window-sum division was treated as standard.
- **Authoritative Text Audit:** Figure 1 caption states: *"normalized to have intensities between 0 and 1, with the whistle regions having values close to 1"*. The mathematical normalization method is not stated.
- **Correction Applied:** Classified as `[IMPLEMENTATION REQUIRED]`.

---

## 3. Whistle Contour Detector Status & Reproducibility Assessment

This is the central scientific limitation of any reproduction of Park et al. (2026):

### 1. Paper-Stated Methodology
Section II.C explicitly specifies:
> *"In the second stage, a neural network-based whistle extraction method [31] was applied to detect individual whistle contours. The extraction process was combined with a post-processing algorithm [32], [33] that converts detection outputs into individual whistle tracks."*

### 2. Status of References [31], [32], [33]
- **[31]** G.-H. Park, J. Ahn, W. Kim, I. S. Kim, and D. H. Lee, *"Implementation of a neural network for cetacean whistle frequency detection based on data synthesis,"* in Proc. Winter Conf. Korean Inst. Commun. Inf. Sci. (KICS), 2024, pp. 690–691.
- **[32]** G.-H. Park, J. Ahn, W. Kim, S. Lee, and D. Lee, *"A design of MCA-CFAR detector for extraction of cetacean whistle contours,"* in Proc. Winter Conf. Korean Inst. Commun. Inf. Sci. (KICS), 2024, pp. 1474–1475.
- **[33]** G. H. Park, I. S. Kim, S. G. Lee, and D. H. Lee, *"Detection and tracking-based dolphin whistle extraction method,"* in Proc. Korea Inst. Mil. Sci. Technol. Annu. Conf., 2024, pp. 382–383.

All three references are 2-page extended abstracts in Korean conference proceedings.
An exhaustive search of academic databases, author profiles, and GitHub confirmed that:
- **No neural network weights were released.**
- **No training datasets or data-synthesis generators were published.**
- **No detector code or MCA-CFAR parameter tables exist online.**
- **No official repository exists for Park et al. (2026).**

### 3. Engineering Substitution Details
We use a **Classical Ridge Tracker Stand-In** with:
- 3-point parabolic sub-bin frequency interpolation.
- Per-bin median background subtraction.
- Gap-bridging across up to 2 frames.
- Minimum local contrast threshold of $4.0\text{ dB}$.

### 4. Consequence for Quantitative Distribution
- **Paper Full Dataset:** 2,486 files produced 228,673 whistle time-frequency vectors ($\approx 92.0\text{ whistles/file}$).
- **Our Test Pipeline:** With deduplication OFF across 6 hydrophones, our stand-in ridge tracker detected $\approx 2,881.4\text{ whistle images/file}$.
- **Root Cause:** The classical tracker detects spectral energy peaks above contrast thresholds, lacking the biological discriminant prior learned by a neural network. It extracts every audible acoustic ridge, including overlapping echolocation click energy and multi-channel arrivals.
- **Conclusion:** **Exact detector-level reproduction is impossible without the authors' proprietary neural network weights and CFAR code.** The stand-in tracker is permanently labeled **`[ENGINEERING SUBSTITUTION]`**.

---

## 4. Phase-1 Engineering Decisions Removed from the Default Path

To align with the research paper, the following Phase-1 choices were audited and removed from the default paper reproduction path:

| Phase-1 Mechanism | Phase-1 Justification | Paper Reality | Status in Final Pipeline |
|---|---|---|---|
| **Cross-Channel Deduplication** | Merge multi-channel hydrophone copies via lag and overlap thresholds | Paper mentions "six hydrophones" but specifies NO deduplication | **DISABLED BY DEFAULT** (`dedup.enabled: false`). All 6 channels are preserved as independent training vectors. Deduplication is isolated as an optional ablation. |
| **Harmonic Rejection** | Eliminate integer-multiple frequency ridges | Paper does not describe harmonic pruning | **DISABLED BY DEFAULT** (`reject_harmonics: false`). |
| **Round-Trip QC Preprocessing Gate** | Halt pipeline or discard images if centroid error $> 1.0\text{ bin}$ | Paper uses 0.05 threshold only to decode generated images, not as a training gate | **SEPARATED FROM PREPROCESSING**. Moved to `outputs/validation/` as an internal diagnostic check that never drops training images. |
| **P99.9 Lag Thresholds** | Enforce hydrophone acoustic transit bounds | Engineering choice | **REMOVED** from default path. |
| **Arbitrary SNR Channel Ranking** | Select single "best" channel | Biases dataset toward high-amplitude emissions | **REMOVED**. Paper neutrality preserves all channels. |

---

## 5. Summary of What Was Changed for Final 363-File Readiness

1. **Clean Directory Segregation:** Created isolated package [`paper_reproduction/`](file:///d:/ALL/whistles/paper_reproduction) dedicated to the 363-file run.
2. **Authoritative 363-File Manifest:** Built [`manifest_363.csv`](file:///d:/ALL/whistles/paper_reproduction/manifest_363.csv) containing cryptographic SHA-256 hashes, byte sizes, and format validation for all 363 files in `encounter_191_audio/` without preprocessing them.
3. **Execution Modes Separated:** Implemented distinct `--dry-run` and `--full-run` CLI modes in `run_pipeline.py`. Dry-run discovers all 363 files, validates formats, and verifies readiness with zero processing.
4. **Lineage Rewrite:** Completely updated [`PAPER_PARAMETER_LINEAGE.md`](file:///d:/ALL/whistles/paper_reproduction/PAPER_PARAMETER_LINEAGE.md) to strip all false claims of paper derivation from H1 and explicitly classify all parameters.
5. **Deleted 10-File Artifacts:** Permanently removed `paper_reproduction_test/raw_10/`, `outputs/`, `visualizations/`, `reports/`, and intermediate test files from disk.

---

## 6. Known Scientific Limitations & Claim Boundaries

1. **Whistle Detector:** Classical ridge tracking is an engineering stand-in. Whistle counts and morphological distributions reflect the stand-in tracker's contrast criteria rather than the paper's neural network [31].
2. **Channel Handling:** The paper does not specify how six hydrophones are mapped to 228,673 vectors. We preserve all 6 channels without deduplication as the most neutral realization.
3. **STFT #2 / H1 Parameters:** $F_s = 12\text{ kHz}$, $N_{\text{FFT}}=128$, $\text{hop}=64$, and Hamming-128 are plausible engineering hypotheses supported by Figure 2, but remain unconfirmed by paper text.
4. **Overlong Whistles:** Dropping whistles $> 128$ frames ($1.04\%$) is an engineering necessity to fit the $128 \times 128$ image canvas without temporal compression.

> **SCIENTIFIC CLAIM BOUNDARY:**  
> This pipeline is a **paper-faithful reconstruction** to the extent supported by the available paper and source evidence. It is **NOT** an exact reproduction because the authors' neural whistle detector weights and implementation code remain unpublished and inaccessible.
