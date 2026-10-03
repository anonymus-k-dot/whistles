# Paper Parameter Lineage & Epistemic Audit Matrix

**Authoritative Paper Reference:**  
G.-H. Park, H.-S. Jeon, J.-H. Baek, S. Huh, and H.-S. Cho, *"A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of Gaussians,"* **IEEE Access**, vol. 14, pp. 27877–27896, 2026.  
**DOI:** [10.1109/ACCESS.2026.3664765](https://doi.org/10.1109/ACCESS.2026.3664765)

---

## 1. Epistemic Classification Scheme

Every single operational parameter, mathematical equation, and architectural choice in this preprocessing pipeline is audited and assigned to one of six mutually exclusive categories:

1. **`[PAPER-SPECIFIED]`**: Explicitly stated in the main text, equations, tables, figures, or captions of Park et al. (2026).
2. **`[DERIVED FROM PAPER]`**: Mathematically or acoustically mandated by explicit paper parameters without ambiguity.
3. **`[SOURCE-VERIFIED]`**: Verified against official external source material (e.g. NOAA DCLDE 2022 HICEAS metadata).
4. **`[NOT SPECIFIED BY PAPER]`**: Topics, parameters, or policies where the research paper is completely silent.
5. **`[IMPLEMENTATION REQUIRED]`**: Concrete numerical and algorithmic choices necessary to execute the pipeline on a computer.
6. **`[ENGINEERING SUBSTITUTION]`**: Stand-in components replacing author artifacts that are unpublished or unavailable.

---

## 2. Comprehensive Parameter Lineage Matrix

### A. Raw Acoustic Recording & Data Format

| Parameter | Value | Evidence Category | Citation | Exact Paper Evidence | Implementation Realization | Confidence | Remaining Ambiguity |
|---|---|---|---|---|---|---|---|
| **Raw Sampling Rate** | $500,000\text{ Hz}$ | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881 | *"and the sampling rate is 500 kHz."* | Enforced via audio validation header check. | High (100%) | None. |
| **Channel Count** | 6 hydrophones | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881 | *"measured from six hydrophones arranged in a towed array configuration"* | 6 channels read sample-synchronously. | High (100%) | None. |
| **Recording Duration** | $\approx 60.0\text{ s}$ ($30,000,000$ samples) | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881 | *"Each file generally contains 60 seconds of underwater acoustic signals"* | Header check asserts $30,000,000$ frames. | High (100%) | None. |
| **Acoustic Dataset Source** | NOAA DCLDE 2022 HICEAS | `[PAPER-SPECIFIED]` | Sec. II.C, Ref. [29] | *"utilized the Detection, Classification, Localization and Density Estimation (DCLDE) 2022 Raw Passive Acoustic Data"* | Hawaii Cruise 1705, Encounter 191 false killer whale recordings. | High (100%) | Paper used 2,486 files across entire DCLDE false killer whale set; local dataset contains 363 files from Encounter 191. |
| **File Format & Encoding** | FLAC, 16-bit signed PCM | `[SOURCE-VERIFIED]` | NOAA DCLDE 2022 metadata | Source archive distributed as `.flac` files with `PCM_16` encoding. | `soundfile.read(dtype='int16')`. | High (100%) | None. |

---

### B. Stage 1: Audio Resampling & High-Pass Filtering

| Parameter | Value | Evidence Category | Citation | Exact Paper Evidence | Implementation Realization | Confidence | Remaining Ambiguity |
|---|---|---|---|---|---|---|---|
| **Resampling Target Sampling Rate** | $192,000\text{ Hz}$ | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881 | *"The spectrogram was generated after resampling the signal to 192 kHz"* | Resampled to $192\text{ kHz}$. | High (100%) | Numerical resampling algorithm is not stated. |
| **Resampling Ratio & Method** | Rational factor $48/125$ ($L=48, M=125$) | `[IMPLEMENTATION REQUIRED]` | Silent | Paper specifies target rate of $192\text{ kHz}$ from $500\text{ kHz}$ input; $192,000 / 500,000 = 48/125$. | Polyphase FIR filter (`scipy.signal.resample_poly`). | High (95%) | Kaiser window vs polyphase vs sinc interpolation unstated in paper. |
| **High-Pass Passband Edge** | $> 2,000\text{ Hz}$ | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881 | *"To suppress low-frequency ocean noise, a high-pass filter with a passband above 2 kHz was applied."* | Passband starts at $2,000\text{ Hz}$. | High (100%) | Filter family, order, and attenuation rate unstated in paper. |
| **High-Pass Filter Family** | Butterworth, Order 4 | `[NOT SPECIFIED BY PAPER]` | Silent | Paper specifies only "a high-pass filter with a passband above 2 kHz". | 4th-order Butterworth filter. | Medium (Hypothesis) | Chebyshev, elliptic, or FIR may have been used by authors. |
| **High-Pass Phase Response** | Zero-phase (forward-backward) | `[IMPLEMENTATION REQUIRED]` | Silent | Paper does not specify phase response. Forward-backward filtering prevents phase distortion of frequency contours. | `scipy.signal.sosfiltfilt`. | High (90%) | Forward-only filtering would introduce frequency-dependent group delay. |
| **High-Pass Design Cutoff Frequency** | $f_{\text{design}} = 1,791.439\text{ Hz}$ | `[IMPLEMENTATION REQUIRED]` | Silent | Mathematical requirement: Forward-backward filtering squares the magnitude response ($|H|^4$). To achieve $-3.01\text{ dB}$ at $2,000\text{ Hz}$, $f_{\text{design}} = 2000 \cdot (10^{3.0103/80} - 1)^{1/8} = 1791.439\text{ Hz}$. | Design cutoff parameter `f_design_hz: 1791.4389`. | High (Mathematical fact) | **CRITICAL:** $1791.439\text{ Hz}$ is an implementation detail, NOT a paper parameter. |

---

### C. Stage 2: STFT #1 (Whistle Detection Spectrogram)

| Parameter | Value | Evidence Category | Citation | Exact Paper Evidence | Implementation Realization | Confidence | Remaining Ambiguity |
|---|---|---|---|---|---|---|---|
| **STFT #1 Window Function** | Hamming window | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881 | *"applying a 2,048-point Hamming window"* | `scipy.signal.windows.hamming(2048, sym=False)`. | High (100%) | Symmetric vs periodic Hamming window unstated (difference $< 0.05\%$). |
| **STFT #1 Window Length** | 2,048 samples | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881 | *"applying a 2,048-point Hamming window"* | $N_{\text{FFT}} = 2048$. | High (100%) | None. |
| **STFT #1 Overlap Ratio** | $50\%$ | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881 | *"and using a 50% overlap."* | Overlap $= 50\%$. | High (100%) | None. |
| **STFT #1 Hop Size** | 1,024 samples | `[DERIVED FROM PAPER]` | Sec. II.C, p. 27881 | Mandated by window length 2,048 and $50\%$ overlap: $2,048 \times (1 - 0.50) = 1,024$. | $\text{hop} = 1024$. | High (100%) | None. |
| **STFT #1 Frequency Bin Spacing** | $93.75\text{ Hz}$ | `[DERIVED FROM PAPER]` | Sec. II.C, p. 27881 | Mandated by $F_s = 192,000\text{ Hz}$ and $N_{\text{FFT}} = 2,048$: $192,000 / 2,048 = 93.75\text{ Hz}$. Explicitly confirmed by text: *"frequency resolutions were approximately ... 93.75 Hz"*. | $\Delta f = 93.75\text{ Hz}$. | High (100%) | None. |
| **STFT #1 Temporal Hop Period** | $\approx 5.333\text{ ms}$ | `[DERIVED FROM PAPER]` | Sec. II.C, p. 27881 | Mandated by $\text{hop} = 1024$ and $F_s = 192,000\text{ Hz}$: $1024 / 192,000 = 5.3333\dots\text{ ms}$. Explicitly confirmed by text: *"time ... resolutions were approximately 5.3 ms"*. | $\Delta t \approx 5.333\text{ ms}$. | High (100%) | None. |

---

### D. Stage 3 & 4: Whistle Detection & Post-Processing

| Parameter | Value | Evidence Category | Citation | Exact Paper Evidence | Implementation Realization | Confidence | Remaining Ambiguity |
|---|---|---|---|---|---|---|---|
| **Whistle Detection Algorithm** | Neural network detector [31] + MCA-CFAR tracker [32],[33] | `[PAPER-SPECIFIED]` | Sec. II.C, p. 27881, Refs. [31]–[33] | *"In the second stage, a neural network-based whistle extraction method [31] was applied to detect individual whistle contours. The extraction process was combined with a post-processing algorithm [32], [33] that converts detection outputs into individual whistle tracks."* | **UNREPRODUCIBLE.** References [31]–[33] are 2-page Korean conference abstracts. Weights, architecture code, training data, and CFAR tables are unpublished. | Zero for author method. | Stand-in classical ridge tracker must be used. |
| **Actual Pipeline Detector** | Classical Ridge Tracker Stand-In | `[ENGINEERING SUBSTITUTION]` | N/A | Substituted due to unavailability of references [31]–[33]. | 3-point parabolic sub-bin peak tracking with per-bin median background subtraction. | High (as a documented substitution) | Does not possess learned biological priors; detects more low-contrast energy ridges than paper neural detector. |
| **Minimum Contour Duration** | 10 frames ($53.33\text{ ms}$) | `[IMPLEMENTATION REQUIRED]` | Silent | Paper specifies extracting individual whistle tracks but does not state the minimum duration cutoff. | Contours $< 10$ frames discarded as transient clicks. | Medium (Hypothesis) | Author cutoff could be 5, 10, or 20 frames. |
| **Harmonic Rejection** | Disabled (`reject_harmonics: false`) | `[NOT SPECIFIED BY PAPER]` | Silent | Paper does not describe harmonic rejection. | Disabled in default paper reproduction path. | High (Paper neutrality) | Whether authors filtered harmonics in [32],[33] is unknown. |

---

### E. Stage 5: Channel Handling & Deduplication

| Parameter | Value | Evidence Category | Citation | Exact Paper Evidence | Implementation Realization | Confidence | Remaining Ambiguity |
|---|---|---|---|---|---|---|---|
| **Hydrophone Array Utilization** | Process all 6 channels | `[NOT SPECIFIED BY PAPER]` | Sec. II.C, p. 27881 | *"measured from six hydrophones arranged in a towed array configuration"* | All 6 hydrophone recordings are processed. | High (Data preservation) | Paper does not specify beamforming, channel averaging, or channel ranking. |
| **Cross-Channel Deduplication** | Disabled (`dedup.enabled: false`) | `[NOT SPECIFIED BY PAPER]` | Silent | The paper does NOT mention cross-channel deduplication anywhere in text or figures. | Disabled in default paper reproduction path. Multi-channel detections preserved as individual vectors. | High (Paper neutrality) | Whether authors used deduplication or single-channel selection is unstated. |

---

### F. Stage 6: Time-Frequency Vectors & Support Screening

| Parameter | Value | Evidence Category | Citation | Exact Paper Evidence | Implementation Realization | Confidence | Remaining Ambiguity |
|---|---|---|---|---|---|---|---|
| **Whistle Time-Frequency Vector Representation** | Dual: discrete $(t_k, f_k)$ and physical $(t_s, f_{\text{Hz}})$ | `[PAPER-SPECIFIED]` | Sec. II.B, Fig. 2(c), Sec. II.C | *"the estimated time-frequency vector derived from the binary whistle image"* and *"A total of 228,673 whistle time-frequency vectors were obtained"* | Dual-format dictionary and CSV storage. | High (100%) | Vector interpolation points unstated. |
| **Center Frequency Removal** | $f_{\text{baseband}}(t) = f(t) - f_c$ | `[PAPER-SPECIFIED]` | Sec. II.B, Fig. 3, App. C | *"depicting the contour of a baseband whistle signal with the center frequency removed as in Fig. 1"* and *"The center frequency was excluded ... as it was normalized during preprocessing"* | Baseband trajectory shifted by $f_c$. | High (100%) | Mathematical definition of $f_c$ is unstated. |
| **Center Frequency Formula** | Midrange: $f_c = (f_{\min} + f_{\max})/2$ | `[NOT SPECIFIED BY PAPER]` | Silent | Paper does not state formula for $f_c$. Midrange centers the contour symmetrically within $\pm 6\text{ kHz}$ baseband. | $f_c = (f_{\min} + f_{\max}) / 2$. | Medium (Hypothesis) | Mean or energy-weighted centroid could have been used. |
| **Handling of Whistles $> 128$ Frames** | Drop and log to CSV ($1.04\%$) | `[NOT SPECIFIED BY PAPER]` | Silent | Paper uses fixed $128 \times 128$ canvas. Paper is silent on whether $>128$-frame whistles were cropped, rescaled, or dropped. | Overlong whistles dropped and logged to `overlong_contours.csv`. | Medium (Hypothesis) | Truncation, temporal pooling, or multi-patching are alternative hypotheses. |

---

### G. Stage 7 & 8: NFM Synthesis & STFT #2 (Hypothesis H1 Audit)

| Parameter | Value | Evidence Category | Citation | Exact Paper Evidence | Implementation Realization | Confidence | Remaining Ambiguity |
|---|---|---|---|---|---|---|---|
| **Constant-Amplitude NFM Waveform** | $s(t) = \cos\left(2\pi \int \Delta f(\tau) d\tau\right)$ | `[PAPER-SPECIFIED]` | Sec. II.B, Fig. 2(d), Sec. II.C | *"Each vector was then used to generate a constant-amplitude NFM baseband signal"* | Numerical phase integration and cosine generation. | High (100%) | Numerical integration method unstated. |
| **Baseband Sampling Rate** | $F_s = 12,000\text{ Hz}$ | `[NOT SPECIFIED BY PAPER]` | Silent (Consistent with Fig. 2) | **AUDIT CORRECTION:** The paper NEVER states $F_s = 12\text{ kHz}$ in text. Inferred from Fig. 2 frequency axis spanning $[-6\text{ kHz}, +6\text{ kHz}]$. | Synthesized at $F_s = 12,000\text{ Hz}$. | Medium (Hypothesis H1) | Authors may have used continuous analytic signal or complex baseband. |
| **STFT #2 FFT Size** | $N_{\text{FFT}} = 128$ | `[NOT SPECIFIED BY PAPER]` | Silent (Consistent with 128 rows) | **AUDIT CORRECTION:** The paper NEVER states $N_{\text{FFT}} = 128$ for STFT #2. Inferred because image height is 128 rows. | $N_{\text{FFT}} = 128$. | Medium (Hypothesis H1) | Resizing/interpolation from a larger FFT is an alternative hypothesis. |
| **STFT #2 Hop Size** | $\text{hop} = 64$ samples ($50\%$ overlap) | `[NOT SPECIFIED BY PAPER]` | Silent | **AUDIT CORRECTION:** The paper NEVER states hop=64 for STFT #2. Inferred because $64 / 12,000 = 5.333\text{ ms}$, matching STFT #1. | $\text{hop} = 64$. | Medium (Hypothesis H1) | Hop size could be uncoupled from STFT #1. |
| **STFT #2 Window Function** | Hamming window, length 128 | `[NOT SPECIFIED BY PAPER]` | Silent | **AUDIT CORRECTION:** Paper mentions Hamming only for STFT #1. Inferred for STFT #2. | 128-point Hamming window. | Medium (Hypothesis H1) | Rectangular, Hanning, or Blackman unstated. |
| **DC Component Mapping** | Row 64 ($0\text{ Hz}$) | `[NOT SPECIFIED BY PAPER]` | Fig. 2, Fig. 8 | Fig. 2 and 8 center frequency distributions peak at center row. Inferred as row 64. | `fftshift` centered at row 64. | Medium (Hypothesis H1) | Row 63 vs 64 index convention unstated. |
| **Horizontal Image Centering** | `col_start = (128 - L) // 2` | `[NOT SPECIFIED BY PAPER]` | Fig. 1, Fig. 2 | Fig. 1 and 2 display whistles centered horizontally in the 128-column frame. | Centered temporally with zero padding outside duration. | Medium (Hypothesis H1) | Left-alignment with right-padding is an alternative hypothesis. |
| **Image Intensity Normalization** | Constant window-sum division | `[IMPLEMENTATION REQUIRED]` | Fig. 1 caption | *"normalized to have intensities between 0 and 1, with the whistle regions having values close to 1."* | Peak single-tone response normalized to 1.0: $I = \|X\| / (\sum w / 2)$. | High (95%) | Per-image min-max vs global peak normalization unstated. |
| **Image Grayscale Format** | Continuous float32 in $[0.0, 1.0]$ | `[PAPER-SPECIFIED]` | Fig. 1 caption, Sec. I.D | *"The actual data used in this study are normalized to have intensities between 0 and 1 ... the images are grayscale"* | Continuous float32 preserved losslessly as `.npy`. | High (100%) | None. Training images are strictly continuous. |
| **Contour Decoding Threshold** | 0.05 intensity threshold | `[PAPER-SPECIFIED]` | Fig. 2(b) | *"by applying a simple threshold of 0.05 to the image, the whistle contour shown in the second panel can be obtained"* | Used **ONLY** for internal reconstruction validation, **NEVER** as a preprocessing drop gate. | High (100%) | Reserved for decoding generated images. |
| **Evaluation Binarization Threshold** | 0.10 intensity threshold | `[PAPER-SPECIFIED]` | Sec. IV.A | Reserved strictly for post-generation FID1 evaluation. | Not used in preprocessing pipeline. | High (100%) | Evaluated on downstream generative models only. |

---

## 3. Summary of Epistemic Corrections from Previous Reporting

1. **Reclassification of Hypothesis H1:** Baseband sampling rate ($12\text{ kHz}$), STFT #2 $N_{\text{FFT}}=128$, $\text{hop}=64$, Hamming window, DC row 64, and temporal centering were previously mislabeled as `[DERIVED FROM PAPER]`. They are now correctly classified as **`[NOT SPECIFIED BY PAPER] / [IMPLEMENTATION REQUIRED] (Hypothesis H1)`**.
2. **Reclassification of High-Pass Cutoff:** The design cutoff $1791.439\text{ Hz}$ is classified strictly as `[IMPLEMENTATION REQUIRED]`, not a paper specification.
3. **Reclassification of Center Frequency:** Midrange formula $(f_{\min} + f_{\max})/2$ is classified as `[NOT SPECIFIED BY PAPER]`.
4. **Reclassification of Intensity Normalization:** Constant window-sum scaling is classified as `[IMPLEMENTATION REQUIRED]`.
5. **Detector Lineage Permanently Locked:** Classical ridge tracking is permanently classified as **`[ENGINEERING SUBSTITUTION]`**. Exact reproduction is impossible without the authors' unpublished neural network weights [31].
