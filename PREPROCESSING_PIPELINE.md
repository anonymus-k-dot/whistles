# Preprocessing Pipeline: Cetacean Whistle Images for VAE-GMM

Reference: G.-H. Park et al., "A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of Gaussians," IEEE Access, vol. 14, 2026.

## 0. Evidence standard and scope

Every decision has two independent annotations, because a point can be unresolved in the paper while our working value is still a deliberate choice.

| Field | Values |
|---|---|
| **Paper status** | **[Paper]** stated in the paper. **[Inferred]** not stated; derived from the paper's text, numbers, or figures (Section 2). **[Unresolved]** not stated, and the available evidence cannot settle it. **n/a** an engineering parameter the paper does not concern. |
| **Our side** | **[Implementation Choice]** a deliberate design decision. Where the paper is [Unresolved] it is the *safest neutral option* we could identify: no data-dependent selection, information-preserving, reversible, or a standard default. It is never a claim about what the paper did. **[Assumption]** a provisional, testable belief about the data or the paper's method (H1, channel synchronization, the species in the files). It may be false. A dash means nothing was chosen. |

**H1** is the hypothesis that STFT #2 and the NFM synthesis use a 12 kHz baseband, a 128-point Hamming window, NFFT 128, and hop 64 (D10). Its parameters are typed in exactly one place, the `h1` block of `config.yaml`. Every value that depends on H1 is **derived** by `h1_image.py` into `derived_h1.yaml` and is never typed by hand (Section 2, D12 to D14).

### Dataset mismatch (D22)

| | Paper | This project |
|---|---|---|
| Raw files | 2,486 [Paper] | **363** (14.6% of the paper's) |
| Whistle vectors | 228,673 [Paper] (about 92 per file) | not yet known; about 33,400 if the yield matched (an estimate, not a prediction) |
| Acquisition | 500 kHz, six hydrophones, about 60 s | all 363 files match (inventory); 6.05 h |
| File selection | files "containing whistles from the false killer whale" [Paper] | **unknown**; species presence is unverified |
| Same files as the paper | n/a | **unknown**; the paper lists no file identifiers |

Consequences: dataset size, population, and preprocessing all differ, so no metric produced here is comparable with the paper's tables. The paper's K = 10,000 would give about three latents per component here (D18). Nothing in this project may be described as the paper's dataset or as a reduced replica of it until the file selection and species are verified.

**Verified for this document:** the paper text; Figs. 2, 8 and 10 re-examined at 220 dpi (read by eye); the inventory of the 363 files; the filter-response arithmetic; and the H1 simulations in `h1_image.py` (all tests pass). **Not verified:** any third-party implementation, the DCLDE 2022 documentation [29] (not in the project, so array geometry, channel synchronization, and file metadata are unknown), and references [31] to [33] (not available).

**Implementation status:** `stage1_audio.py` (Stage 1) and `h1_image.py` (H1 synthesis, derivation, and staleness guard) are implemented and tested. Stages 2 to 10 are otherwise specifications.

---

## 1. Decision register

| ID | Decision | Paper status | Our side | Working value | H1-dependent |
|---|---|---|---|---|---|
| D1 | Channel handling | [Unresolved] | [Implementation Choice] | Detect on all six channels, deduplicate | no |
| D2 | Resampling | [Paper]: 192 kHz; method [Unresolved] | [Implementation Choice]: method | Polyphase 48/125 | no |
| D3a | High-pass edge | [Paper]: 2 kHz | none | 2 kHz | no |
| D3b | Realization of the edge: meaning of "passband", filter family, order, phase, order relative to STFT | [Unresolved] | [Implementation Choice] | -3 dB at 2 kHz after zero-phase filtering; Butterworth n = 4; design cutoff 1791.4 Hz (derived in code; **not a paper parameter**) | no |
| D4 | STFT #1 | [Paper]: 2,048 Hamming, 50% overlap; [Inferred]: NFFT, hop | none | 93.75 Hz bins, 5.333 ms hop | no |
| D5 | STFT #1 normalization | [Unresolved] | [Implementation Choice] | Detector-side per-bin median subtraction, ablated | no |
| D6 | Contour detector and post-processing | [Unresolved]: [31], [32], [33] named, not reproducible | [Implementation Choice]: stand-in | Classical ridge tracker | no |
| D7 | Cross-channel deduplication | [Unresolved]: not mentioned | [Implementation Choice]: our independent engineering choice | Enabled, measured tolerances | no |
| D8 | Which duplicate to keep | [Unresolved] | [Implementation Choice] | Seeded random | no |
| D9 | Whistles longer than 128 frames | [Unresolved]: width 128 is [Paper] | [Implementation Choice] | Drop and report | yes |
| D10 | NFM rate; STFT #2 window, NFFT, hop | [Unresolved] | [Assumption]: H1 | 12 kHz, Hamming, 128, 128, 64 | defines H1 |
| D11 | Center-frequency definition | [Unresolved] | [Implementation Choice] | Midrange | no |
| D12 | Frequency centering in the image | [Paper]: centered; row [Inferred] | none | Row 64 (derived) | **yes** |
| D13 | Time alignment in the image | [Inferred]: centered (figures) | [Implementation Choice]: start rule | Centered; start = (128 − L) // 2 | rule: **yes** |
| D14 | Padding and resizing | [Unresolved] | [Implementation Choice] | Zero time-padding; no resize | **yes** |
| D15 | Intensity normalization | [Paper]: range [0, 1]; rule [Unresolved] | [Implementation Choice] | Constant (window-sum) | **yes** |
| D16 | Continuous training images; thresholds | [Paper] | none | No threshold on training data; 0.05 decode/QC; 0.1 FID₁ | no |
| D17 | QC failure policy | n/a | [Implementation Choice] | Fatal integrity gates; quarantined fidelity failures | tolerances: **yes** |
| D18 | Data protocols | [Inferred]: no split described | [Implementation Choice] | Protocol A (paper-aligned), Protocol B (leakage-controlled) | no |
| D19 | Rotation augmentation | [Paper]: 0/90/180/270°, equal probability; details [Unresolved] | [Implementation Choice]: how it is applied | Training-time only (Section 2, D19) | no |
| D20 | Time coordinate across channels | n/a | [Implementation Choice]; synchronization [Assumption] | One file-relative coordinate for all channels | no |
| D21 | Reproducibility controls | n/a | [Implementation Choice] | Section 9 | no |
| D22 | Dataset population | [Paper]: 2,486 files | [Assumption]: species and selection unverified | 363 files, not comparable | no |

---

## 2. Evidence and reasoning

### D1. Channel handling

**Paper: [Unresolved]. Our side: [Implementation Choice].** The paper says each file holds about 60 s from six hydrophones in a towed array (Sec. II.C) and gives no channel procedure, so every option, including channel 0, is an assumption about the paper's method.
- A single channel discards most of the data and ties quality to one hydrophone.
- Averaging channels is invalid for a towed array unless beamformed, because arrival delays differ across elements.
- Safest neutral option: process every channel independently, with no data-dependent channel selection, and deduplicate afterwards (D7).

The paper's count of 228,673 vectors from 2,486 files does not reveal whether it came from one channel or several.

### D3. High-pass edge and its realization

**Paper: [Paper] only for "a high-pass filter with a passband above 2 kHz". Filter family, order, phase, the meaning of "passband", and the order relative to the STFT are [Unresolved]. Our side: [Implementation Choice].**

**The 2 kHz edge is the only paper-stated parameter. The design cutoff 1791.4 Hz is our implementation realization of that edge. It is not a paper parameter and must never be reported as one.** `stage1_audio.design_cutoff_hz()` computes it from the edge, so it cannot drift from the config.

Zero-phase filtering applies the filter twice, which squares the amplitude response. For a 4th-order Butterworth at 192 kHz:

| Design cutoff | Response at 2,000 Hz | -3 dB frequency |
|---|---|---|
| 2,000 Hz (`design_cutoff_equals_edge`) | -6.02 dB | 2,233 Hz |
| 2,300 Hz | -12.17 dB | about 2,568 Hz |
| **1,791.4 Hz** (`minus3db_at_passband_edge`) | **-3.01 dB** | 2,000 Hz |

Safest neutral option: realize "passband above 2 kHz" with a -3 dB edge at 2 kHz, the least lossy reading, so the stated passband is not attenuated more than the conventional edge definition implies. The paper lists the filter after describing the STFT, so filtering the waveform first is also our choice, not a stated order.

### D4. STFT #1

**Paper: [Paper] for the Hamming window, 2,048-point length, and 50% overlap; [Inferred] for NFFT and hop.** 192,000 / 2,048 = 93.75 Hz and 1,024 / 192,000 = 5.333 ms match the paper's quoted 93.75 Hz and about 5.3 ms. The hop follows from the stated overlap, and NFFT follows from the quoted resolution; neither is stated directly.

### D5. STFT #1 normalization

**Paper: [Unresolved]. Our side: [Implementation Choice].** The paper specifies no normalization, so per-bin median normalization is not justified by the paper.
- The title of reference [32] ("A design of MCA-CFAR detector...") indicates a CFAR-type detector, which estimates local background by construction. Its content is not available to us, so we cannot say what normalization the authors' pipeline applied.
- A per-bin median over a 60 s file suppresses stationary tonal noise and spectral tilt, which helps a ridge tracker. It can bias results where a bin is occupied more than half the file, and it changes what "SNR" means.
- Safest neutral option: keep it detector-side (D6), never present it as a paper step, and choose among `none`, `per_bin_median_subtraction_db`, and `sliding_local_median_db` by recall and precision on the annotated development files (Section 8).

### D6. Detector and post-processing

**Paper: [Unresolved]: a neural detector [31] plus post-processing [32], [33] are named but not reproducible from the paper. Our side: [Implementation Choice].** The detector defines the distribution p(x) the VAE-GMM learns, and whether our stand-in produces a similar distribution is untested. Do not describe the dataset as paper-exact. The 2 to 20 kHz detection band (high-pass edge to the Fig. 3 display range) and sub-bin parabolic interpolation are our choices.

### D7. Cross-channel deduplication

**Paper: [Unresolved]; the paper does not mention deduplication or per-channel handling. Our side: [Implementation Choice]; deduplication is our independent engineering choice and is not required or implied by the paper.**

Eq. 1 is the paper's modeling assumption that whistles are i.i.d. draws from p(x). It is a statement about the model and does not, by itself, show that multi-channel duplicates must be removed. Our motivation is practical:
- The same emission reaches different hydrophones with a delay bounded by aperture divided by sound speed, so one emission can appear as several near-identical contours in one file.
- Keeping every copy gives that emission extra weight in the training distribution and in the reference set used by distribution metrics.

Deduplication can also remove genuine samples if the matching rule is too loose, and it changes the dataset relative to the paper's (whose handling is unknown). A variant built with `dedup.enabled: false` is therefore compared against the main dataset (Section 8).

**Matching rule (all conditions required), within one file, in the common time coordinate (D20).**
1. |lag| ≤ `max_lag_s`, measured, not assumed: cross-correlate band-passed channel pairs around high-contrast detections in the development files, take the 99.9th percentile of |lag|, and add one frame (5.333 ms) for quantization.
2. Temporal overlap of at least 50% of the shorter track after lag alignment.
3. Mean absolute frequency difference over the overlap of at most one STFT #1 bin (93.75 Hz).

Overlap alone would merge two animals whistling simultaneously; shape alone would merge stereotyped whistles repeated at different times. A match group may hold at most one track per channel. If two tracks from one channel match the same group, nothing is merged and the event is counted as ambiguous.

**Can be claimed.** We deduplicate by our own rule to limit repeated records of the same emission; dataset counts are not expected to equal the paper's.

**Cannot be claimed.** That the paper deduplicated, or did not; that Eq. 1 requires deduplication; that the resulting samples are independent (whistles from one animal, group, or encounter remain correlated, which the split protocol in Section 5 addresses).

### D8. Which duplicate to keep

**Paper: [Unresolved]. Our side: [Implementation Choice].** The paper defines no selection rule, so an SNR-based rule is not supported. SNR is undefined in the paper, and any definition depends on the unresolved normalization in D5. Choosing the best copy would systematically favor cleaner, better-tracked contours. Safest neutral option: seeded random choice among matched copies using the per-file RNG (Section 9). A descriptive contrast value is stored as metadata only. An SNR-based rule may be run only as a labeled sensitivity variant after D5 is settled.

### D9. Long whistles

**Paper: [Unresolved] (the 128-column image width is [Paper]). Our side: [Implementation Choice].** The paper does not say what happened to whistles that exceed the width.
- Whistles filling almost the full width exist: the input whistle in Fig. 10(c) spans roughly columns 4 to 128, and the length histogram for component 1,253 in Fig. 8(c) reaches about 125 frames (read from the figures).
- Dropping, truncating, splitting, and time-rescaling are all compatible with what is shown; the right edge of Fig. 10(c) touching the border is compatible with truncation but does not prove it.

Safest neutral option: drop whistles longer than 128 frames, count them, and keep their contours separately, so nothing is fabricated and the choice is reversible. Truncation fabricates start and end points and corrupts the start frequency, end frequency, and length features used for conditional generation; time-scaling changes durations. Dropping is still a selection effect: the model's support is limited to 128 frames (0.683 s under H1), and any statement about generated length must say so. Report the dropped fraction and the length histogram beyond 128. If the fraction exceeds the pre-registered 5% trigger, report it as a limitation; do not change the policy after seeing results without recording the change.

### D10. Vector-to-image synthesis (hypothesis H1)

**Paper: [Paper] for the method (constant-amplitude NFM baseband signal, its spectrogram, a 128 × 128 grayscale image); [Unresolved] for baseband rate, window, NFFT, hop, and any resizing. Our side: [Assumption] (H1).**

H1: baseband rate 12 kHz, Hamming window 128, NFFT 128, hop 64, two-sided complex spectrum. Evidence consistent with H1:
- Fig. 2 frequency axes: the lower limit is labeled -6 kHz; the upper limit is about +6 kHz by eye (the top tick label is +4). 128 × 93.75 Hz = 12 kHz.
- Fig. 2 time axes extend slightly beyond the ±0.3 tick marks (about ±0.34 s by eye). 128 frames × 5.333 ms = 0.683 s.
- The staircase in Fig. 2(b) has steps of about 0.09 kHz and about 0.045 s, which matches one 93.75 Hz bin per roughly 9 frames on the drawn slope (by eye).
- Simulated thresholded band thickness at 0.05 for a slow contour: Hamming 3 bins, Hann 4, rectangular 9. Fig. 2(b)'s band is about 3 bins thick by eye, which is consistent with the Hamming result and not with the rectangular one; the Hann result cannot be excluded at this reading precision.

**Caveats.** Fig. 2(a) is described as an image *generated by the VAE-GMM*, not a training image. Applying its physical-unit axes to the training-image synthesis assumes the decoder output shares the training grid, which is plausible (both are 128 × 128) but not stated. Each observation also has alternatives (for example, a larger spectrogram resized to 128 × 128 would give the same axis). H1 fits every figure we could test; it is not established.

**Falsification test.** If measured features of the paper's figures (thickness, onset/offset extension, centering) differ from H1 simulations by more than one bin or one frame, reject H1.

### D11. Center-frequency definition

**Paper: [Unresolved]: the paper says center frequency is normalized to the image center (Sec. IV.D.4; Appendix C) but not how it is computed. Our side: [Implementation Choice].** In all four Fig. 8 panels, the minimum- and maximum-frequency histograms put the midrange (fmin + fmax)/2 at about 65 bins, and the Center Freq histograms sit at 64 to 66 (by eye). The midrange is consistent with all four; the mean cannot be tested from the figures. Safest neutral option: the midrange, which also guarantees symmetric bounds around the image center.

### D12 to D14. Image geometry (conditional on H1, derived automatically)

**Every statement in this subsection assumes H1**: a two-sided complex baseband spectrum of 128 bins, fftshifted, with one image column per STFT #1 frame.

**Automatic re-derivation.** `h1_image.py derive` recomputes, from the `h1` block and the inputs listed in `fingerprint()`, every H1-dependent value, and writes them to `derived_h1.yaml` with a fingerprint:

| Derived value | Rule | Current value |
|---|---|---|
| Center row (0-based) | nfft // 2 (DC bin after fftshift) | 64 |
| Support limit \|f − center\| | nfft // 2 − `baseband_margin_bins` | 62 bins |
| QC centroid tolerance | ceil(`centroid_safety_factor` × simulated max centroid error) | 1 bin |
| QC length tolerance | simulated max length extension + `length_margin_frames` | 3 frames |
| Columns per STFT #1 frame | (h1.hop / baseband_fs) / (stft1.hop / fs_out) | 1 |

Stages 7 to 9 call `assert_fresh()`, which stops the run if `derived_h1.yaml` is missing or was computed from different inputs (any change to the `h1` block, image size, STFT #1 window or hop, resampling rate, decode threshold, normalization rule, margin, or tolerance rule; tested in `test_h1.py`). It also rejects an H1 whose column-to-frame mapping breaks (columns per frame ≠ 1, non-integer samples per frame, rows ≠ NFFT, or image width ≠ `max_length_frames`), because the length semantics and the start rule below would then be invalid.

**What is not automatic.** Comparing the synthesized images with the paper's figures (D10 falsification test) and re-reading the figure-based centering evidence remain manual and must be repeated if H1 changes (Section 8).

- **D12, frequency.** Under H1, baseband 0 Hz falls on row 64 (0-based; bin 65 in 1-based MATLAB indexing), matching the 64 to 66 center histograms. Center removal then centers the contour automatically. Paper status: centering is [Paper]; the row is [Inferred].
- **D13, time.** The figure observation does not depend on H1: measured by eye, contour time centers are about 66, 65, and 67.5 for Fig. 8(a) to (c), and about 66, 65.5, and 66 for Fig. 10(a) to (c), although lengths range from about 32 to about 124 frames. The images appear time-centered to within about ±2 columns [Inferred]. The start rule (128 − L) // 2, with L in STFT #1 frames equal to columns, is H1-dependent and is our choice. The floor puts odd lengths half a column left of exact center. Figs. 9, 13, and 14 plot contours starting near time 0; these are plots of extracted vectors, the likely explanation but not proven.
- **D14, padding and resizing.** Under H1 the frequency axis is 128 bins by construction, time outside the whistle is a zero signal (exactly zero magnitude), and no resizing is needed. The paper's phrase "further transformed into a 128 × 128 grayscale image" does not exclude a resize or crop, so this remains [Unresolved].
- **Support limit.** The complex-baseband spectrum wraps at ±nfft/2 bins, so contours near the edge alias. Whistles with |f − center| above 62 bins are out of support: dropped and counted, like D9.
- **Decoded-length bias (H1).** Thresholding a synthesized image at 0.05 extends the decoded contour by 2 frames in total (the window overlaps the whistle's edges). Features decoded from images by thresholding carry this bias; the paper's own decoded features may as well, but its decoding details are unknown.

### D15. Intensity normalization

**Paper: [Paper] for the range [0, 1] with whistle regions "close to 1" (Fig. 1 caption); [Unresolved] for per-image versus global. Our side: [Implementation Choice]. H1-dependent (window sum).**

The caption's wording is consistent with constant window-sum normalization. It is equally consistent with per-image maximum normalization, so the caption does not discriminate between them.

Simulated peak magnitude under constant normalization (H1, unit amplitude, division by the window sum; all contours 30 frames long, so the column is comparable across rows):

| Sweep rate (bins/frame) | Peak |
|---|---|
| 0 | 1.00 |
| 0.5 | 0.94 |
| 1 | 0.84 |
| 2 | 0.84 |
| 3 | 0.81 |
| 4 | 0.73 |

Safest neutral option: constant normalization (window sum × amplitude). It guarantees values in [0, 1] by the triangle inequality, uses no data-dependent statistic, and preserves sweep-rate information through peak height. Per-image maximum forces every image to reach exactly 1 and erases that information. `alt_normalization: per_image_max` is available as a switch; only one rule may be used across the whole dataset.

### D16. Continuous grayscale and thresholds

**Paper: [Paper].** Training images are continuous grayscale in [0, 1] and are never thresholded or quantized. The threshold 0.05 is used only to decode generated images into contours (Fig. 2) and for round-trip QC. The threshold 0.1 is used only for FID₁. Store images as float32 with lossless compression.

### D18. Data protocols

See Section 5. **Paper: [Inferred]**: the paper describes no train/validation/test split and reports distances between synthesized whistles and "the training data".

### D19. Rotation augmentation

**Paper: [Paper]. Appendix A.B states that a data-augmentation technique was applied during VAE training in which each image was randomly rotated by 0°, 90°, 180°, or 270° with equal probability. The paper's stated purpose is to prevent the VAE from over-focusing on horizontal line features and to encourage balanced learning of directional features. This is a paper-specified step and is kept.** It is a training-time operation, not preprocessing.

**Not stated in the paper ([Unresolved]), with our [Implementation Choice] for each:**

| Open point | Our choice | Reason it is the safest option |
|---|---|---|
| Is the rotated image also the reconstruction target? | Yes: one rotated tensor is both encoder input and target | Reconstructing an unrotated target from a rotated input would require learning the inverse rotation, which the paper does not describe |
| Does the critic's real sample use the same rotation? | Yes | The VAE output is in the rotated domain; the critic should compare like with like |
| Is the angle redrawn every iteration? | Yes | "Randomly rotated" during training; no fixed per-image angle is stated |
| Are rotated images stored? | No | Keeps the stored dataset un-augmented and the dataset hash independent of training randomness |
| Latents used to fit the GMM | Un-augmented encoder means | The paper says only "latent mean outputs of encoder"; this keeps the fitted prior on the upright whistle distribution |

**Consequence (our analysis, not a paper statement).** A 90° rotation swaps the time and frequency axes, and 180° reverses both, so the VAE also learns rotated patterns that a GMM fit on upright latents does not model. `training_time.ablation_without_rotation` allows a training run without rotation as a separate, labeled variant. That variant is not paper-aligned and must never be mixed into paper-aligned results.

### D20. Common time coordinate

**Paper: n/a. Our side: [Implementation Choice]; channel synchronization is [Assumption].** Before any cross-channel operation, every contour from every channel of a file is expressed in one file-relative coordinate: seconds from the first sample of the file, with frame k centered at t_k = (k·hop + nfft/2) / fs_out. All channels share the same frame grid (same sampling, framing, and origin), and a file whose channels disagree on frame count is a fatal error. The polyphase resampler and zero-phase filter apply identical operations to every channel, so they add no channel-dependent delay. Channels are assumed sample-synchronous: an [Assumption] to be tested against the dataset documentation and the measured lags. Track-relative time cannot support lag or overlap computation and is not used before deduplication.

### D22. Dataset population

**Paper: [Paper] (2,486 files containing false killer whale whistles; 228,673 vectors). Our side: [Assumption] that the local files are suitable; species and selection unverified.** See the table in Section 0. Before any scientific claim about false killer whales, verify that the local files contain such whistles (from metadata or manual inspection of the golden set).

---

## 3. Pipeline stages

```
RAW 6-CHANNEL FLAC (500 kHz)
  Stage 0   Configuration, H1 derivation (h1_image.py derive)
  Stage 1   Load, per-channel QC, resample 500 -> 192 kHz, high-pass        [implemented]
  Stage 2   STFT #1 (common frame grid)
  Stage 3   Whistle detection (per channel)
  Stage 4   Contour post-processing
  Stage 5   Cross-channel deduplication (common file-relative time)
  Stage 6   Time-frequency vectors
  Stage 7   Center removal -> constant-amplitude NFM baseband waveform       [assert_fresh]
  Stage 8   STFT #2 -> 128 x 128 continuous grayscale image in [0, 1]        [assert_fresh; synth in h1_image.py]
  Stage 9   QC (integrity gates, round-trip fidelity)                        [assert_fresh]
  Stage 10  Dataset assembly
VAE INPUT
```

**Stage 0.** All parameters are in Section 7. Run `python h1_image.py derive --config config.yaml` after any change to an H1 input. Save the config, its hash, the environment record (Section 9), and the seeds with every output.

**Stage 1 (implemented).** Validate the header (500 kHz, 6 channels); read int16; per-channel QC (dead if RMS below 1 count, skipped; clipped if above 0.1% of samples at full scale, flagged); scale to float64; polyphase resample 48/125; Butterworth n = 4 zero-phase high-pass with the design cutoff derived from the 2 kHz edge (D3); return float32. Nominal file duration is not enforced here; the inventory screens it against `duration_warn_range_s`. Synthetic test with the current config: output length exact; 500 Hz tone attenuated 88.7 dB; 6 to 9 kHz sweep gain +0.01 dB; realized response -3.01 dB at 2 kHz; dead and clipped channels detected. About 7.8 s per file in the test environment.

**Stage 2.** STFT with a 2,048-point Hamming window and hop 1,024. Every channel uses the identical frame grid (D20). Use `rfft` on strided frames.

**Stage 3.** Detector stand-in per D5 and D6. Develop and tune on the annotated files before any full run (Section 8).

**Stage 4.** Post-processing rules, all logged choices: minimum duration, maximum bridged gap, harmonic rejection, one track per whistle. Count removals per rule.

**Stage 5.** Deduplication per D7, D8 and D20. Contours are first mapped to the common file-relative time coordinate; all lag and overlap computations use it.

**Stage 6.** Per whistle: array of (frame index k, file-relative time, frequency in Hz as float) with file, channel, and contrast metadata. Contours are stored; spectrograms are not.

**Stage 7.** Call `assert_fresh()`. Drop out-of-support whistles (length above 128 frames, or |f − center| above the derived limit) with reason codes. Subtract the midrange center. Convert to STFT #2 bins, interpolate linearly to the baseband rate, integrate to phase with a float64 cumulative sum, and form exp(jφ).

**Stage 8.** Use `h1_image.synth_image()`: center the signal in the 128-frame canvas with start frame (128 − L) // 2, compute the complex STFT with the `h1` parameters, take the magnitude, apply fftshift along frequency, and normalize per D15. Output: float32, 128 × 128, continuous in [0, 1], with no threshold, resize, or quantization.

**Stage 9.** See Section 4.

**Stage 10.** Save images (float32, lossless compression, memory-mapped at training), contour vectors, the seven-feature table (minimum, maximum, start, and end frequency; bandwidth; length; center frequency), and the funnel report. Assign splits per Section 5. No normalization statistic is computed from data, so none can cross a split.

---

## 4. QC policy

The aim is that no filter changes the dataset without being counted and inspected.

- **Integrity gates (fatal).** Non-finite values, values outside [0, 1], or wrong shape indicate a bug. The run halts; nothing is excluded to work around it.
- **Fidelity gate (quarantine, never silent deletion).** Decode the image at 0.05, re-extract the contour, and compare to the input vector. The tolerances are H1-derived (1 bin centroid error and 3 frames of decoded length; simulated max centroid error 0.27 bin, simulated length extension 2 frames). Failing items go to `quarantine/` with reason codes.
- **Bias check.** Compare the seven-feature distributions of passed and quarantined items. A material difference means QC is biasing the dataset; investigate the cause instead of dropping items.
- **Halt rule.** If the fidelity failure rate exceeds 1% (pre-registered), stop and investigate. Tolerances are not loosened after seeing failures without recording the change and its reason.
- **Mandatory funnel report.** Counts at every step, per file and overall, with reasons: detections per channel → after post-processing (per rule) → after deduplication (merged, ambiguous) → out-of-support (length, bandwidth) → fidelity failures → final. The dropped-for-length fraction is compared with the 5% trigger.
- **Sensitivity runs.** Rerun detection thresholds at ±20% and compare feature distributions. Selection effects at the detector are larger than those of the later QC.

---

## 5. Data protocols

The two protocols must never be mixed in one results table.

### Protocol A: paper-aligned evaluation

**Paper: [Inferred].** The paper describes no held-out set and reports distances between synthesized whistles and "the training data"; that the VAE and GMM were trained on all 228,673 whistles is inferred, not stated. Protocol A aligns the *evaluation design* with that reading: train on all whistles from all files and compute generation metrics against the training set. It measures fit to the training distribution, not generalization, and cannot detect memorization.

**It is not a paper-exact reproduction.** The detector, post-processing, deduplication, channel handling, image-synthesis parameters, and normalization are unresolved or stand-ins (D1 to D15), and the dataset is 14.6% of the paper's file count with unverified population (D22). Its numbers are not expected to match the paper's.

### Protocol B: leakage-controlled evaluation (our own; not in the paper)

- **Unit.** Contiguous time blocks of files with at least one guard file between blocks, because adjacent 60 s files can contain the same animals and acoustic conditions. If encounter or recording identifiers exist, split by those. A random file-level split is the minimum fallback. All channels and duplicates of an emission stay together, since they come from one file.
- **Assignment timing.** Blocks are assigned at the start from file metadata, before any detection, so no detector output influences the split.
- **Fractions.** 80 / 10 / 10 is an arbitrary convention.
- **Roles.** The validation block selects K, β, and the GMM covariance type. The test block is evaluated once.
- **Fit on train only.** The VAE, GMM, and any clustering indices.
- **Metrics.** FID₁, FID₂, precision, recall, and F1 against held-out images, with bootstrap confidence intervals. With a few thousand held-out images, FID on 2,048-dimensional features is noisy and biased. Also report the nearest-training-neighbor distance of generated and held-out samples as a memorization check.

### Consequence of dataset size

With about 33,400 whistles expected (363 files × the paper's 91.98 per file; an estimate to be replaced by the measured count), the paper's K = 10,000 would give about three latents per component. Use K in the range 500 to 2,000 and diagonal covariance, and treat all numbers as not comparable with the paper's tables.

---

## 6. Claim boundaries

**Permitted.** "The pipeline follows the paper's stated steps: 192 kHz resampling, a high-pass with a 2 kHz passband edge, a 2,048-point Hamming STFT at 50% overlap, contour extraction, constant-amplitude NFM baseband reconstruction, and 128 × 128 continuous grayscale images in [0, 1]." "Items whose paper status is [Unresolved], and every [Implementation Choice] and [Assumption], are listed in the decision register." "Protocol A is a paper-aligned evaluation."

**Not permitted.**
- Calling the detector, post-processing, dataset, or Protocol A "paper-exact" or a reproduction.
- Reporting 1791.4 Hz, or any design cutoff, as a paper parameter.
- Stating that the paper handled channels, deduplicated, dropped or truncated long whistles, or split the data in any particular way; or that Eq. 1 requires deduplication.
- Presenting 12 kHz, NFFT 128, hop 64, the STFT #2 window, the image-geometry rules (D12 to D14), or the normalization rule as stated by the paper.
- Reporting detector recall or precision measured on the annotated files as independent validation.
- Describing the 363 files as the paper's dataset, or as a subset of it, or as verified false killer whale recordings, before D22 is resolved.
- Comparing any result with the paper's tables.

---

## 7. Configuration file

```yaml
# config.yaml -- every preprocessing decision. All stages read this file.
# Each key carries two independent annotations:
#   P: paper status   [Paper] stated in Park et al. 2026 | [Inferred] derived from the paper's text, numbers or figures |
#                     [Unresolved] not stated and evidence insufficient | n/a (engineering parameter)
#   C: our side       [Implementation Choice] a deliberate design decision; where the paper is [Unresolved] it is the
#                         safest neutral option (no data-dependent selection, information-preserving, reversible or a
#                         standard default). It is NEVER a claim about what the paper did.
#                     [Assumption] a provisional, testable belief about the data or the paper's method (H1, channel
#                         synchronisation, species of the files). It may be false.
#                     - = nothing chosen
# H1 = hypothesis about STFT#2 / NFM image synthesis. Its parameters are typed ONLY in the `h1` block. Every value that
#      depends on H1 is "derived" and is computed by h1_image.py into derived_h1.yaml (with a fingerprint). Stages 7-9
#      call assert_fresh() and refuse to run if H1 or any input of the derivation changed.
# null = measure on development files, then freeze before the full run.

dataset:
  paper_files: 2486                     # P:[Paper] Sec. II.C
  paper_whistle_vectors: 228673         # P:[Paper] Sec. II.C
  paper_vectors_per_file: 91.98         # derived: 228673 / 2486 = 91.984
  local_files: 363                      # P:n/a measured by inventory (6.05 h)
  local_fraction_of_paper_files: 0.146  # derived: 363 / 2486 = 14.6%
  expected_vectors_estimate: 33400      # P:n/a estimate = 363 x 91.98 = 33392; NOT a prediction (yield depends on our detector and files)
  same_files_as_paper: unknown          # P:[Unresolved] the paper lists no file identifiers
  target_species_present: unverified    # P:[Paper] the paper selected files "containing whistles from the false killer whale"; the criteria for the local 363 files are unknown
  comparability_with_paper_tables: none # dataset size, population and preprocessing all differ

run:
  seed: 0                               # P:n/a C:[Implementation Choice] root seed; every other seed is derived from it
  n_workers: 4                          # P:n/a C:[Implementation Choice] outputs must not depend on this value
  annotated_files: 20                   # P:n/a C:[Implementation Choice] detector-DEVELOPMENT data (threshold / input-transform tuning); not independent validation
  golden_set_files: 10                  # P:n/a C:[Implementation Choice] regression and inspection set; a SUBSET of annotated_files
  dev_files_source: train_blocks_only   # P:n/a C:[Implementation Choice] under Protocol B, development files come from training blocks only;
                                        #   blocks are assigned at the start from file metadata, before any detection

input:
  flac_dir: /path/to/flac               # edit
  expected_fs: 500000                   # P:[Paper] 500 kHz; matched by all 363 files in the inventory
  expected_channels: 6                  # P:[Paper] six hydrophones; matched by all 363 files in the inventory
  expected_duration_s: 60.0             # P:[Paper] files "generally" contain ~60 s. NOMINAL ONLY, not enforced by stage1_audio.py.
                                        #   The inventory total (6.05 h = 363 x 60 s) shows only that the AVERAGE is ~60 s;
                                        #   per-file durations are screened with duration_warn_range_s.
  duration_warn_range_s: [55.0, 65.0]   # P:n/a C:[Implementation Choice] inventory reports files outside this range; reported, not rejected

channels:
  policy: all_then_dedup                # P:[Unresolved] (paper states only "six hydrophones") C:[Implementation Choice]
  use: [0, 1, 2, 3, 4, 5]

time:
  origin: file_start                    # P:n/a C:[Implementation Choice] ONE file-relative time coordinate shared by all channels (seconds from first sample)
  frame_center_s: "(k*hop + nfft/2)/fs_out"   # P:n/a C:[Implementation Choice] same formula and frame grid for every channel; never a per-channel or per-track origin
  require_same_frame_count: true        # fatal if channels of one file disagree
  synchronous_channels: true            # P:[Unresolved] C:[Assumption] channels are sample-synchronous; test against DCLDE documentation [29] and the lag measurement

resample:
  fs_out: 192000                        # P:[Paper]
  up: 48                                # 192000/500000 = 48/125
  down: 125
  method: polyphase_kaiser              # P:[Unresolved] C:[Implementation Choice] scipy.signal.resample_poly (Kaiser beta 5)

highpass:
  passband_edge_hz: 2000                # P:[Paper] "passband above 2 kHz" (the ONLY paper-stated high-pass parameter)
  corner_spec: minus3db_at_passband_edge   # P:[Unresolved] C:[Implementation Choice] (alternative: design_cutoff_equals_edge) meaning of "passband above 2 kHz"
  design_cutoff_hz: derived             # C:[Implementation Choice] realisation of the 2 kHz edge, NOT a paper parameter. stage1_audio.design_cutoff_hz()
                                        #   computes edge * (sqrt(2)-1)^(1/(2n)) = 1791.4 Hz for n=4, zero-phase, so the realised response is -3.01 dB at 2000 Hz
  order: 4                              # P:[Unresolved] C:[Implementation Choice]
  type: butterworth                     # P:[Unresolved] C:[Implementation Choice]
  zero_phase: true                      # P:[Unresolved] C:[Implementation Choice] sosfiltfilt; the response is squared
  apply_before_stft: true               # P:[Unresolved] C:[Implementation Choice] the paper mentions the filter after describing the STFT

stft1:
  window: hamming                       # P:[Paper]
  nfft: 2048                            # P:[Inferred] 192000/2048 = 93.75 Hz bin spacing quoted in the paper
  hop: 1024                             # P:[Inferred] from "50% overlap" of a 2048-point window; consistent with the quoted ~5.3 ms
  bin_hz: 93.75
  frame_ms: 5.3333

detector:
  type: classical_ridge_tracker         # P:[Unresolved] (neural detector [31] named, not reproducible) C:[Implementation Choice] stand-in; equivalence to the paper's detector is untested; never "paper-exact"
  band_hz: [2000, 20000]                # P:[Unresolved] C:[Implementation Choice] high-pass edge to the Fig. 3 display range
  input_transform: per_bin_median_subtraction_db   # P:[Unresolved] C:[Implementation Choice] detector-side only; ablate
  ablate_transforms: [none, per_bin_median_subtraction_db, sliding_local_median_db]
  subbin_frequency: parabolic           # P:[Unresolved] C:[Implementation Choice] vector frequencies stored as float Hz
  min_contrast_db: null                 # set on annotated (development) files
  min_duration_frames: null             # set on annotated (development) files
  max_gap_frames: null                  # set on annotated (development) files
  reject_harmonics: true                # P:[Unresolved] C:[Implementation Choice]

dedup:
  enabled: true                         # P:[Unresolved] (paper does not mention it) C:[Implementation Choice] our independent engineering choice, not a paper requirement
  scope: within_file                    # P:n/a C:[Implementation Choice] uses the common file-relative time coordinate (see time)
  max_lag_s: null                       # measure on development files: P99.9 |inter-channel lag| + one frame (5.333 ms)
  min_temporal_overlap_fraction: 0.5    # P:n/a C:[Implementation Choice] of the shorter track, after lag alignment
  max_mean_abs_freq_diff_hz: 93.75      # P:n/a C:[Implementation Choice] one STFT#1 bin, over overlapping frames
  keep: seeded_random                   # P:[Unresolved] C:[Implementation Choice] an SNR-based rule is NOT supported and is not used
  rng: per_file_seedsequence            # see reproducibility
  ambiguous_group_policy: keep_all_and_count   # two tracks of one channel in a match group -> do not merge
  store_contrast_db_metadata: true      # descriptive only; never used for selection
  sensitivity_variant_without_dedup: true   # also build a variant with enabled=false and compare feature distributions

support:
  max_length_frames: 128                # P:[Paper] the image is 128 wide; "one column = one STFT#1 frame" is H1 (checked in h1_image.py)
  long_whistle_policy: drop_and_report  # P:[Unresolved] C:[Implementation Choice] never truncate / split / time-scale in the main dataset
  keep_overlong_contours: true          # stored outside the dataset for sensitivity analysis
  baseband_margin_bins: 2               # P:n/a C:[Implementation Choice] guard against spectral wrap-around at +-nfft/2
  max_abs_baseband_bins: derived        # H1-derived: nfft/2 - baseband_margin_bins
  review_trigger_dropped_fraction: 0.05 # P:n/a C:[Implementation Choice] pre-registered; report as a limitation if exceeded

vector:
  center_definition: midrange           # P:[Unresolved] C:[Implementation Choice] (fmin+fmax)/2 is consistent with Fig. 8; the mean is untestable from the figures
  freq_units: hz_float

h1:                                     # P:[Unresolved] C:[Assumption] hypothesis H1; the ONLY place its parameters are typed
  baseband_fs: 12000                    # NFM baseband sample rate (Hz)
  window: hamming                       # STFT#2 window
  nperseg: 128
  nfft: 128                             # two-sided complex spectrum, fftshifted: rows = nfft
  hop: 64                               # 64/12000 = 5.333 ms = STFT#1 frame period (checked)
  spectrum: two_sided_fftshift
  derive_script: h1_image.py
  derived_file: derived_h1.yaml         # generated; never edited by hand
  stale_policy: fatal                   # stages 7-9 refuse to run if the fingerprint does not match

nfm:
  interp: linear                        # P:[Unresolved] C:[Implementation Choice]
  phase_dtype: float64                  # P:n/a C:[Implementation Choice]
  amplitude: 1.0                        # P:[Paper] constant amplitude (the value 1 is arbitrary)
                                        # baseband sample rate: h1.baseband_fs

stft2:
  magnitude: linear                     # P:[Inferred] (Fig. 1 background ~0) C:[Implementation Choice]
  normalization: constant_window_sum    # P:[Unresolved] (range [0,1] and "close to 1" are [Paper]) C:[Implementation Choice] divide by sum(window) x amplitude: data-independent, guarantees [0,1]
  alt_normalization: per_image_max      # switch; only one rule may be used across the whole dataset
                                        # window, nperseg, nfft, hop: h1.*

image:
  size: [128, 128]                      # P:[Paper] rows, columns
  center_row_0based: derived            # H1-derived: nfft // 2 (DC bin after fftshift)
  time_alignment: centered              # P:[Inferred] Figs. 8 and 10 (read by eye, +-2 columns)
  start_column: floor_half_of_remaining # C:[Implementation Choice] H1-dependent rule (128 - L) // 2, one column per STFT#1 frame
  resize: false                         # P:[Unresolved] C:[Implementation Choice] H1 needs no resize; the paper's wording does not exclude one
  pad: zero_time_padding_of_signal      # P:[Unresolved] C:[Implementation Choice] H1
  training_threshold: null              # P:[Paper] training images are continuous grayscale in [0, 1]; never thresholded or quantized
  decode_threshold: 0.05                # P:[Paper] Fig. 2(b); contour decoding and round-trip QC only
  fid1_threshold: 0.1                   # P:[Paper] FID_1 evaluation only

qc:
  clip_fraction_warn: 0.001             # P:n/a C:[Implementation Choice]
  dead_channel_rms: 1.0                 # P:n/a C:[Implementation Choice] int16 counts
  integrity:                            # fatal: indicates a bug; halt the run
    require_finite: true
    require_range: [0.0, 1.0]
    require_shape: [128, 128]
  fidelity:                             # quarantined, never silently dropped
    tolerance_rule:                     # C:[Implementation Choice] how H1-derived tolerances are set from the H1 simulation
      centroid_safety_factor: 1.5       #   max_centroid_error_bins = ceil(factor x simulated max centroid error)
      length_margin_frames: 1           #   max_length_error_frames = simulated max length extension + margin
    max_centroid_error_bins: derived    # H1-derived
    max_length_error_frames: derived    # H1-derived
    halt_if_failure_rate_above: 0.01    # P:n/a C:[Implementation Choice] pre-registered review trigger
    quarantine_dir: quarantine
    bias_check: true                    # compare 7-feature distributions of passed vs quarantined

protocols:
  A_paper_aligned_evaluation:           # P:[Inferred] the paper describes no held-out set and compares with "the training data"
    train: all_files                    # C:[Implementation Choice] NOT a paper-exact reproduction: detector, post-processing, dedup,
    evaluate_against: training_set      #   channel handling, image-synthesis parameters and normalization are unresolved or stand-ins
  B_leakage_controlled:                 # P:n/a C:[Implementation Choice] independently introduced; not in the paper
    unit: contiguous_time_block         # fallback: file; use encounter/recording id if available
    guard_files: 1
    fractions: {train: 0.8, val: 0.1, test: 0.1}
    test_used_once: true

training_time:                          # NOT preprocessing; never stored in the dataset
  rotation_augmentation:                # P:[Paper] Appendix A.B: each training image randomly rotated by 0/90/180/270 deg, equal probability
    angles_deg: [0, 90, 180, 270]       # P:[Paper]
    probability: equal                  # P:[Paper]
    applied_to_vae_input_and_target: true     # P:[Unresolved] (paper: "each image") C:[Implementation Choice] the rotated tensor is both encoder input and reconstruction target
    applied_to_critic_real_samples: true      # P:[Unresolved] C:[Implementation Choice] same rotated tensor, so the critic compares like with like
    resampled_each_iteration: true            # P:[Unresolved] ("randomly rotated") C:[Implementation Choice]
    stored_in_dataset: false                  # P:n/a C:[Implementation Choice]
    used_in_preprocessing_or_qc: false        # P:n/a C:[Implementation Choice]
  gmm_fit_latents: unaugmented_encoder_means  # P:[Unresolved] ("latent mean outputs of encoder") C:[Implementation Choice]
  ablation_without_rotation: separate_labelled_run_only   # NOT paper-aligned; never mixed into paper-aligned results

reproducibility:                        # P:n/a C:[Implementation Choice]; see Section 9 of the document
  root_seed_from: run.seed
  per_file_seed: "SeedSequence([run.seed, stable_hash(file_name)])"   # independent of worker count and scheduling order
  seed_python_random: true
  numpy_rng: generator_from_seedsequence   # never the global np.random state
  env:                                  # set BEFORE the interpreter / CUDA starts
    PYTHONHASHSEED: "0"
    OMP_NUM_THREADS: "1"
    MKL_NUM_THREADS: "1"
    OPENBLAS_NUM_THREADS: "1"
    CUBLAS_WORKSPACE_CONFIG: ":4096:8"
  scipy_fft_workers: 1
  torch:                                # only where PyTorch is used (GPU detection, VAE / critic training)
    manual_seed: true                   # torch.manual_seed seeds CPU and all CUDA devices
    use_deterministic_algorithms: true  # ops without a deterministic implementation raise an error; resolve, do not silence
    cudnn_deterministic: true
    cudnn_benchmark: false
    allow_tf32: false                   # torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
  dataloader:
    generator_seed_from: run.seed       # torch.Generator().manual_seed(...) passed to DataLoader (controls shuffling)
    worker_init_fn: seed_python_and_numpy_from_worker_seed
    num_workers: explicit               # recorded with the run
    augmentation_rng: per_sample_seed(run.seed, epoch, sample_index)   # rotation choice independent of num_workers
  record_environment: [python, numpy, scipy, soundfile, libsndfile, torch, cuda, cudnn, driver, gpu_model, git_commit, config_hash]
  golden_regression:
    same_environment:                   # identical software, hardware, config and seeds
      images: bitwise_identical
      contour_set: identical
    cross_environment:                  # different library build, CPU/GPU model, thread count or driver: numerical tolerances only
      image_max_abs: 1.0e-5
      contour_freq_hz_max_abs: 1.0e-3
      contour_matching: one_to_one_within_tolerance   # every contour needs a counterpart; start and end frames equal
      unmatched_contours: report_and_review           # borderline detections can flip; never auto-accepted

storage:
  image_dtype: float32
  compression: lossless
  keep_contours: true
  keep_features: true
```

---

## 8. Validation procedures

**Delivered code and tests.** `stage1_audio.py` and `test_stage1.py` (Stage 1); `h1_image.py`, `derived_h1.yaml`, and `test_h1.py` (H1 synthesis, derivation, and staleness guard). Run both test files after any change to the config or code.

**Development data, stated plainly.** The annotated files (20, a working choice) and the golden set (10) are drawn from the same 363 files that feed the model. The golden set is a *subset* of the annotated set, so the golden files have ground truth for regression checks. Both are **detector-development data**: thresholds, the D5 input transform, and the dedup lag are tuned on them, so any recall or precision measured on them is in-sample performance and not independent validation. Independent detector validation would require annotating further files never used for tuning, after the detector is frozen; this is optional and not assumed. Under Protocol B, development files come from training blocks only.

1. **Stage 1 smoke check.** Run on one real file and inspect the six per-channel RMS values. Repeat on about 10 files from different parts of the dataset.
   ```
   python stage1_audio.py --config config.yaml --file /path/to/one.flac
   ```
2. **Dataset population (D22).** Check whether the golden files contain false killer whale whistles, and record how the 363 files were obtained and selected.
3. **Annotate** the development files (recordings and noise levels spanning the dataset). Use them to set detector thresholds and choose the D5 transform.
4. **Golden set.** Run Stages 2 to 9 on the golden files and inspect contour overlays by eye.
5. **Lag measurement (D7).** Measure inter-channel lags on high-contrast whistles in the development files, in the common time coordinate (D20), and set `dedup.max_lag_s`. Check the channel-synchronization assumption from the lag distribution.
6. **H1 checks (D10).** Compare synthesized images with Figs. 1, 2, 8 and 10 for thickness, onset/offset extension, and centering. If any H1 parameter changes, re-run `h1_image.py derive`, `test_h1.py`, and this step.
7. **Dedup sensitivity.** Build the variant with `dedup.enabled: false` and compare the seven-feature distributions and counts with the main dataset.
8. **Fill every `null`, then freeze the config** before the full run.
9. **Full run, then the funnel report.** Review it against the pre-registered triggers.

---

## 9. Reproducibility requirements

The pipeline is **deterministic by design**: the same inputs, config, seeds, and software environment must give the same outputs.

**Regression criteria (`reproducibility.golden_regression`).**
- **Same environment** (identical software, hardware, config, and seeds): images must be bitwise identical and the contour set must be identical (`contour_set: identical` applies here only).
- **Different environment** (different library build, CPU or GPU model, thread count, or driver): bitwise identity is not expected. Results must be **numerically equivalent within tolerance**: image maximum absolute difference ≤ 1e-5, matched contour frequency difference ≤ 1e-3 Hz, every contour matched one-to-one with equal start and end frames. Unmatched contours (borderline detections can flip) are reported and reviewed, never accepted automatically.

**Seeds.**
- One root seed (`run.seed`). Everything else is derived from it.
- Per-file preprocessing randomness (for example, the duplicate choice in D8) uses `SeedSequence([seed, stable_hash(file_name)])`, so results do not depend on the worker count, scheduling order, or number of files.
- Python `random` is seeded. NumPy uses a `Generator` built from a `SeedSequence`; the global `np.random` state is not used.
- `PYTHONHASHSEED` is set in the environment before the interpreter starts.
- PyTorch (where used: GPU detection, VAE and critic training): `torch.manual_seed` (seeds CPU and all CUDA devices).
- CUDA: `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA initializes.

**Deterministic settings.**
- `torch.use_deterministic_algorithms(True)`. An operation without a deterministic implementation raises an error; resolve it by substituting the operation, and do not silence the error.
- `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- TF32 disabled for matmul and cuDNN.
- BLAS and OpenMP threads set to 1 per worker; `scipy.fft` with `workers=1`.

**DataLoader.**
- Shuffling uses an explicitly seeded `torch.Generator`.
- A `worker_init_fn` seeds Python and NumPy inside each worker from the worker seed.
- `num_workers` is explicit and recorded.
- Rotation augmentation (D19) draws its angle from a per-sample seed `(seed, epoch, sample_index)`, so the augmentation stream does not change with `num_workers` or batch composition.

**Environment record (saved with every run).** Python, NumPy, SciPy, soundfile and libsndfile, PyTorch, CUDA, cuDNN, driver, GPU model, git commit, and the config hash.

**Limits.** GPU training may still differ across GPU models and across the number of devices. The paper trained on seven V100 GPUs with a total batch of 896 (Appendix A.B), and exact agreement with its training runs is not expected.

---

## 10. Performance notes

The work is dominated by resampling, STFT, and detection over about 6.5 × 10¹⁰ samples. Stage 1 takes about 7.8 s per file in the test environment (single process).

**Deterministic by design** (same environment, same results; no change to the computation): file-level parallelism; streaming one file and one channel at a time; precomputed window, resampler taps, and filter coefficients; `rfft` on strided views; sort-and-sweep deduplication; resumable per-file outputs keyed by file name and config hash.

**Numerically equivalent within tolerance** (verify against the cross-environment criteria in Section 9 before adopting): GPU resampling, STFT, and detector inference; batched NFM synthesis with phase accumulation kept in float64.

**May change the output; validate on the golden set and compare the funnel report first:** mixed-precision detector inference; energy gating; fewer channels; uint8 or float16 image storage.

---

## 11. Unresolved scientific questions

These cannot be settled from the paper, the 363 files, or anything else in the project. Each keeps its [Unresolved] status until new evidence (the authors, the cited conference papers, or the dataset documentation) is obtained.

1. **Channel handling in the paper** (single channel, all channels, or beamforming), and whether duplicates were removed (D1, D7).
2. **The neural detector [31] and the post-processing [32], [33]:** architecture, inputs, thresholds, and whether our stand-in yields a comparable contour distribution (D6).
3. **Input normalization applied before detection** (D5).
4. **Handling of whistles longer than 128 frames** (D9).
5. **STFT #2 / NFM parameters** (baseband rate, window, NFFT, hop) and whether any resizing or cropping produced the 128 × 128 image (D10, D14). H1 is consistent with the figures but not established.
6. **The definition of "center frequency"** and the exact alignment convention of the images (D11, D13); figure-based evidence is accurate only to about ±2 columns by eye.
7. **Per-image versus global intensity normalization** (D15).
8. **The meaning of "passband above 2 kHz"**, the filter family and order, and whether the filter preceded the STFT (D3).
9. **Whether the VAE and GMM were trained on all 228,673 whistles** and exactly which set the reported metrics compare against (D18).
10. **Rotation-augmentation details:** whether the critic's real samples and the reconstruction targets were rotated, and whether GMM latents came from rotated images (D19).
11. **How the paper converts image features to Hz and ms** in Table 3 (its frequency conditions are in bins, while its Wasserstein errors are in Hz and ms).
12. **Whether the 363 local files are a subset of the paper's 2,486, and whether they contain false killer whale whistles** (D22).
13. **Array geometry and channel synchronization of the recordings** (D20): not in the project; obtainable from the DCLDE 2022 documentation [29].
