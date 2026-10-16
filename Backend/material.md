# Panel Presentation Material — Model Architectures & Configurations

Fetal Cardiac Ageing Clock. All values below are read from `configs/default.yaml` and `src/fcac/`.
Parameter counts were measured by instantiating the models (not estimated).

---

## 0. The 60-second story (open with this)

Noisy 4-channel maternal abdominal ECG goes in. Five stages follow:

```
Abdominal ECG (4 ch, 1 kHz)
  │  Preprocess: bandpass 1–100 Hz → (50 Hz notch*) → z-score → 5 s windows (4 × 5000)
  ▼
[1] MEMD          remove motion / baseline drift / muscle artefact
  ▼
[2] Fetal ECG extraction   maternal-ECG cancellation → PINN  (fallback: FastICA)   "hybrid"
  ▼
[3] 12 biomarkers          5 morphological + 7 heart-rate-dynamics
  ▼
[4] CVAE                   12 features → 64-d latent  (contrastive, no GA input)
  ▼
[5] Transformer            sequence of latents → Cardiac Age (CA, days)
  ▼
ΔAge = CA − GA     (GA = LMP-dated, an *input label*, never a model output)
```
\*notch applies to NInFEA and CinC 2013 only.

**Why the split into stages?** Each stage is separately testable and interpretable. The 12
biomarkers are clinically meaningful, so the age estimate is not a black box on raw waveforms.

---

## 1. MEMD — Multivariate Empirical Mode Decomposition

**Role (say this precisely):** removes motion, baseline drift and muscle artefact **only**.
Maternal–fetal separation is *not* MEMD's job.

### What it does
EMD splits a signal into Intrinsic Mode Functions (IMFs), ordered high → low frequency. The
*multivariate* version decomposes all 4 channels **jointly**, so IMF *k* means the same
frequency scale in every channel (plain per-channel EMD misaligns modes across channels).

Algorithm (`memd/sift.py`, vendored):
1. Generate quasi-uniform direction vectors on the hypersphere (Hammersley sequence — low-discrepancy,
   so 64 directions cover the sphere without gaps; random vectors would leave visible holes).
2. Project the 4-channel signal along each direction; find extrema of each projection.
3. Cubic-spline interpolate the extrema → envelopes; average over directions → multivariate mean envelope.
4. Subtract the mean (= sifting). Repeat until stopping tolerance is met → one IMF. Subtract, repeat on the residue.
5. A direction "abstains" if its projection has < 3 extrema; when ≥ 50% abstain, the residue is a trend → stop.

### IMF selection (`memd/select.py`)
Not a fixed "IMF 3–5". Each IMF is scored per recording:
**score = spectral coherence × in-band power**, fetal band 20–70 Hz. Selected IMFs are summed
into the "fetal-band window" that feeds the next stage. (ADFECGDB `Direct_1` is used as a scoring
reference only, never as a model input.)

### Configuration
| Parameter | Value |
|---|---|
| Directions | 64 |
| Stopping tolerance | 0.075 |
| Max sift iterations | 300 |
| Fetal band | 20–70 Hz |
| Abstain fraction | 0.5 |
| IMF count | data-determined (~10–14) |
| Input / output | (W, 4, 5000) → IMFs (W, K, 4, 5000), scores (W, K) |
| Caching | `data/memd/{dataset}/{record}/win{idx}.npz` (MEMD is slow) |

**Likely panel question:** *"Why MEMD and not a bandpass filter?"* — Bandpass is linear and fixed;
motion/muscle artefact overlaps the fetal band in frequency. MEMD is data-adaptive and non-stationary-friendly,
and joint decomposition preserves inter-channel structure that the PINN uses.

---

## 2. Maternal ECG cancellation (bridge between MEMD and PINN)

Template subtraction of the averaged maternal QRST complex (`mecg.py`).

- Config: template half-width 120 ms, min 3 maternal beats, `mode: adaptive`.
- **Why adaptive:** measured on ADFECGDB (QRS F1):

| Record | MEMD | + cancel | Effect |
|---|---|---|---|
| r01 (clean) | 0.934 | 0.760 | hurts |
| r08 (clean) | 0.959 | 0.722 | hurts |
| r04 (maternal-locked) | 0.202 | 0.332 | helps |
| r07 | 0.190 | 0.453 | helps |
| r10 | 0.217 | 0.388 | helps |

Cancelling on an already-clean recording destroys fetal signal, so it is applied only when the
window's heart rate is outside the fetal band (unsupervised criterion, no ground truth used).

---

## 3. PINN — Physics-Informed Neural Network

### Architecture (LOCKED)
```
input (8) → FC512 → Tanh → FC512 → Tanh → FC512 → Tanh
          → FC512 → ReLU → Dropout(0.2) → FC512 → ReLU → Dropout(0.2) → FC(1) = V
```
- 5 hidden layers × 512 units. **Tanh on layers 1–3** (smooth, infinitely differentiable —
  required because we take second derivatives), **ReLU + Dropout 0.2 on layers 4–5**.
- **1,055,745 parameters.**
- Output: scalar potential **V**.

### Physics constraint
Laplace's equation **∇²V = 0**, computed by **double autograd** (`pinn/physics.py`): first
`autograd.grad` gives ∇V, then each ∂²V/∂xᵢ² is differentiated again and summed.
The Laplacian is taken over the **spatial** columns only (x, y, z), not time — quasi-static
approximation: time enters V as a parameter. Evaluated at 512 interior collocation points
(not at the electrodes: four near-coplanar samples cannot identify a Laplacian).

### Loss (LOCKED)
```
L = 0.7 · L_data + 0.3 · L_physics
L_data    = MSE(V at electrodes, observed fetal-band signal)
L_physics = mean( (∇²V)² ) at collocation points
```

### Input coordinates (**ASSUMPTION — be upfront about this**)
`[x, y, z, t, obs₀..obs₃]` → 8 dims. Electrode positions are an approximate 4-around-the-navel
layout (EDF headers carry no geometry). The spec fixes width/depth/activations/loss weights but
**not** the coordinate scheme; this is one concrete instantiation, **pending supervisor sign-off**.
No PINN-specific number should be presented as a final result.

### Configuration
| Parameter | Value |
|---|---|
| Hidden layers | [512 ×5] |
| Activations | tanh, tanh, tanh, relu, relu |
| Dropout | 0.2 (layers 4–5) |
| Loss weights | data 0.7 / physics 0.3 |
| Optimiser | Adam, lr 1e-3 |
| Epochs / batch | 200 / 32 |
| Collocation points | 512 per step |
| Data points per window | 1024 |
| Readout | `potential` (V itself) |
| Training data | ADFECGDB only, then **frozen** (no fine-tuning) |

### Honest finding to volunteer
Readout `potential` (QRS F1 0.519) beat `residual` (0.443), and beat MEMD's own input (0.501).
So the Laplace constraint behaves as a **spatial denoiser**, not a maternal-absorber as first
hypothesised. Training the PINN on NInFEA too gave no improvement (NInFEA failure is maternal-lock,
not lack of data), so ADFECGDB-only stays the default.

### Extraction: why "hybrid" is the default
Per window: cancel → PINN output if it passes the quality gate (110–160 bpm, ≤ 25% RR artefacts,
≥ 5 beats), otherwise FastICA blind source separation. Measured QRS F1:

| Cohort | PINN | cancel+PINN | FastICA | **Hybrid** |
|---|---|---|---|---|
| ADFECGDB (PINN in-sample) | – | 0.630 | 0.557 | **0.637** |
| CinC 2013 (zero-shot) | 0.377 | 0.440 | 0.519 | **0.542** |

On NInFEA the PINN passed the gate on only ~6% of windows; most NInFEA windows end up as FastICA output.

---

## 4. CVAE — Contrastive Variational Autoencoder

**Contrastive, not Conditional.** GA is *never* an input (it would be circular: the model would just
copy the label).

### Architecture (`cvae/model.py`)
```
x (12 features, standardised)
 ├─ Encoder:  [Linear→LayerNorm→GELU→Dropout] 12→128→64 ; heads μ, logvar (64 each; logvar clamped ±8)
 ├─ z = μ + σ·ε   (reparameterisation; at eval z = μ)
 ├─ Decoder:  [Linear→LayerNorm→GELU→Dropout] 64→64→128 → Linear → 12
 └─ Projector (InfoNCE only): Linear 64→64 → GELU → Linear 64→64 → L2-normalise
```
- **41,356 parameters.**
- Why a **projection head**: InfoNCE on `z` directly would fight the reconstruction objective for the
  same embedding space; the SimCLR-style head keeps `z` clean for the Transformer.
- Why **standardise inputs**: raw features differ in scale by orders of magnitude and the largest
  dominated the MSE. Measured held-out per-feature R² median: **−0.01 → 0.86** after standardising;
  active latent dims → 64/64 (no posterior collapse).

### Loss
```
L = L_recon (MSE) + β·L_KL + L_InfoNCE
β: linear warm-up 0 → 0.1 over 20 epochs  (prevents posterior collapse)
```
InfoNCE, **τ = 0.07**, multi-positive. **Positives** = same-subject windows, or windows whose GA is within
**±7 days**. (A hard "same GA" match yields almost no positives since GA is effectively continuous.)

### Configuration
| Parameter | Value |
|---|---|
| Input dim | 12 |
| Latent dim | 64 |
| Encoder / decoder hidden | [128, 64] / [64, 128] |
| Projection dim | 64 |
| Dropout | 0.2 |
| β_KL / warm-up | 0.1 / 20 epochs |
| Temperature τ | 0.07 |
| GA window for positives | ±7 days |
| GA as input | **false (LOCKED)** |
| Optimiser | AdamW, lr 1e-3, wd 1e-4, grad-clip 1.0 |
| Epochs / batch | 200 / 64 |

### The 12 input features (fixed order, `FEATURE_NAMES`)
Morphological (5): QTc σ, P-dispersion, QRS mean, QRS variance, QRS asymmetry.
Dynamic (7): PRSA, multiscale entropy, ApEn, SampEn, DFA α, LF/HF ratio, symbolic dynamics.
Features return NaN when a 5 s window has too few beats (≈ 8–12 RR intervals) — never a fabricated number.

### Role beyond encoding
Replaces the dropped WGAN: augmentation by sampling latent-space neighbours (`cvae/augment.py`).
WGAN was dropped because it is unstable at ~90 recordings.

---

## 5. Transformer — Cardiac Age regressor

### Architecture (`transformer/model.py`)
```
z sequence (L ≤ 64, 64-d)
 → Linear 64→256 (input projection)
 → prepend learnable [CLS] token, add learnable positional embeddings
 → 3 × TransformerEncoderLayer (pre-LN, 4 heads, d_k = 64, FFN 1024, GELU, dropout 0.2)
 → LayerNorm → take CLS → MLP head 256→128→1
 → Cardiac Age (days)
```
- **2,436,353 parameters.**
- **d_model 256 = 4 heads × d_k 64** (checked in the constructor).
- **CLS-token pooling, not mean-pooling:** recordings have variable window counts; mean-pooling
  weights every window equally, a CLS token lets attention decide which windows matter. A boolean
  padding mask handles variable length.
- **Target scaling:** GA is ~266–301 days; a fresh head outputs ~0 and AdamW at 1e-4 can't close a
  ~280-day gap, so targets are z-scored for training and un-scaled (buffers `target_mean/std`) at
  inference. (We reproduced a collapsed MAE ≈ 266 d before this fix.)

### Configuration
| Parameter | Value |
|---|---|
| Heads / d_model / d_k | 4 / 256 / 64 |
| Layers | 3 |
| FFN dim | 1024 |
| Dropout / head dropout | 0.2 / 0.1 |
| Max sequence length | 64 |
| Loss | MAE |
| Optimiser | AdamW, lr 1e-4, wd 1e-2, 10 warm-up epochs, grad-clip 1.0 |
| Epochs / batch | 200 / 16 |
| Training crops | random 16-window crop per step, 10 repeats/epoch |

---

## 6. Evaluation protocol

- **Pooled leave-one-subject-out CV** over ADFECGDB (5) + gated NInFEA (grouped by mother, so a mother's
  recordings are held out together). CVAE is **refit inside each fold** on training subjects only (no leakage).
- **CinC 2013** = zero-shot, never trained on (but has no published GA, so no ageing metric).
- **MAE in days**, 95% bootstrap CI (1000 resamples, 2.5–97.5 percentile); mandatory unit-verification script.
- Datasets: ADFECGDB (5 rec, GA 38–41 wk), NInFEA (60 rec / 39 subjects, GA 21–27 wk), CinC 2013 (25 rec).
  **28–37 wk is unrepresented** — never assume overlapping GA ranges.

---

## 7. Be ready to state these limitations (panels reward candour)

1. **ADFECGDB GA labels are currently synthetic** (seeded random 38–41 wk). Any ADFECGDB-based age
   number is a pipeline check, **not reportable**. Pooled MAE (22.6 d, r 0.56 vs 20.6 d for predicting
   the mean) is dominated by cohort separation, not within-cohort age signal.
2. **Within NInFEA (GA 147–194 d) no feature correlates with GA** (|ρ| ≤ 0.2, nothing significant after
   Bonferroni; the transformer's within-cohort r ≈ −0.02). Weak effects can't be excluded with ~50
   recordings, but they're too weak to build a predictor on.
3. **PINN coordinate scheme is an unsigned assumption.** Report PINN as "implemented, working
   hypothesis", not as a validated result.
4. SNR vs the scalp lead is ~0.4–0.5 dB for all arms and does not separate methods; the informative
   metrics are beat-timing ones (QRS F1, heart-rate agreement). Not comparable with the ICC 2026
   CNN's 14.4 dB, which is trained to regress the scalp signal.
5. NInFEA gate-pass is optimistic for FastICA output: the gate keeps 98% of right-rate windows but
   only 63% of what it keeps are right.
6. CVAE latent (64) > input (12): high reconstruction R² shows the autoencoder works, not that the
   latent carries age information.

---

## 8. Quick-reference summary table

| Model | Type | Params | Key dims | Loss | Optimiser |
|---|---|---|---|---|---|
| MEMD | Signal decomposition | – | 64 directions, 4 ch, band 20–70 Hz | – | – |
| PINN | FC network + Laplace PDE | 1,055,745 | 8→5×512→1 | 0.7·MSE + 0.3·‖∇²V‖² | Adam 1e-3 |
| CVAE | Contrastive VAE | 41,356 | 12→128→64→z64 | MSE + β·KL + InfoNCE (τ 0.07) | AdamW 1e-3 |
| Transformer | Encoder + CLS head | 2,436,353 | d256, 4 heads, 3 layers | MAE | AdamW 1e-4 |

## 9. Likely questions, short answers

- **Why Tanh in early PINN layers?** Smooth and twice-differentiable, needed for the Laplacian via double autograd; ReLU has zero second derivative.
- **Why not feed GA to the CVAE?** Circular — the model would learn the label rather than cardiac maturity.
- **Why contrastive?** With ~90 recordings a pure VAE latent has no reason to organise by maturity; same-subject/similar-GA positives inject that structure without using GA as input.
- **Why not a bigger/raw-signal model?** ~90 recordings; 12 interpretable biomarkers keep the model small and clinically explainable.
- **Why a transformer on ~64 recordings?** Honest answer: it's the specified architecture; current results show it cannot find within-cohort age signal, which we attribute to the features/data rather than model capacity (see §7).
- **Why drop WGAN?** Unstable at ~90 recordings; CVAE latent sampling replaces it.
