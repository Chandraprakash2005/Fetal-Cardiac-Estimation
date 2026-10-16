# Project Flow, Challenges and Next Phase

Fetal Cardiac Ageing Clock. Companion to `material.md` (model architectures and configs).
Sources: `reports/engineering_log.md`, `configs/default.yaml`, and the supervisor's paper
(`eECG_Extraction_Muthukumar_Sir.pdf`, ICC 2026).

---

## 1. How the project works, input to output

1. **Input:** 4-channel maternal abdominal ECG at 1 kHz, plus the LMP-dated gestational age (GA) as a label.
   GA is never a model output.
   - Datasets: ADFECGDB (5 recordings, 38–41 wk), NInFEA (60 recordings, 21–27 wk),
     CinC 2013 (25 recordings, no GA, zero-shot).
2. **Preprocessing:**
   - Channel choice. For NInFEA this means the top 4 of 24 by kurtosis, computed after filtering.
   - Bandpass 1–100 Hz, with a 50 Hz notch on NInFEA and CinC.
   - Resample to 1 kHz, z-score, cut into 5 s windows of shape (4, 5000).
3. **MEMD:**
   - Decompose all 4 channels jointly (64 directions) into IMFs.
   - Score each IMF by coherence × in-band power (20–70 Hz) and sum the selected ones.
   - Removes drift, motion and muscle artefact only.
4. **Maternal cancellation:** subtract an averaged maternal QRS-T template, only when the window's rate is outside the fetal band.
5. **Hybrid extractor:**
   - Run the frozen PINN (potential V, Laplace constraint, 0.7/0.3 loss).
   - If the window passes the quality gate (110–160 bpm, few RR artefacts, at least 5 beats), keep it. Otherwise use FastICA.
   - Output: fetal ECG of shape (W, 5000).
6. **12 biomarkers:** 5 morphological and 7 heart-rate-dynamics features (QTc σ, P-dispersion, QRS stats, PRSA, MSE, ApEn, SampEn, DFA, LF/HF, symbolic).
7. **CVAE:** 12 features to a 64-d latent, trained with reconstruction, KL and InfoNCE (τ = 0.07). GA is not an input.
8. **Transformer:** the sequence of latents goes through 4 heads, d_model 256 and 3 layers. The CLS token feeds a regression head that outputs Cardiac Age (CA) in days.
9. **Output:** ΔAge = CA − GA, a candidate IUGR screening index. Reported as MAE in days with a 95% bootstrap CI, from pooled leave-one-subject-out CV.

```
Abdominal ECG → Preprocess → MEMD → Maternal cancel → Hybrid (PINN | FastICA)
   → 12 biomarkers → CVAE (64-d) → Transformer → CA → ΔAge = CA − GA
```

---

## 2. Challenges faced

| Area | Problem | What we did |
|---|---|---|
| Maternal lock | On r04, r07 and r10 the "fetal" output sat at 82–93 bpm (the mother's rate) against a true fetal rate of 125–131 bpm. | Added adaptive maternal cancellation and a quality gate. |
| Cancellation | It helped locked records but hurt clean ones (F1 0.934 → 0.760 on r01). | Made it adaptive: it runs only when the rate is out of band. |
| PINN | The coordinate scheme isn't fixed by the spec. It passed the gate on only ~6% of NInFEA windows, and training it on NInFEA didn't help. | Added the FastICA fallback and the hybrid extractor. The PINN scheme is still awaiting sign-off. |
| NInFEA | Dead channel 34 was picked by kurtosis. Recording 50 had a saturated channel. There is no scalp ECG. | Fixed the NaN ranking. Built a Doppler beat reference for timing metrics only. |
| Metrics | SNR against the scalp lead is ~0.4 dB for every method, so it doesn't separate them. It also isn't comparable to the paper's 14.4 dB. | Use QRS F1 and heart-rate agreement instead. |
| Features | A 5 s window holds only 8–12 RR intervals, so MSE, entropy and DFA are mostly NaN. | Features return NaN rather than a made-up value. |
| CVAE | Raw features had very different scales, and held-out R² was about −0.01. | Standardised the inputs (R² → 0.86). |
| Transformer | A fresh head outputs ~0 and couldn't reach the ~280-day targets. | Scaled the targets. |
| Data | ADFECGDB GA labels are synthetic. GA ranges are 21–27 wk vs 38–41 wk, with nothing between. | Flagged every affected number as not reportable. |
| Result | Within NInFEA no feature correlates with GA. Pooled MAE 22.6 d (r 0.56) is worse than predicting the mean (20.6 d) and mostly reflects the gap between the two cohorts. | Reported honestly as no within-cohort age signal. |

---

## 3. Next phase: what to improve

**From the supervisor's paper** (conclusion and limitations; the first two items are my inference):
- **Expand dataset diversity.** The paper lists generalisation to diverse fetal ECG patterns as an open challenge. For us that means real GA labels for ADFECGDB, and ideally data in the 28–37 wk gap. Ask the supervisor whether any suitable dataset exists.
- **Attention mechanisms and hybrid deep architectures.** The paper's extractor is a plain 3-layer 1D-CNN with 152,065 parameters. Our hybrid PINN + FastICA extractor goes in that direction. A supervised extractor, such as the paper's CNN or the Wahbah BiLSTM, is the natural comparison arm; neither is implemented in our repo yet.
- **Reproduce the paper's 1D-CNN baseline** (trained on the scalp lead) as the reference point for any extraction claim.
- **Strengthen the quality gate.** The paper uses an Earth Mover's Distance (EMD) check to classify records as high or low quality. Our gate is rate- and beat-based. On NInFEA it keeps 98% of right-rate windows, but only 63% of what it passes are right.
- **Real-time and edge deployment.** The paper's last future-work item. Not started.

**From our own results:**
1. Get real ADFECGDB GA labels. Nothing age-related is reportable without them.
2. Get the PINN coordinate scheme signed off, or replace it. It is currently an unsigned assumption.
3. Use longer windows or recordings so the entropy, DFA and LF/HF features are populated.
4. Re-test the age signal on cohorts with a wider GA span before investing in a bigger Transformer.

**Caution:** the paper's Table I describes "NIFEADB" (26 recordings, arrhythmia labels). We use NInFEA (60 recordings), a different database. Be ready to explain that difference if a panel member compares the two.

---

## 4. Plain-language explanations

### 4.1 MEMD in simple words

**The problem:** the abdominal signal is a mix of many wiggles at different speeds. These include slow baseline drift from breathing, motion bumps, fast muscle noise, the maternal heartbeat and the faint fetal heartbeat.

**IMFs:** MEMD splits the signal into layers called IMFs (Intrinsic Mode Functions). IMF 1 is the fastest wiggle and the last IMFs are the slowest drift. Adding all the layers back together gives the original signal.

**"All 4 channels jointly":** the 4 abdominal electrodes are split together, so layer *k* means the same speed in every channel. If each channel were split alone, "layer 3" could mean different speeds in different channels.

**"64 directions":** it looks at the 4-channel signal from 64 different angles to find its slow and fast trends. More directions give a more accurate split but are slower.

**Scoring and selecting layers:** the fetal heartbeat's energy sits roughly at 20–70 Hz. Each layer gets a score for how much of its power falls in that band, multiplied by how well it matches a reference signal. The best-scoring layers are added back together and the rest, such as drift and very fast noise, are thrown away.

**"Removes drift, motion and muscle artefact only":** MEMD is a cleaner, not a separator. The maternal heartbeat is still in its output. Separating mother from baby is the next stage's job.

### 4.2 What the CVAE does

**Input:** one vector of 12 numbers per 5 s window. These are the biomarkers (QTc σ, P-dispersion, QRS stats, entropy and so on). Missing values (NaN) are filled with the median first.

**What it does:** the CVAE (contrastive variational autoencoder) is a neural network with two halves.
- **The encoder** squeezes the 12 numbers into a 64-number code called the latent `z`.
- **The decoder** tries to rebuild the original 12 numbers from that code. If it can, the code kept the important information.

**Output:** `z`, a 64-number vector for each window. The decoder's reconstruction is only used for training.

**How it transforms the data:** the 12 features are first standardised (mean 0, spread 1). The code is built so that:
- windows from the same baby, or with similar gestational age (within ±7 days), end up close together in latent space. This is the contrastive part, with temperature 0.07;
- windows from different babies are pushed apart;
- the codes stay smooth and well spread, which is the "variational" part and stops the model memorising.

GA is never given to the model, so it can't just copy the label.

**Why use it:**
- With only about 64 recordings, a Transformer on raw features overfits. The CVAE gives it a cleaner, standardised, structured input.
- It replaced the WGAN, which was unstable at this size. We can now sample neighbours of real latent codes to create extra training data.
- It handles the wildly different feature scales.

**Honest caveat:** 64 numbers from 12 inputs is an expansion, not a compression. High reconstruction accuracy (R² 0.86) only shows the autoencoder works. So far the latent has not been shown to carry gestational-age information.

### 4.3 What (W, 5000) means

- **5000** is the number of samples in one window: 5 seconds × 1000 samples per second.
- **W** is the number of windows the recording is cut into.

The PINN output has shape (W, 5000): one cleaned-up fetal ECG trace of 5000 samples per window, with W windows stacked. It has one row per window and one channel, because the 4 abdominal channels have been combined into a single fetal signal.

Example: ADFECGDB r08 is 5 minutes long. Cut into 5 s windows after edge trimming, it gives W = 59 (the figure for that record shows 59 windows). So its output is (59, 5000), and the 12 features and the CVAE then work on each of those 59 windows.

The same W later becomes the sequence length for the Transformer. It reads the 59 latent vectors of 64 numbers each, in crops of 16, and outputs one Cardiac Age for the whole recording.
