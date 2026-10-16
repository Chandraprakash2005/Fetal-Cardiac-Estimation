# Pipeline-Stage Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/fcac/` the single, authoritative per-pipeline-stage package by porting the
working ideas from the untracked top-level `cvae/` and `transformer/` folders into
`src/fcac/cvae/` and `src/fcac/transformer/`, then deleting those top-level folders and the
legacy flat `src/preprocessing|extraction|model|features|evaluation/` tree.

**Architecture:** `src/fcac/<stage>/` keeps its existing convention — every model/train/loss
function takes the frozen `Config` object from `fcac.config.load_config()`, never raw
hyperparameters (CLAUDE.md §6) — while absorbing the richer, working logic (LayerNorm/GELU
encoder blocks + separate InfoNCE projection head, GA-window positive pairing, KL warmup,
CLS-token transformer pooling with padding-mask support, variable-length recording batching,
LOSOCV/zero-shot wiring, Mahalanobis OOD gating) that currently only exists in the top-level
folders. Shared cross-stage logic (OOD detection, bootstrap CI, LOSOCV folds, ΔAge) goes in
`src/fcac/evaluation/`, reused by both `cvae` and `transformer` rather than duplicated.

**Tech Stack:** Python 3.11, PyTorch 2.3.1, numpy, pytest. No new dependencies.

**Spec:** `CLAUDE.md` (this repo's single source of truth) — particularly §2 "CVAE",
§2 "Transformer", §3 "GA coverage gap", §4 (repo layout), §5 (data contracts), §6 (coding
conventions), §9 (legacy code note).

## Global Constraints

- Every model/train/loss function reads hyperparameters from the `Config` object, never a
  hardcoded number or a bare kwarg default that matters (CLAUDE.md §6).
- CVAE latent dim stays 64, InfoNCE `tau` stays 0.07, GA is never fed into the CVAE as input
  (CLAUDE.md §2 "CVAE" — all LOCKED).
- Transformer stays 4 heads, `d_model=256`, `d_k=64`, 3 encoder layers, MAE loss, AdamW
  (CLAUDE.md §2 "Transformer" — LOCKED).
- No WGAN scaffolding of any kind (CLAUDE.md §2 "WGAN" — LOCKED, dropped).
- GA/CA are stored and computed in days everywhere internally (CLAUDE.md §5).
- `pytest` (all of `tests/`) must pass after every task, not just at the end.
- Do not touch CinC2013 references, `fetal-cardiac-ageing-clock_version1.py`, or reintroduce
  the dropped dataset anywhere — out of scope for this plan (left undecided by the user).
- Do not fabricate GA labels, MEMD source, or PINN coordinate scheme — those stay `null`/TODO
  exactly as they are; this plan only touches CVAE/transformer stages and the legacy tree.

## Review Focus

- **Batch size larger than the dataset in `train_cvae`/`train_transformer_on_recordings`:**
  the smoke tests and placeholder scripts run with small `n` (8–64 samples); `DataLoader`
  must not raise or silently drop the whole dataset when `batch_size > len(dataset)` —
  every task that builds a `DataLoader` clamps `batch_size = min(cfg.<stage>.batch_size, len(dataset))`.
- **No positive pairs in an InfoNCE batch:** `info_nce_loss` must return a finite `0`-valued
  loss instead of `NaN`/exception when a sampled batch happens to contain no positive pairs
  (covered in Task 3).
- **Single-recording LOSOCV fold:** `losocv_adfecgdb` must produce exactly one prediction per
  held-out subject even when that subject's `test_idx` has only one recording (covered in
  Task 8).
- **A recording longer than `max_seq_len`:** `predict_recordings` must chunk it into
  `crop_len`-sized pieces and average, not feed the whole sequence through the transformer in
  one shot and hit the `ValueError` in `CardiacAgeTransformer.forward` (covered in Task 8).
- **Every latent in an OOD batch identical (zero variance):** `fit_ood`'s covariance ridge
  term must keep `torch.linalg.inv` from raising on a singular matrix (covered in Task 1).

---

## Task 1: Shared Mahalanobis OOD detector

**Files:**
- Create: `src/fcac/evaluation/ood.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: nothing new (plain `torch`/`numpy`).
- Produces: `OODStats` (namedtuple: `mean`, `inv_cov`, `threshold`), `fit_ood(z_train, percentile=99.0, ridge=1e-3) -> OODStats`, `mahalanobis_distance(z, stats) -> torch.Tensor`, `is_out_of_distribution(z, stats) -> torch.Tensor`. Tasks 4 and 8 import all three.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation.py`:

```python
def test_fit_ood_flags_far_points_and_keeps_near_points_in_distribution():
    from fcac.evaluation.ood import fit_ood, is_out_of_distribution

    rng = np.random.default_rng(0)
    train = torch.as_tensor(rng.normal(0, 1, size=(200, 4)), dtype=torch.float32)
    stats = fit_ood(train, percentile=95.0)

    near = train[:5]
    far = torch.full((3, 4), 50.0)
    assert not is_out_of_distribution(near, stats).any()
    assert is_out_of_distribution(far, stats).all()


def test_fit_ood_handles_zero_variance_dimension():
    from fcac.evaluation.ood import fit_ood

    z = torch.zeros(20, 3)
    z[:, 0] = torch.randn(20)
    stats = fit_ood(z)  # must not raise on the singular covariance
    assert torch.isfinite(stats.mean).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_evaluation.py -k ood -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fcac.evaluation.ood'`

- [ ] **Step 3: Implement**

Create `src/fcac/evaluation/ood.py`:

```python
"""src/fcac/evaluation/ood.py

Mahalanobis out-of-distribution (OOD) detector for latent-space representations.
Used to flag NInFEA zero-shot windows/recordings that fall far outside the ADFECGDB
training distribution in CVAE latent space (CLAUDE.md §3 "GA coverage gap": training
is 38-41 wks, validation is 21-27 wks -- downstream code must never assume overlapping
GA ranges, so an explicit OOD flag on the zero-shot side is required, not optional).

Implements: CLAUDE.md §3 "GA coverage gap" note; additive diagnostic, not a locked
architecture change.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class OODStats(NamedTuple):
    mean: torch.Tensor      # (D,)
    inv_cov: torch.Tensor   # (D, D)
    threshold: float        # percentile of training distances


def fit_ood(z_train: torch.Tensor, percentile: float = 99.0, ridge: float = 1e-3) -> OODStats:
    """Fit a Mahalanobis OOD detector on training latents.

    z_train    : (N, D) training-distribution latent vectors (e.g. ADFECGDB CVAE means).
    percentile : threshold is this percentile of the training distances themselves.
    ridge      : diagonal regularisation so inversion is stable even when a latent
                 dimension has near-zero variance in the training set.
    """
    mean = z_train.mean(0)
    cov = torch.cov(z_train.T) + ridge * torch.eye(z_train.shape[1], dtype=z_train.dtype)
    inv_cov = torch.linalg.inv(cov)
    provisional = OODStats(mean=mean, inv_cov=inv_cov, threshold=0.0)
    train_distances = mahalanobis_distance(z_train, provisional)
    threshold = float(np.percentile(train_distances.numpy(), percentile))
    return OODStats(mean=mean, inv_cov=inv_cov, threshold=threshold)


def mahalanobis_distance(z: torch.Tensor, stats: OODStats) -> torch.Tensor:
    """z : (N, D) -> (N,) Mahalanobis distance from the fitted training distribution."""
    diff = z - stats.mean
    return torch.sqrt(((diff @ stats.inv_cov) * diff).sum(dim=-1).clamp_min(0))


def is_out_of_distribution(z: torch.Tensor, stats: OODStats) -> torch.Tensor:
    """z : (N, D) -> (N,) bool, True where the distance exceeds the fitted threshold."""
    return mahalanobis_distance(z, stats) > stats.threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluation.py -v`
Expected: PASS (all tests in the file, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/fcac/evaluation/ood.py tests/test_evaluation.py
git commit -m "feat(evaluation): add shared Mahalanobis OOD detector"
```

---

## Task 2: CVAE model — projection head + LayerNorm/GELU blocks

**Files:**
- Modify: `src/fcac/cvae/model.py`
- Modify: `configs/default.yaml` (cvae section)
- Modify: `tests/test_cvae.py`

**Interfaces:**
- Consumes: `fcac.config.Config`.
- Produces: `ContrastiveVAE(cfg).forward(x) -> dict` with keys `x_hat (N,12)`, `mu (N,64)`,
  `logvar (N,64)`, `z (N,64)`, `p (N, proj_dim)` (L2-normalised). `model.encoder`, `model.decoder`
  keep their names (Task 5's `augment.py` calls `model.decoder` directly, unaffected). This is a
  **breaking change** to `forward`'s return type (tuple → dict) — Task 3/4 and their tests are
  updated in the same commit sequence as this task so the suite never sits broken between tasks.

- [ ] **Step 1: Update `configs/default.yaml`'s `cvae:` block**

Add `dropout` and `proj_dim` fields (used by Step 3's model code):

```yaml
cvae:
  input_dim: 12
  latent_dim: 64
  encoder_hidden: [128, 64]
  decoder_hidden: [64, 128]
  proj_dim: 64                     # InfoNCE projection head output dim
  dropout: 0.2
  contrastive:
    loss: infonce
    temperature: 0.07
    positive_rule: same_subject_or_ga_within_window
  ga_label_as_input: false          # LOCKED: never true (circular)
  optimizer: adamw
  learning_rate: 1.0e-3
  epochs: 200
  batch_size: 64
```

(Leave the rest of the file as-is; `weight_decay`/`grad_clip`/`beta_kl`/`kl_warmup_epochs`/
`ga_window_days` are added in Task 3/4, not this step — keeps each task's config diff scoped
to what that task's code actually reads.)

- [ ] **Step 2: Rewrite the forward-shape test for the new dict return type**

Replace `test_cvae_forward_shapes` in `tests/test_cvae.py`:

```python
def test_cvae_forward_shapes():
    cfg = load_config()
    model = ContrastiveVAE(cfg)
    x = torch.randn(16, cfg.cvae.input_dim)
    out = model(x)
    assert out["x_hat"].shape == x.shape
    assert out["mu"].shape == (16, cfg.cvae.latent_dim)
    assert out["logvar"].shape == (16, cfg.cvae.latent_dim)
    assert out["z"].shape == (16, cfg.cvae.latent_dim)
    assert out["p"].shape == (16, cfg.cvae.proj_dim)
    assert torch.allclose(out["p"].norm(dim=-1), torch.ones(16), atol=1e-5)


def test_cvae_eval_mode_is_deterministic():
    cfg = load_config()
    model = ContrastiveVAE(cfg)
    model.eval()
    x = torch.randn(4, cfg.cvae.input_dim)
    out1, out2 = model(x), model(x)
    assert torch.equal(out1["z"], out2["z"])  # eval: z = mu, no sampling noise
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cvae.py -k "forward_shapes or eval_mode" -v`
Expected: FAIL — `out["x_hat"]` raises because `model(x)` still returns a tuple.

- [ ] **Step 4: Implement**

Replace `src/fcac/cvae/model.py`:

```python
"""src/fcac/cvae/model.py

Contrastive VAE (CLAUDE.md §2 "CVAE" -- Contrastive, not Conditional): encodes a
12-feature window vector to a 64-dim latent, trained with reconstruction + KL +
InfoNCE contrastive loss (cvae/losses.py). GA is never fed as input (circular).

The InfoNCE term operates on a separate projection of `z` (`p`), not `z` itself --
optimising the contrastive objective directly against the latent used for
reconstruction and downstream (CVAE -> Transformer) tasks distorts both; a small
projection head is the standard SimCLR-style fix and costs nothing else.

Implements: CLAUDE.md §2 "CVAE" architecture.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from fcac.config import Config


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
    )


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        dims = [cfg.cvae.input_dim] + list(cfg.cvae.encoder_hidden)
        self.backbone = nn.Sequential(
            *[_mlp_block(a, b, cfg.cvae.dropout) for a, b in zip(dims[:-1], dims[1:])]
        )
        self.mu = nn.Linear(dims[-1], cfg.cvae.latent_dim)
        self.logvar = nn.Linear(dims[-1], cfg.cvae.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        dims = [cfg.cvae.latent_dim] + list(cfg.cvae.decoder_hidden)
        blocks = [_mlp_block(a, b, cfg.cvae.dropout) for a, b in zip(dims[:-1], dims[1:])]
        self.net = nn.Sequential(*blocks, nn.Linear(dims[-1], cfg.cvae.input_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Projector(nn.Module):
    """Maps the VAE latent to the space the InfoNCE loss actually operates in."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.cvae.latent_dim, cfg.cvae.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.cvae.latent_dim, cfg.cvae.proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class ContrastiveVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.projector = Projector(cfg)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x : (N, 12) -> {x_hat (N,12), mu/logvar/z (N,64), p (N, proj_dim)}."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return {
            "x_hat": self.decoder(z),
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "p": self.projector(z),
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cvae.py -k "forward_shapes or eval_mode" -v`
Expected: PASS. (The rest of `tests/test_cvae.py` will still fail until Tasks 3–4 update
`losses.py`/`train.py`/`augment.py` — that's expected and fixed in those tasks, not this one.)

- [ ] **Step 6: Commit**

```bash
git add src/fcac/cvae/model.py configs/default.yaml tests/test_cvae.py
git commit -m "feat(cvae): add InfoNCE projection head, LayerNorm/GELU encoder blocks"
```

---

## Task 3: CVAE losses — GA-window positives, KL warmup, projection-based InfoNCE

**Files:**
- Modify: `src/fcac/cvae/losses.py`
- Modify: `configs/default.yaml` (cvae section)
- Modify: `tests/test_cvae.py`

**Interfaces:**
- Consumes: `ContrastiveVAE.forward()`'s dict output (Task 2).
- Produces: `reconstruction_loss(x_hat, x)`, `kl_divergence(mu, logvar)`,
  `kl_warmup_beta(epoch, cfg) -> float`, `info_nce_loss(p, positive_mask, temperature) -> Tensor`,
  `hybrid_loss(outputs: dict, x, positive_mask, epoch, cfg) -> Tensor`. Task 4 imports
  `hybrid_loss` and `kl_warmup_beta`.

- [ ] **Step 1: Add `ga_window_days`, `beta_kl`, `kl_warmup_epochs` to `configs/default.yaml`'s `cvae:` block**

```yaml
cvae:
  input_dim: 12
  latent_dim: 64
  encoder_hidden: [128, 64]
  decoder_hidden: [64, 128]
  proj_dim: 64
  dropout: 0.2
  beta_kl: 0.1                     # small + warmup to avoid posterior collapse
  kl_warmup_epochs: 20
  contrastive:
    loss: infonce
    temperature: 0.07
    positive_rule: same_subject_or_ga_within_window
    ga_window_days: 7.0            # resolves CLAUDE.md open question: "same-GA" = +/-7 days
  ga_label_as_input: false          # LOCKED: never true (circular)
  optimizer: adamw
  learning_rate: 1.0e-3
  epochs: 200
  batch_size: 64
```

- [ ] **Step 2: Rewrite the loss tests in `tests/test_cvae.py`**

Replace `test_losses_are_finite`:

```python
def test_losses_are_finite():
    cfg = load_config()
    model = ContrastiveVAE(cfg)
    x = torch.randn(8, cfg.cvae.input_dim)
    out = model(x)

    assert torch.isfinite(reconstruction_loss(out["x_hat"], x))
    assert torch.isfinite(kl_divergence(out["mu"], out["logvar"]))

    positive_mask = torch.eye(8, dtype=torch.bool).roll(1, dims=1)
    assert torch.isfinite(
        info_nce_loss(out["p"], positive_mask, cfg.cvae.contrastive.temperature)
    )
    assert torch.isfinite(hybrid_loss(out, x, positive_mask, epoch=0, cfg=cfg))


def test_info_nce_loss_is_zero_not_nan_with_no_positives():
    p = F.normalize(torch.randn(6, 8), dim=-1)
    no_positives = torch.zeros(6, 6, dtype=torch.bool)
    loss = info_nce_loss(p, no_positives, temperature=0.07)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_kl_warmup_beta_ramps_linearly_then_holds():
    cfg = load_config()
    assert kl_warmup_beta(0, cfg) == pytest.approx(0.0)
    half = cfg.cvae.kl_warmup_epochs // 2
    assert kl_warmup_beta(half, cfg) == pytest.approx(cfg.cvae.beta_kl * half / cfg.cvae.kl_warmup_epochs)
    assert kl_warmup_beta(cfg.cvae.kl_warmup_epochs * 10, cfg) == pytest.approx(cfg.cvae.beta_kl)
```

Update the imports at the top of `tests/test_cvae.py`:

```python
import pytest
import torch
import torch.nn.functional as F

from fcac.config import Config, load_config
from fcac.cvae.augment import sample_latent_neighbours
from fcac.cvae.losses import (
    hybrid_loss,
    info_nce_loss,
    kl_divergence,
    kl_warmup_beta,
    reconstruction_loss,
)
from fcac.cvae.model import ContrastiveVAE
from fcac.cvae.train import build_positive_mask, train_cvae
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cvae.py -k "losses_are_finite or no_positives or kl_warmup" -v`
Expected: FAIL — `kl_warmup_beta` doesn't exist yet, `hybrid_loss` still takes the old
positional signature.

- [ ] **Step 4: Implement**

Replace `src/fcac/cvae/losses.py`:

```python
"""src/fcac/cvae/losses.py

Contrastive VAE loss: reconstruction + KL + InfoNCE (CLAUDE.md §2 "CVAE", tau=0.07).
Positive pairs are same-subject windows, or windows within `ga_window_days` of each
other -- CLAUDE.md leaves "same-GA" undefined, and a hard equality match discards
almost every pair in practice (GA is effectively continuous per-window), so a
day-tolerance window is the only version of "same-GA" that produces any positives at
all. InfoNCE runs on the encoder's projection head output (`p`), not the VAE latent
(`z`) itself, so the contrastive objective doesn't fight the reconstruction /
downstream-task objective for the same embedding space.

TODO(tejasree): CLAUDE.md does not lock a relative weighting between recon and KL
(unlike the PINN's explicit 0.7/0.3) -- recon weight is implicitly 1.0 below, KL is
warmed up to `beta_kl` (0.1) over `kl_warmup_epochs`, InfoNCE weight is implicitly 1.0.
Flag to the supervisor if a different weighting is wanted.

Implements: CLAUDE.md §2 "CVAE" loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fcac.config import Config


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def kl_warmup_beta(epoch: int, cfg: Config) -> float:
    """Linear KL warmup 0 -> cfg.cvae.beta_kl over cfg.cvae.kl_warmup_epochs, so the
    reconstruction term gets to shape the latent before the KL term starts pulling it
    toward the prior (avoids posterior collapse)."""
    warmup_epochs = max(1, cfg.cvae.kl_warmup_epochs)
    return cfg.cvae.beta_kl * min(1.0, epoch / warmup_epochs)


def info_nce_loss(
    p: torch.Tensor, positive_mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Multi-positive InfoNCE over a batch of projections.

    p             : (N, D), already L2-normalised (Projector normalises internally).
    positive_mask : (N, N) bool; positive_mask[i, j] True iff j is a positive for i;
                    diagonal is ignored regardless of its value.
    temperature   : tau (0.07, CLAUDE.md §2).
    """
    n = p.shape[0]
    sim = p @ p.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=p.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)

    positive_mask = positive_mask & ~self_mask
    n_positives = positive_mask.sum(dim=-1).clamp(min=1)
    # log_prob is -inf on the self-diagonal; log_prob * positive_mask would compute
    # -inf * False = NaN there (IEEE754) even though masked out, so use torch.where.
    masked_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    loss_per_sample = -masked_log_prob.sum(dim=-1) / n_positives

    has_positive = positive_mask.sum(dim=-1) > 0
    if not torch.any(has_positive):
        return p.sum() * 0.0
    return loss_per_sample[has_positive].mean()


def hybrid_loss(
    outputs: dict,
    x: torch.Tensor,
    positive_mask: torch.Tensor,
    epoch: int,
    cfg: Config,
) -> torch.Tensor:
    """outputs is a ContrastiveVAE.forward() dict: {x_hat, mu, logvar, z, p}."""
    recon = reconstruction_loss(outputs["x_hat"], x)
    kl = kl_divergence(outputs["mu"], outputs["logvar"])
    contrastive = info_nce_loss(outputs["p"], positive_mask, cfg.cvae.contrastive.temperature)
    beta = kl_warmup_beta(epoch, cfg)
    return recon + beta * kl + contrastive
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cvae.py -k "losses_are_finite or no_positives or kl_warmup" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/fcac/cvae/losses.py configs/default.yaml tests/test_cvae.py
git commit -m "feat(cvae): GA-window positive pairing, KL warmup, projection-based InfoNCE"
```

---

## Task 4: CVAE train loop (AdamW + mini-batches) + diagnostics module

**Files:**
- Modify: `src/fcac/cvae/train.py`
- Create: `src/fcac/cvae/diagnostics.py`
- Modify: `configs/default.yaml` (cvae section)
- Modify: `tests/test_cvae.py`

**Interfaces:**
- Consumes: `hybrid_loss` (Task 3), `ContrastiveVAE` (Task 2), `fcac.seed.set_seed`.
- Produces: `build_positive_mask(subject_ids, ga_days, ga_window_days) -> Tensor`,
  `train_cvae(cfg, features, subject_ids, ga_days) -> ContrastiveVAE` (signature unchanged from
  before this plan — `scripts/06_train_cvae.py` keeps calling it as-is),
  `spearman_corr`, `latent_distance_vs_ga_gap`, `ridge_regression_loro_mae_days` in
  `diagnostics.py` (pure functions, no `Config` needed — they operate on already-computed
  latents/targets).

- [ ] **Step 1: Add `weight_decay`, `grad_clip` to `configs/default.yaml`'s `cvae:` block**

```yaml
  optimizer: adamw
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  grad_clip: 1.0
  epochs: 200
  batch_size: 64
```

- [ ] **Step 2: Update `train_cvae`/`build_positive_mask` tests and add diagnostics tests**

Replace `test_build_positive_mask_same_subject_or_ga` and
`test_train_cvae_runs_end_to_end_on_random_tensors` in `tests/test_cvae.py`, and append the
diagnostics tests:

```python
def test_build_positive_mask_same_subject_or_ga_within_window():
    subject_ids = np.array(["a", "a", "b", "b"])
    ga_days = np.array([280.0, 284.0, 280.0, 300.0])
    mask = build_positive_mask(subject_ids, ga_days, ga_window_days=7.0)
    assert bool(mask[0, 1])  # same subject
    assert bool(mask[0, 2])  # different subject, |ΔGA|=0 <= 7
    assert not bool(mask[1, 3])  # different subject, |ΔGA|=20 > 7


def test_train_cvae_runs_end_to_end_on_random_tensors():
    cfg = _with_epochs(load_config(), epochs=2)
    n = 16
    features = torch.randn(n, cfg.cvae.input_dim)
    subject_ids = np.array([f"s{i % 4}" for i in range(n)])
    ga_days = np.full(n, 280.0)

    model = train_cvae(cfg, features, subject_ids, ga_days)
    out = model(features)
    assert torch.isfinite(out["x_hat"]).all()


def test_spearman_corr_perfect_rank_agreement():
    from fcac.cvae.diagnostics import spearman_corr

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert spearman_corr(a, b) == pytest.approx(1.0)


def test_latent_distance_vs_ga_gap_nan_when_all_same_ga():
    from fcac.cvae.diagnostics import latent_distance_vs_ga_gap

    z = torch.randn(10, 4)
    ga = torch.full((10,), 280.0)
    result = latent_distance_vs_ga_gap(z, ga)
    assert result != result  # NaN != NaN


def test_ridge_regression_loro_mae_days_recovers_linear_signal():
    from fcac.cvae.diagnostics import ridge_regression_loro_mae_days

    torch.manual_seed(0)
    n_per_subject, n_subjects = 20, 4
    subject_idx = torch.arange(n_subjects).repeat_interleave(n_per_subject)
    ga = 280.0 + subject_idx.float() * 5.0 + torch.randn(n_per_subject * n_subjects) * 0.01
    z = torch.stack([ga, torch.randn(n_per_subject * n_subjects)], dim=1)

    mae_days = ridge_regression_loro_mae_days(z, ga, subject_idx)
    assert mae_days < 5.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cvae.py -v`
Expected: FAIL — `build_positive_mask` doesn't accept `ga_window_days` yet, `train_cvae`'s
model output is still consumed as a tuple internally, `fcac.cvae.diagnostics` doesn't exist.

- [ ] **Step 4: Implement**

Replace `src/fcac/cvae/train.py`:

```python
"""src/fcac/cvae/train.py

Training loop for the Contrastive VAE (CLAUDE.md §2 "CVAE"). Uses AdamW with a cosine
learning-rate schedule and gradient clipping over mini-batches, rather than one
full-batch Adam step per epoch -- with as few as ~90 recordings' worth of windows,
mini-batching is what makes the InfoNCE term see different positive/negative
arrangements across epochs instead of the same single batch every time.

Implements: CLAUDE.md §8 item 6 (train/infer loops that run end-to-end on random tensors).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fcac.config import Config
from fcac.cvae.losses import hybrid_loss
from fcac.cvae.model import ContrastiveVAE
from fcac.seed import set_seed

logger = logging.getLogger(__name__)


def build_positive_mask(
    subject_ids: np.ndarray, ga_days: np.ndarray, ga_window_days: float
) -> torch.Tensor:
    """positive_mask[i, j] = True iff window j is a positive for window i: same
    subject, OR GA within `ga_window_days` days of each other (CLAUDE.md §2 "CVAE"
    positive_rule).

    subject_ids : (N,) array of hashable ids.
    ga_days     : (N,) array of GA in days.
    """
    same_subject = subject_ids[:, None] == subject_ids[None, :]
    ga = torch.as_tensor(ga_days, dtype=torch.float32)
    close_ga = (ga[:, None] - ga[None, :]).abs() <= ga_window_days
    return torch.from_numpy(same_subject) | close_ga


def train_cvae(
    cfg: Config, features: torch.Tensor, subject_ids: np.ndarray, ga_days: np.ndarray
) -> ContrastiveVAE:
    """features : (N, 12) feature vectors (FeatureVector.f, CLAUDE.md §5)."""
    set_seed(cfg.seed)
    model = ContrastiveVAE(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.cvae.learning_rate, weight_decay=cfg.cvae.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.cvae.epochs)

    positive_mask_full = build_positive_mask(
        subject_ids, ga_days, cfg.cvae.contrastive.ga_window_days
    )
    batch_size = min(cfg.cvae.batch_size, len(features))
    dataset = TensorDataset(features, torch.arange(len(features)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(cfg.cvae.epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, idx_batch in loader:
            positive_mask = positive_mask_full[idx_batch][:, idx_batch]
            outputs = model(x_batch)
            loss = hybrid_loss(outputs, x_batch, positive_mask, epoch, cfg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.cvae.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % max(1, cfg.cvae.epochs // 10) == 0:
            logger.info("epoch %d: loss=%.6f", epoch, epoch_loss / len(loader))

    return model
```

Create `src/fcac/cvae/diagnostics.py`:

```python
"""src/fcac/cvae/diagnostics.py

Sanity checks for a trained ContrastiveVAE's latent space, run after training (not
part of the loss): does latent distance track |ΔGA| at all (it should, if the
contrastive objective is doing anything), and can a trivial ridge regression from the
latent recover GA under leave-one-recording-out (a floor the Transformer stage should
comfortably beat).

Implements: CLAUDE.md §2 "CVAE" -- diagnostic support for reports/engineering_log.md
decisions, not a locked architecture requirement.
"""

from __future__ import annotations

import numpy as np
import torch


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency."""
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def latent_distance_vs_ga_gap(
    z: torch.Tensor, ga_days: torch.Tensor, max_pairs: int = 2000, seed: int = 0
) -> float:
    """Spearman correlation between pairwise latent distance and pairwise |ΔGA| over
    a random sample of window pairs. Should be > 0 if the contrastive loss is
    separating windows by GA at all; NaN if every sampled window shares the same GA."""
    generator = torch.Generator().manual_seed(seed)
    if len(z) > max_pairs:
        idx = torch.randperm(len(z), generator=generator)[:max_pairs]
        z, ga_days = z[idx], ga_days[idx]
    i, j = torch.triu_indices(len(z), len(z), offset=1)
    latent_dist = torch.cdist(z, z)[i, j].numpy()
    ga_gap = (ga_days[:, None] - ga_days[None, :]).abs()[i, j].numpy()
    if ga_gap.std() == 0:
        return float("nan")
    return spearman_corr(latent_dist, ga_gap)


def ridge_regression_loro_mae_days(
    z: torch.Tensor, ga_days: torch.Tensor, subject_idx: torch.Tensor, alpha: float = 1.0
) -> float:
    """Leave-one-recording-out ridge regression z -> GA. Returns MAE in days -- a
    linear floor the Transformer stage should beat by a wide margin."""
    errors = []
    for subject in torch.unique(subject_idx):
        test_mask, train_mask = subject_idx == subject, subject_idx != subject
        design = torch.cat([z[train_mask], torch.ones(train_mask.sum(), 1)], dim=1)
        weights = torch.linalg.solve(
            design.T @ design + alpha * torch.eye(design.shape[1]),
            design.T @ ga_days[train_mask],
        )
        test_design = torch.cat([z[test_mask], torch.ones(test_mask.sum(), 1)], dim=1)
        prediction = (test_design @ weights).mean()
        errors.append((prediction - ga_days[test_mask][0]).abs().item())
    return float(np.mean(errors))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cvae.py -v`
Expected: PASS (all of `test_cvae.py`, including `test_sample_latent_neighbours_shape`, which
is untouched since `augment.py` calls `model.decoder` directly and doesn't depend on the
`forward()` dict change).

- [ ] **Step 6: Commit**

```bash
git add src/fcac/cvae/train.py src/fcac/cvae/diagnostics.py configs/default.yaml tests/test_cvae.py
git commit -m "feat(cvae): AdamW mini-batch training loop, latent diagnostics module"
```

---

## Task 5: Retire the top-level `cvae/` folder

**Files:**
- Delete: `cvae/` (entire directory — `model.py`, `train.py`, `smoke_test.py`, `utils.py`,
  `data.py`, `config.yaml`)
- Modify: `reports/engineering_log.md`

**Interfaces:** none — this task removes code, it doesn't add any.

- [ ] **Step 1: Confirm nothing else references the top-level `cvae/` package**

Run: `grep -rn "^from data import\|^from model import\|^from utils import" cvae/ 2>/dev/null; grep -rln "cvae/train\|cvae/smoke_test\|cvae\.config\.yaml" --include="*.py" --include="*.md" --include="*.sh" . | grep -v "^\./cvae/\|^\./src/fcac/cvae\|^\./\.venv"`
Expected: the first command lists `cvae/train.py` and `cvae/smoke_test.py` (their own
self-imports, about to be deleted); the second command prints nothing (no other file in the
repo references the top-level `cvae/` scripts by path).

- [ ] **Step 2: Delete the directory**

```bash
git rm -r cvae/
```

- [ ] **Step 3: Verify the full suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (this directory was never imported by `tests/`, `src/fcac/`, or `scripts/`).

- [ ] **Step 4: Log the decision**

Append to `reports/engineering_log.md`:

```markdown
- 2026-09-28: Ported the top-level `cvae/` folder's working logic (InfoNCE projection
  head, GA-window positive pairing, KL warmup, AdamW mini-batch training, Mahalanobis
  OOD stats, latent diagnostics) into `src/fcac/cvae/`, then deleted the top-level
  folder -- `src/fcac/` is now the only place CVAE code lives. -> CLAUDE.md §4, §7.
```

- [ ] **Step 5: Commit**

```bash
git add reports/engineering_log.md
git commit -m "chore(cvae): remove top-level cvae/ folder now ported into src/fcac/cvae/"
```

---

## Task 6: Transformer model — CLS-token pooling + padding mask

**Files:**
- Modify: `src/fcac/transformer/model.py`
- Modify: `configs/default.yaml` (transformer section)
- Modify: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `fcac.config.Config`.
- Produces: `CardiacAgeTransformer(cfg).forward(z_seq, pad_mask=None) -> Tensor (N,)`.
  `pad_mask` is optional and backward-compatible — existing calls without it are unaffected.

- [ ] **Step 1: Add `d_ff`, `dropout`, `max_seq_len`, `head_hidden`, `head_dropout` to `configs/default.yaml`'s `transformer:` block**

```yaml
transformer:
  n_heads: 4
  d_model: 256
  d_k: 64
  n_layers: 3
  d_ff: 1024
  dropout: 0.2
  max_seq_len: 64
  head_hidden: 128
  head_dropout: 0.1
  loss: mae
  optimizer: adamw
  learning_rate: 1.0e-4
  weight_decay: 1.0e-2
  epochs: 200
  batch_size: 16
```

- [ ] **Step 2: Add padding-mask and CLS-pooling tests to `tests/test_transformer.py`**

Append:

```python
def test_transformer_forward_with_padding_mask():
    cfg = load_config()
    model = CardiacAgeTransformer(cfg)
    z_seq = torch.randn(3, 10, cfg.cvae.latent_dim)
    pad_mask = torch.zeros(3, 10, dtype=torch.bool)
    pad_mask[:, 7:] = True  # last 3 positions are padding
    ca = model(z_seq, pad_mask)
    assert ca.shape == (3,)
    assert torch.isfinite(ca).all()


def test_transformer_rejects_sequence_longer_than_max_seq_len():
    cfg = load_config()
    model = CardiacAgeTransformer(cfg)
    z_seq = torch.randn(1, cfg.transformer.max_seq_len + 1, cfg.cvae.latent_dim)
    try:
        model(z_seq)
        assert False, "expected ValueError for sequence longer than max_seq_len"
    except ValueError:
        pass
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transformer.py -k "padding_mask or longer_than" -v`
Expected: FAIL — `CardiacAgeTransformer.forward` doesn't accept a second argument yet.

- [ ] **Step 4: Implement**

Replace `src/fcac/transformer/model.py`:

```python
"""src/fcac/transformer/model.py

Transformer encoder over a sequence of CVAE latents, regressing to a scalar Cardiac
Age. Architecture is LOCKED (CLAUDE.md §2 "Transformer"): 4 heads, d_model=256,
d_k=64 (= d_model / n_heads), 3 encoder layers.

Pools via a learned CLS token rather than mean-pooling the sequence: mean-pooling
weighs every window equally regardless of how many windows a recording has, while a
CLS token lets attention decide which windows matter for the age estimate. Recordings
have a variable number of windows (CLAUDE.md §5 Latent.z is (W, 64), W varies per
recording), so the encoder takes an optional boolean padding mask.

Implements: CLAUDE.md §2 "Transformer" architecture.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from fcac.config import Config


class CardiacAgeTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_model = cfg.transformer.d_model
        n_heads = cfg.transformer.n_heads
        if d_model % n_heads != 0 or d_model // n_heads != cfg.transformer.d_k:
            raise ValueError("transformer.d_model / n_heads must equal transformer.d_k")

        self.max_seq_len = cfg.transformer.max_seq_len
        self.input_proj = nn.Linear(cfg.cvae.latent_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=cfg.transformer.d_ff,
            dropout=cfg.transformer.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.transformer.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, cfg.transformer.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.transformer.head_dropout),
            nn.Linear(cfg.transformer.head_hidden, 1),
        )

    def forward(
        self, z_seq: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """z_seq : (N, L, latent_dim), pad_mask : (N, L) bool, True = padded position.
        Returns ca_days : (N,)."""
        batch_size, seq_len, _ = z_seq.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        x = self.input_proj(z_seq)
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        x = x + self.pos_embedding[:, : seq_len + 1]

        if pad_mask is not None:
            cls_keep = torch.zeros(batch_size, 1, dtype=torch.bool, device=z_seq.device)
            pad_mask = torch.cat([cls_keep, pad_mask], dim=1)

        encoded = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(self.norm(encoded[:, 0])).squeeze(-1)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transformer.py -v`
Expected: PASS (all tests, including the pre-existing `test_transformer_forward_shape` and
`test_transformer_rejects_inconsistent_head_config`, both untouched and still valid).

- [ ] **Step 6: Commit**

```bash
git add src/fcac/transformer/model.py configs/default.yaml tests/test_transformer.py
git commit -m "feat(transformer): CLS-token pooling with padding-mask support"
```

---

## Task 7: Transformer dataset utilities (variable-length recording batching)

**Files:**
- Create: `src/fcac/transformer/dataset.py`
- Create: `tests/test_transformer_dataset.py`

**Interfaces:**
- Consumes: nothing new (plain `torch`).
- Produces: `pad_batch(sequences: List[Tensor]) -> (Tensor, Tensor)`,
  `RecordingDataset(z_by_recording, ca_days_target, crop_len, repeats_per_epoch=1)` (a
  `torch.utils.data.Dataset`), `collate(batch) -> (Tensor, Tensor, Tensor)`. Task 8 imports all
  three.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transformer_dataset.py`:

```python
"""tests/test_transformer_dataset.py

Synthetic-data tests for src/fcac/transformer/dataset.py (CLAUDE.md §8 item 9).
"""

import torch

from fcac.transformer.dataset import RecordingDataset, collate, pad_batch


def test_pad_batch_shapes_and_mask():
    sequences = [torch.randn(5, 4), torch.randn(8, 4), torch.randn(3, 4)]
    padded, pad_mask = pad_batch(sequences)
    assert padded.shape == (3, 8, 4)
    assert pad_mask.shape == (3, 8)
    assert not pad_mask[1].any()  # the longest sequence has no padding
    assert pad_mask[0, 5:].all() and not pad_mask[0, :5].any()
    assert pad_mask[2, 3:].all() and not pad_mask[2, :3].any()


def test_recording_dataset_crops_long_recordings():
    z_by_recording = [torch.randn(20, 4), torch.randn(3, 4)]
    targets = torch.tensor([280.0, 300.0])
    dataset = RecordingDataset(z_by_recording, targets, crop_len=5, repeats_per_epoch=2)
    assert len(dataset) == 4  # 2 recordings * repeats_per_epoch=2

    z, y = dataset[0]
    assert z.shape == (5, 4)  # cropped down from 20
    assert y == 280.0

    z_short, y_short = dataset[1]
    assert z_short.shape == (3, 4)  # shorter than crop_len -> kept whole
    assert y_short == 300.0


def test_collate_produces_padded_batch_and_mask():
    z_by_recording = [torch.randn(6, 4), torch.randn(2, 4)]
    targets = torch.tensor([280.0, 290.0])
    dataset = RecordingDataset(z_by_recording, targets, crop_len=100, repeats_per_epoch=1)
    batch = [dataset[i] for i in range(len(dataset))]

    z_padded, pad_mask, y = collate(batch)
    assert z_padded.shape == (2, 6, 4)
    assert pad_mask.shape == (2, 6)
    assert torch.equal(y, targets)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transformer_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fcac.transformer.dataset'`

- [ ] **Step 3: Implement**

Create `src/fcac/transformer/dataset.py`:

```python
"""src/fcac/transformer/dataset.py

Batching utilities for variable-length per-recording sequences of CVAE latents. Each
ADFECGDB/NInFEA recording produces a different number of 5 s windows (CLAUDE.md §5
Latent.z is (W, 64) per recording, W varies by recording length), so batches need
padding plus a mask, and long recordings are randomly cropped per epoch so every
window in a recording gets seen across training rather than only the first
`crop_len` of them.

Implements: CLAUDE.md §5 Latent -> Transformer input; CLAUDE.md §8 item 6.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset


def pad_batch(sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """List of (T_i, D) -> padded (B, T_max, D) and bool mask (B, T_max), True = pad."""
    max_len = max(seq.shape[0] for seq in sequences)
    dim = sequences[0].shape[1]
    padded = torch.zeros(len(sequences), max_len, dim)
    pad_mask = torch.ones(len(sequences), max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
        pad_mask[i, : len(seq)] = False
    return padded, pad_mask


class RecordingDataset(Dataset):
    """One item = a random crop of one recording's window-latent sequence, so a long
    recording contributes different windows on different epochs instead of always the
    same first `crop_len`."""

    def __init__(
        self,
        z_by_recording: List[torch.Tensor],
        ca_days_target: torch.Tensor,
        crop_len: int,
        repeats_per_epoch: int = 1,
    ):
        self.z_by_recording = z_by_recording
        self.ca_days_target = ca_days_target
        self.crop_len = crop_len
        self.repeats_per_epoch = repeats_per_epoch

    def __len__(self) -> int:
        return len(self.z_by_recording) * self.repeats_per_epoch

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        recording_idx = index % len(self.z_by_recording)
        z = self.z_by_recording[recording_idx]
        if len(z) > self.crop_len:
            start = torch.randint(0, len(z) - self.crop_len + 1, (1,)).item()
            z = z[start : start + self.crop_len]
        return z, self.ca_days_target[recording_idx]


def collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_padded, pad_mask = pad_batch([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return z_padded, pad_mask, targets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transformer_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fcac/transformer/dataset.py tests/test_transformer_dataset.py
git commit -m "feat(transformer): variable-length recording batching (pad_batch, RecordingDataset)"
```

---

## Task 8: Transformer training on recordings + LOSOCV/zero-shot evaluation

**Files:**
- Modify: `src/fcac/transformer/train.py`
- Create: `src/fcac/transformer/evaluate.py`
- Modify: `configs/default.yaml` (transformer section)
- Modify: `tests/test_transformer.py`
- Create: `tests/test_transformer_evaluate.py`

**Interfaces:**
- Consumes: `CardiacAgeTransformer` (Task 6), `RecordingDataset`/`collate`/`pad_batch` (Task 7),
  `fcac.evaluation.losocv.run_losocv`, `fcac.evaluation.ageing.{mae,rmse,bootstrap_ci}`,
  `fcac.evaluation.delta_age.delta_age_days`, `fcac.evaluation.ood.{OODStats, fit_ood,
  mahalanobis_distance}` (Task 1).
- Produces: `train_transformer(cfg, z_seq, ca_days_target, pad_mask=None) -> model` (unchanged
  call signature used by `scripts/07_train_transformer.py` and the existing smoke test, `pad_mask`
  optional so existing callers are unaffected), `train_transformer_on_recordings(cfg,
  z_by_recording, ca_days_target) -> model`, `predict_recordings(model, z_by_recording, crop_len)
  -> Tensor (R,)`, `losocv_adfecgdb(cfg, z_by_recording, ga_days, subject_ids) -> list[dict]`,
  `zero_shot_ninfea(cfg, model, z_by_recording, ga_days, ood_stats) -> dict`.

- [ ] **Step 1: Add `warmup_epochs`, `grad_clip`, `crop_len`, `repeats_per_epoch` to `configs/default.yaml`'s `transformer:` block**

```yaml
transformer:
  n_heads: 4
  d_model: 256
  d_k: 64
  n_layers: 3
  d_ff: 1024
  dropout: 0.2
  max_seq_len: 64
  head_hidden: 128
  head_dropout: 0.1
  loss: mae
  optimizer: adamw
  learning_rate: 1.0e-4
  weight_decay: 1.0e-2
  warmup_epochs: 10
  grad_clip: 1.0
  epochs: 200
  batch_size: 16
  crop_len: 16          # random window of segment latents per recording each training step
  repeats_per_epoch: 10 # each recording is seen this many times per epoch (different crops)
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_transformer.py` (the `train_transformer` smoke test stays as-is per
Global Constraints; these are additions):

```python
def test_train_transformer_accepts_pad_mask():
    cfg = _with_epochs(load_config(), epochs=2)
    n, seq_len = 8, 4
    z_seq = torch.randn(n, seq_len, cfg.cvae.latent_dim)
    pad_mask = torch.zeros(n, seq_len, dtype=torch.bool)
    ca_days_target = torch.rand(n) * 7 + 280

    from fcac.transformer.train import train_transformer

    model = train_transformer(cfg, z_seq, ca_days_target, pad_mask)
    prediction = infer(model, z_seq)
    assert prediction.ca_days.shape == (n,)


def test_train_transformer_on_recordings_runs_end_to_end():
    from fcac.transformer.train import train_transformer_on_recordings

    cfg = _with_epochs(load_config(), epochs=2)
    z_by_recording = [torch.randn(6, cfg.cvae.latent_dim), torch.randn(3, cfg.cvae.latent_dim),
                       torch.randn(9, cfg.cvae.latent_dim)]
    ca_days_target = torch.tensor([280.0, 285.0, 279.0])

    model = train_transformer_on_recordings(cfg, z_by_recording, ca_days_target)
    from fcac.transformer.evaluate import predict_recordings

    predictions = predict_recordings(model, z_by_recording, cfg.transformer.crop_len)
    assert predictions.shape == (3,)
    assert torch.isfinite(predictions).all()
```

Create `tests/test_transformer_evaluate.py`:

```python
"""tests/test_transformer_evaluate.py

Synthetic-data tests for src/fcac/transformer/evaluate.py (CLAUDE.md §8 item 9,
CLAUDE.md §2 "Evaluation").
"""

import torch

from fcac.config import Config, load_config
from fcac.evaluation.ood import fit_ood
from fcac.transformer.evaluate import losocv_adfecgdb, zero_shot_ninfea
from fcac.transformer.train import train_transformer_on_recordings


def _with_epochs(cfg: Config, epochs: int) -> Config:
    data = cfg.to_dict()
    data["transformer"]["epochs"] = epochs
    return Config(data)


def test_losocv_adfecgdb_produces_one_prediction_per_recording():
    cfg = _with_epochs(load_config(), epochs=2)
    n_recordings = 5
    z_by_recording = [torch.randn(6, cfg.cvae.latent_dim) for _ in range(n_recordings)]
    ga_days = torch.tensor([266.0, 273.0, 280.0, 287.0, 294.0])
    subject_ids = ["r01", "r04", "r07", "r08", "r10"]

    results = losocv_adfecgdb(cfg, z_by_recording, ga_days, subject_ids)
    assert len(results) == n_recordings
    assert {r["held_out_subject"] for r in results} == set(subject_ids)
    for r in results:
        assert "predicted_ca_days" in r and "ga_days" in r


def test_zero_shot_ninfea_reports_mae_and_ood_flags():
    cfg = _with_epochs(load_config(), epochs=2)
    z_train = [torch.randn(6, cfg.cvae.latent_dim) for _ in range(5)]
    ga_train = torch.tensor([266.0, 273.0, 280.0, 287.0, 294.0])
    model = train_transformer_on_recordings(cfg, z_train, ga_train)
    ood_stats = fit_ood(torch.stack([z.mean(dim=0) for z in z_train]))

    z_zero_shot = [torch.randn(4, cfg.cvae.latent_dim) for _ in range(3)]
    ga_zero_shot = torch.tensor([147.0, 154.0, 161.0])  # 21-27 wks, CLAUDE.md §3

    report = zero_shot_ninfea(cfg, model, z_zero_shot, ga_zero_shot, ood_stats)
    assert len(report["predicted_ca_days"]) == 3
    assert "mae_days" in report and "mae_ci95" in report
    assert len(report["ood_flag"]) == 3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transformer.py tests/test_transformer_evaluate.py -v`
Expected: FAIL — `train_transformer_on_recordings` and `fcac.transformer.evaluate` don't exist
yet; `train_transformer` doesn't accept a `pad_mask` argument yet.

- [ ] **Step 4: Implement**

Replace `src/fcac/transformer/train.py`:

```python
"""src/fcac/transformer/train.py

Training for the Cardiac Age transformer: MAE loss, AdamW, warmup + cosine decay
schedule, gradient clipping (CLAUDE.md §2 "Transformer").

Two entry points:
- `train_transformer` takes one fixed-shape batch and trains for `cfg.transformer.epochs`
  steps -- the CLAUDE.md §8 item 6 "runs end-to-end on random tensors" path, used by
  `scripts/07_train_transformer.py` and unit tests.
- `train_transformer_on_recordings` takes a list of variable-length per-recording
  latent sequences and trains with real epoch x mini-batch iteration over
  `RecordingDataset` -- the path `evaluate.py`'s LOSOCV/zero-shot driver actually uses.

Implements: CLAUDE.md §8 item 6.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from fcac.config import Config
from fcac.seed import set_seed
from fcac.transformer.dataset import RecordingDataset, collate
from fcac.transformer.model import CardiacAgeTransformer

logger = logging.getLogger(__name__)


def _warmup_cosine_lr(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_transformer(
    cfg: Config,
    z_seq: torch.Tensor,
    ca_days_target: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
) -> CardiacAgeTransformer:
    """z_seq : (N, L, latent_dim), ca_days_target : (N,), pad_mask : (N, L) bool or None."""
    set_seed(cfg.seed)
    model = CardiacAgeTransformer(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.transformer.learning_rate,
        weight_decay=cfg.transformer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _warmup_cosine_lr(
            epoch, cfg.transformer.warmup_epochs, cfg.transformer.epochs
        ),
    )
    loss_fn = nn.L1Loss()  # MAE

    for epoch in range(cfg.transformer.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(z_seq, pad_mask), ca_days_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.transformer.grad_clip)
        optimizer.step()
        scheduler.step()
        if epoch % max(1, cfg.transformer.epochs // 10) == 0:
            logger.info("epoch %d: MAE=%.4f days", epoch, loss.item())

    return model


def train_transformer_on_recordings(
    cfg: Config, z_by_recording: List[torch.Tensor], ca_days_target: torch.Tensor
) -> CardiacAgeTransformer:
    """Real training path for variable-length per-recording sequences: crops, pads
    and mini-batches recordings across `cfg.transformer.epochs`, unlike
    `train_transformer`, which trains on one fixed-shape batch."""
    set_seed(cfg.seed)
    model = CardiacAgeTransformer(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.transformer.learning_rate,
        weight_decay=cfg.transformer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _warmup_cosine_lr(
            epoch, cfg.transformer.warmup_epochs, cfg.transformer.epochs
        ),
    )
    loss_fn = nn.L1Loss()
    dataset = RecordingDataset(
        z_by_recording, ca_days_target, cfg.transformer.crop_len, cfg.transformer.repeats_per_epoch
    )
    loader = DataLoader(
        dataset,
        batch_size=min(cfg.transformer.batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate,
    )

    last_loss = torch.tensor(float("nan"))
    for epoch in range(cfg.transformer.epochs):
        model.train()
        for z_batch, mask_batch, y_batch in loader:
            optimizer.zero_grad()
            last_loss = loss_fn(model(z_batch, mask_batch), y_batch)
            last_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.transformer.grad_clip)
            optimizer.step()
        scheduler.step()
        if epoch % max(1, cfg.transformer.epochs // 10) == 0:
            logger.info("epoch %d: MAE(last batch)=%.4f days", epoch, last_loss.item())

    return model
```

Create `src/fcac/transformer/evaluate.py`:

```python
"""src/fcac/transformer/evaluate.py

LOSOCV over ADFECGDB and zero-shot evaluation on NInFEA for the Cardiac Age
transformer (CLAUDE.md §2 "Evaluation"). Wraps the generic fold driver in
fcac.evaluation.losocv so the fold-indexing logic has one implementation shared with
every other LOSOCV consumer, rather than being re-written per stage.

Implements: CLAUDE.md §2 "Evaluation" (LOSOCV, zero-shot, bootstrap CI, ΔAge).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from fcac.config import Config
from fcac.evaluation.ageing import bootstrap_ci, mae, rmse
from fcac.evaluation.delta_age import delta_age_days
from fcac.evaluation.losocv import run_losocv
from fcac.evaluation.ood import OODStats, mahalanobis_distance
from fcac.transformer.dataset import pad_batch
from fcac.transformer.model import CardiacAgeTransformer
from fcac.transformer.train import train_transformer_on_recordings


@torch.no_grad()
def predict_recordings(
    model: CardiacAgeTransformer, z_by_recording: List[torch.Tensor], crop_len: int
) -> torch.Tensor:
    """One CA prediction per recording, averaged over non-overlapping crop_len chunks
    so a recording longer than max_seq_len is still fully covered."""
    model.eval()
    predictions = []
    for z in z_by_recording:
        chunks = [z[i : i + crop_len] for i in range(0, len(z), crop_len)]
        z_padded, pad_mask = pad_batch(chunks)
        predictions.append(model(z_padded, pad_mask).mean())
    return torch.stack(predictions)


def losocv_adfecgdb(
    cfg: Config,
    z_by_recording: List[torch.Tensor],
    ga_days: torch.Tensor,
    subject_ids: List[str],
) -> List[Dict[str, Any]]:
    """CLAUDE.md §2 "Evaluation": ADFECGDB, LOSOCV, 5 folds (one per recording, since
    each of the 5 ADFECGDB recordings is its own subject)."""

    def train_fn(train_idx):
        z_train = [z_by_recording[i] for i in train_idx]
        ga_train = ga_days[torch.as_tensor(train_idx)]
        return train_transformer_on_recordings(cfg, z_train, ga_train)

    def eval_fn(model, test_idx):
        z_test = [z_by_recording[i] for i in test_idx]
        pred = predict_recordings(model, z_test, cfg.transformer.crop_len)
        target = ga_days[torch.as_tensor(test_idx)]
        return {"predicted_ca_days": float(pred[0]), "ga_days": float(target[0])}

    return run_losocv(subject_ids, train_fn, eval_fn)


def zero_shot_ninfea(
    cfg: Config,
    model: CardiacAgeTransformer,
    z_by_recording: List[torch.Tensor],
    ga_days: torch.Tensor,
    ood_stats: OODStats,
) -> Dict[str, Any]:
    """CLAUDE.md §2 "Evaluation": NInFEA, zero-shot, no fine-tuning. CLAUDE.md §3
    "GA coverage gap": training is 38-41 wks, NInFEA is 21-27 wks, so every zero-shot
    recording is flagged against the ADFECGDB-fitted OOD threshold rather than
    silently treated as in-distribution."""
    predicted = predict_recordings(model, z_by_recording, cfg.transformer.crop_len)
    latent_means = torch.stack([z.mean(dim=0) for z in z_by_recording])
    ood_distance = mahalanobis_distance(latent_means, ood_stats)

    mae_point, ci_low, ci_high = bootstrap_ci(ga_days.numpy(), predicted.numpy(), mae, cfg)
    return {
        "predicted_ca_days": predicted.tolist(),
        "ga_days": ga_days.tolist(),
        "delta_age_days": delta_age_days(predicted.numpy(), ga_days.numpy()).tolist(),
        "mae_days": mae_point,
        "mae_ci95": [ci_low, ci_high],
        "rmse_days": rmse(ga_days.numpy(), predicted.numpy()),
        "ood_distance": ood_distance.tolist(),
        "ood_flag": (ood_distance > ood_stats.threshold).tolist(),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transformer.py tests/test_transformer_evaluate.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/fcac/transformer/train.py src/fcac/transformer/evaluate.py configs/default.yaml \
        tests/test_transformer.py tests/test_transformer_evaluate.py
git commit -m "feat(transformer): recording-batched training, LOSOCV + zero-shot OOD-gated evaluation"
```

---

## Task 9: Retire the top-level `transformer/` folder, update scripts 06/07

**Files:**
- Delete: `transformer/` (entire directory)
- Modify: `scripts/06_train_cvae.py`
- Modify: `scripts/07_train_transformer.py`
- Modify: `reports/engineering_log.md`

**Interfaces:** none new — `train_cvae`/`train_transformer` signatures are unchanged from
before this plan, so the scripts only need their random-placeholder-tensor comments checked,
not their call sites.

- [ ] **Step 1: Confirm nothing else references the top-level `transformer/` package**

Run: `grep -rln "transformer/train\|transformer/smoke_test\|transformer\.config\.yaml" --include="*.py" --include="*.md" --include="*.sh" . | grep -v "^\./transformer/\|^\./src/fcac/transformer\|^\./\.venv"`
Expected: no output.

- [ ] **Step 2: Delete the directory**

```bash
git rm -r transformer/
```

- [ ] **Step 3: Verify `scripts/06_train_cvae.py` and `scripts/07_train_transformer.py` still work against the upgraded API**

Read both files. `scripts/06_train_cvae.py` calls `train_cvae(cfg, features, subject_ids,
ga_days)` — unchanged signature, no edit needed. `scripts/07_train_transformer.py` calls
`train_transformer(cfg, z_seq, ca_days_target)` — unchanged signature (Task 8 added an optional
trailing `pad_mask` arg), no edit needed either. Run them to confirm end-to-end on the
placeholder tensors:

Run: `.venv/bin/python scripts/06_train_cvae.py --n-samples 32`
Expected: exits 0, writes `data/processed/cvae/cvae.pt`

Run: `.venv/bin/python scripts/07_train_transformer.py --n-samples 16 --seq-len 4`
Expected: exits 0, writes `data/processed/transformer/transformer.pt`

If either fails, fix the script's call site to match Task 4/8's final function signatures
before moving on — do not delete the top-level folder's logic without both scripts running.

- [ ] **Step 4: Verify the full suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Log the decision**

Append to `reports/engineering_log.md`:

```markdown
- 2026-09-28: Ported the top-level `transformer/` folder's working logic (CLS-token
  pooling, padding-mask support, variable-length recording batching, LOSOCV +
  zero-shot-with-OOD-gating evaluation reusing `src/fcac/evaluation/`) into
  `src/fcac/transformer/`, then deleted the top-level folder. -> CLAUDE.md §4, §7.
```

- [ ] **Step 6: Commit**

```bash
git add reports/engineering_log.md
git commit -m "chore(transformer): remove top-level transformer/ folder now ported into src/fcac/transformer/"
```

---

## Task 10: Retire the legacy flat `src/` tree

**Files:**
- Delete: `src/preprocessing/`, `src/extraction/`, `src/model/`, `src/features/`, `src/evaluation/`
- Modify: `README.md` ("Note on legacy code" section)
- Modify: `CLAUDE.md` §9 ("Note on pre-existing code")

**Interfaces:** none — confirmed in this plan's initial research that no file outside these
five directories imports from them.

- [ ] **Step 1: Re-confirm no references, including anything added by Tasks 1–9**

Run: `grep -rnE "from (preprocessing|extraction|model|features|evaluation) import|^import (preprocessing|extraction|model|features|evaluation)\b" --include="*.py" . | grep -v "^\./src/preprocessing\|^\./src/extraction\|^\./src/model\|^\./src/features\|^\./src/evaluation\|^\./\.venv"`
Expected: no output.

- [ ] **Step 2: Delete the five legacy directories**

```bash
git rm -r src/preprocessing src/extraction src/model src/features src/evaluation
```

- [ ] **Step 3: Update `README.md`**

In `README.md`, replace the "Note on legacy code" section:

```markdown
## Note on legacy code

An older, flat `src/preprocessing/`, `src/extraction/`, `src/model/`, `src/features/`,
`src/evaluation/` tree existed from before the `src/fcac/` scaffold. Its working logic
(ADFECGDB/NInFEA loaders, CVAE, transformer) has since been ported into `src/fcac/` and
the legacy tree was removed; `src/fcac/` is now the only package new or existing code
imports from.
```

- [ ] **Step 4: Update `CLAUDE.md` §9**

In `CLAUDE.md`, replace the "## 9. Note on pre-existing code..." section (everything from that
heading to the end of the file) with:

```markdown
## 9. Note on the legacy flat `src/` tree (historical)

Before the `src/fcac/` scaffold, the repo had a flat `src/preprocessing/`,
`src/extraction/`, `src/model/`, `src/features/`, `src/evaluation/` tree with mostly
empty stub files, plus two working modules (`src/preprocessing/adfecgdb.py` and
`src/preprocessing/ninfea.py`) that were ported into `src/fcac/io/` and
`src/fcac/preprocessing/`, fixing two bugs relative to the LOCKED decisions in §2
(notch applied to both datasets instead of NInFEA-only; NInFEA filtering happened
*after* resampling instead of before) along the way.

A second, untracked pair of top-level `cvae/` and `transformer/` folders later
appeared with a more complete CVAE/Transformer implementation (InfoNCE projection
head, GA-window positive pairing, KL warmup, CLS-token pooling, LOSOCV/zero-shot
wiring, Mahalanobis OOD gating) than the `src/fcac/cvae`/`src/fcac/transformer`
scaffold had at the time. That logic was ported into `src/fcac/cvae/` and
`src/fcac/transformer/` (see `reports/engineering_log.md`, 2026-09-28 entries), and
both the legacy flat tree and the top-level `cvae/`/`transformer/` folders were then
deleted. `src/fcac/` has been the sole authoritative package since; this section is
kept only as a historical record of where each piece of logic originally came from.
```

- [ ] **Step 5: Verify the full suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "chore: retire legacy flat src/ tree now that src/fcac/ is the sole package"
```

---

## Task 11: Final verification

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full test suite one more time**

Run: `.venv/bin/python -m pytest -v`
Expected: PASS, and count the tests — should be strictly more than the 47 that passed before
this plan started (Task 1: +2, Task 2: +1, Task 3: +2, Task 4: +4, Task 6: +2, Task 7: +3,
Task 8: +3, so at minimum 47 + 17 = 64 tests).

- [ ] **Step 2: Confirm the repo tree matches CLAUDE.md §4 again**

Run: `find . -maxdepth 1 -not -path '.' -not -path './.git' -not -path './.venv' | sort`
Expected: `cvae/`, `transformer/`, and the five legacy `src/*` directories are gone;
`fetal-cardiac-ageing-clock_version1.py` is still present untouched (explicitly out of scope
per Global Constraints).

- [ ] **Step 3: Confirm `git status` is clean except for intended changes**

Run: `git status`
Expected: working tree clean (everything from Tasks 1–10 was committed); no accidental leftover
files (e.g. no `__pycache__` additions, no stray `.pt`/`.json` files written by Task 9 Step 3's
manual script runs — those are gitignored under `data/processed/`, confirm with `git status
--ignored=no` that they don't show as untracked-and-uncommitted).

- [ ] **Step 4: Report to the user**

Summarize: which tasks landed, current test count, and that GA labels / MEMD source / PINN
coordinate scheme remain the open blockers untouched by this plan (per `plan.md`).
