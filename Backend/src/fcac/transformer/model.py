"""src/fcac/transformer/model.py

Transformer encoder over a sequence of CVAE latents, regressing to a scalar Cardiac
Age. Architecture is LOCKED (CLAUDE.md §2 "Transformer"): 4 heads, d_model=256,
d_k=64 (= d_model / n_heads), 3 encoder layers.

Pools via a learned CLS token rather than mean-pooling the sequence: mean-pooling
weighs every window equally regardless of how many windows a recording has, while a
CLS token lets attention decide which windows matter for the age estimate. Recordings
have a variable number of windows (CLAUDE.md §5 Latent.z is (W, 64), W varies per
recording), so the encoder takes an optional boolean padding mask.

`target_mean`/`target_std` are non-trainable buffers, defaulting to identity (0, 1).
`forward()` never touches them -- it always returns the raw regression head output.
`train_transformer_on_recordings` (CLAUDE.md §2 "Evaluation" GA is ~266-301 days) sets
them to the real training-target mean/std and trains the head against the *scaled*
target, because a freshly-initialised head starts near 0 and an AdamW step at
lr=1e-4 cannot close a ~280-day gap in any reasonable epoch budget -- confirmed by a
failing test that reproduced MAE ~266 days (a collapsed, input-independent output)
before this fix. `predict_recordings` reads the buffers back to un-scale predictions
into real days. `train_transformer`/`infer()` (the CLAUDE.md §8 item 6 fixed-batch
smoke-test path) never set them, so they stay at the identity default and that path's
behaviour is unchanged.

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
        self.register_buffer("target_mean", torch.zeros(()))
        self.register_buffer("target_std", torch.ones(()))
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
