#!/usr/bin/env python3
"""scripts/visualize_transformer_attention.py

Extracts and visualizes the Transformer Multi-Head Self-Attention matrices:
1. Computes the 4-head attention weights across layers.
2. Analyzes which temporal ECG windows the CLS token attends to most for Cardiac Age prediction.
3. Saves a publication-quality figure to reports/figures/transformer_attention.png.
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from fcac.config import load_config
from fcac.transformer.latents import load_cvae_and_fill_values
from fcac.transformer.model import CardiacAgeTransformer
from fcac.viz.plots import apply_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("transformer_attn")


def compute_layer_attention(layer, x):
    """
    Computes (n_heads, L, L) attention matrix for a TransformerEncoderLayer
    using its exact in_proj_weight and in_proj_bias.
    """
    w_q, w_k, _ = layer.self_attn.in_proj_weight.chunk(3)
    b_q, b_k, _ = layer.self_attn.in_proj_bias.chunk(3)

    q = F.linear(x, w_q, b_q)  # (1, L, d_model)
    k = F.linear(x, w_k, b_k)

    bsz, seq_len, d_model = q.shape
    n_heads = layer.self_attn.num_heads
    d_k = d_model // n_heads

    # Reshape into (bsz, n_heads, seq_len, d_k)
    q = q.view(bsz, seq_len, n_heads, d_k).transpose(1, 2)
    k = k.view(bsz, seq_len, n_heads, d_k).transpose(1, 2)

    # Scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)  # (bsz, n_heads, seq_len, seq_len)
    return attn[0]


def main():
    cfg = load_config(dataset="adfecgdb", experiment="synthetic_ga_smoketest")
    cvae, fill = load_cvae_and_fill_values(cfg)

    # Load r01 features
    feat_path = Path("data/processed/features/adfecgdb/r01.npz")
    feats = np.load(feat_path, allow_pickle=True)["x"].astype(np.float32)
    feats = np.where(np.isnan(feats), fill, feats).astype(np.float32)

    with torch.no_grad():
        z = cvae(cvae.standardise(torch.from_numpy(feats)))["mu"]  # (59, 64)

    transformer = CardiacAgeTransformer(cfg)
    transformer.load_state_dict(torch.load("data/processed/transformer/transformer.pt", map_location="cpu"))
    transformer.eval()

    # Crop to 16 windows for crisp attention visualization
    crop_z = z[:16].unsqueeze(0)  # (1, 16, 64)
    with torch.no_grad():
        x = transformer.input_proj(crop_z)
        x = torch.cat([transformer.cls_token.expand(1, -1, -1), x], dim=1)
        x = x + transformer.pos_embedding[:, : 17]

        # Extract attention weights from Layer 1, Layer 2, Layer 3
        attn_layers = []
        for l in transformer.encoder.layers:
            attn = compute_layer_attention(l, x)
            attn_layers.append(attn.numpy())
            x = l(x)

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Heads 1 to 4 of the final Transformer layer
    layer_idx = 2  # Layer 3
    attn_last = attn_layers[layer_idx]  # (4, 17, 17)

    labels = ["CLS"] + [f"W{i+1}" for i in range(16)]

    for h in range(4):
        ax = axes[h // 2, h % 2]
        mat = attn_last[h]
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=np.percentile(mat, 98))
        ax.set_title(f"Transformer Layer 3 — Attention Head {h+1}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(17))
        ax.set_yticks(range(17))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Key Window Index")
        ax.set_ylabel("Query Window Index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")

    fig.suptitle("Transformer Multi-Head Self-Attention Diagnostic (ADFECGDB r01)", fontsize=13, fontweight="bold")
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transformer_attention.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    logger.info("Saved attention weights figure to %s", out_path)

    # Copy to artifact directory
    artifact_path = Path(r"C:\Users\Dayanand\.gemini\antigravity-ide\brain\80bb8912-db5b-43c6-95bf-1b045c00ed60\transformer_attention.png")
    import shutil
    shutil.copy(out_path, artifact_path)
    logger.info("Copied figure to %s", artifact_path)


if __name__ == "__main__":
    main()
