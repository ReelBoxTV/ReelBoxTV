"""
DeepSeek-R1 Rotary Position Embedding (RoPE) — decoupled ("pre" / "post") RoPE.
=============================================================================

This module implements the *exact* RoPE mechanics used inside DeepSeek-R1's
Multi-head Latent Attention (MLA), including the YaRN long-context extension.

Why "pre" and "post" RoPE?
--------------------------
In a vanilla Transformer, the *entire* query/key head is rotated by RoPE.
DeepSeek-R1 instead **decouples** the head into two disjoint sub-vectors:

    head = [ nope_part ;  rope_part ]
            ^^^^^^^^^^^   ^^^^^^^^^^^
            carries        carries the
            content,       position via
            NO rotation    a rotation R_m

So for every query/key there are two distinct "moments":

    * PRE-RoPE : the raw projections straight out of the linear layers,
                 BEFORE any rotation matrix is applied.  This is what MLA
                 actually stores / absorbs (the KV-cache holds the *latent*
                 plus the single shared *pre/post-rotated* key-rope vector).

    * POST-RoPE: the rope_part after the rotation R_m has been multiplied in.
                 The nope_part is identical pre and post (it is never rotated).

This file provides the primitives; `mla_attention.py` wires them into MLA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Frequency table  theta_i = base^(-2i/d)   (the "inverse frequencies")
# ---------------------------------------------------------------------------
def build_inv_freq(rope_dim: int, base: float = 10000.0,
                   device=None, dtype=torch.float32) -> torch.Tensor:
    """Return inv_freq of shape (rope_dim/2,).

    Mathematically  theta_i = base^(-2i/d)  for i = 0 .. d/2-1.
    Low i  -> high frequency (fast rotation, fine position resolution).
    High i -> low  frequency (slow rotation, coarse / long-range).
    """
    assert rope_dim % 2 == 0, "rope dimension must be even (rotated in 2D pairs)"
    i = torch.arange(0, rope_dim, 2, device=device, dtype=dtype)  # 0,2,4,...
    return base ** (-(i / rope_dim))                              # (d/2,)


# ---------------------------------------------------------------------------
# 2. YaRN scaling  (DeepSeek's long-context trick — optional)
# ---------------------------------------------------------------------------
def _yarn_correction_dim(num_rotations, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(low_rot, high_rot, dim, base, max_pos):
    low = math.floor(_yarn_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_yarn_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(lo, hi, size, device, dtype):
    if lo == hi:
        hi += 0.001
    ramp = (torch.arange(size, device=device, dtype=dtype) - lo) / (hi - lo)
    return torch.clamp(ramp, 0, 1)


def build_inv_freq_yarn(rope_dim: int, base: float, scaling_factor: float,
                        original_max_pos: int, beta_fast: float = 32,
                        beta_slow: float = 1, device=None,
                        dtype=torch.float32) -> Tuple[torch.Tensor, float]:
    """YaRN-interpolated inverse frequencies + attention temperature `mscale`.

    YaRN interpolates between:
      * pure position *interpolation*  (divide angle by `scaling_factor`),
        which preserves high-frequency dims badly, and
      * pure *extrapolation* (leave angle as-is),
    using a per-dimension ramp so high frequencies extrapolate and low
    frequencies interpolate.  `mscale` rescales attention logits to keep the
    softmax temperature stable after the context is stretched.
    """
    freq_extra = build_inv_freq(rope_dim, base, device, dtype)          # extrapolate
    freq_inter = freq_extra / scaling_factor                            # interpolate
    low, high = _yarn_correction_range(beta_fast, beta_slow, rope_dim, base, original_max_pos)
    # mask == 1 where we keep extrapolation (high freq), 0 where we interpolate
    mask = 1.0 - _yarn_linear_ramp(low, high, rope_dim // 2, device, dtype)
    inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    mscale = 0.1 * math.log(scaling_factor) + 1.0 if scaling_factor > 1 else 1.0
    return inv_freq, mscale


# ---------------------------------------------------------------------------
# 3. cos / sin tables for a sequence of positions
# ---------------------------------------------------------------------------
def build_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """positions: (S,) integer/float positions m.
    inv_freq:    (d/2,)
    Returns cos, sin of shape (S, d) in the *half-split* (GPT-NeoX) layout,
    i.e. each angle m*theta_i is duplicated so the table aligns with
    [first half | second half] of the vector.

        angles  = m * theta_i                      -> (S, d/2)
        emb     = cat(angles, angles, dim=-1)      -> (S, d)
        cos, sin= cos(emb), sin(emb)
    """
    angles = torch.einsum("s,d->sd", positions.to(inv_freq.dtype), inv_freq)  # (S, d/2)
    emb = torch.cat([angles, angles], dim=-1)                                 # (S, d)
    return emb.cos(), emb.sin()


# ---------------------------------------------------------------------------
# 4. rotate_half  — the GPT-NeoX / HF formulation of the 2D rotation
# ---------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Given x = [x1 ; x2] (split in half on the last dim) return [-x2 ; x1].

    This realises the 2x2 rotation
        [cos -sin][x1]
        [sin  cos][x2]
    in vectorised form:  RoPE(x) = x*cos + rotate_half(x)*sin.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               unsqueeze_dim: int = 1) -> torch.Tensor:
    """Apply RoPE to `x`.

    x:        (B, H, S, d)  or (B, 1, S, d) for the shared key-rope vector.
    cos/sin:  (B, S, d)  ->  unsqueezed to broadcast over the head dim.

    Returns the **POST-RoPE** tensor.  Input is the **PRE-RoPE** tensor.
    """
    cos = cos.unsqueeze(unsqueeze_dim)   # (B, 1, S, d)
    sin = sin.unsqueeze(unsqueeze_dim)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# 5. A small nn.Module that owns the table and serves cos/sin on demand.
# ---------------------------------------------------------------------------
@dataclass
class RopeConfig:
    rope_dim: int = 64               # qk_rope_head_dim in DeepSeek-R1
    base: float = 10000.0
    use_yarn: bool = False
    scaling_factor: float = 40.0     # DeepSeek-R1 stretches 4k -> 160k
    original_max_pos: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0


class DeepseekRotaryEmbedding(nn.Module):
    def __init__(self, cfg: RopeConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_yarn:
            inv_freq, self.mscale = build_inv_freq_yarn(
                cfg.rope_dim, cfg.base, cfg.scaling_factor,
                cfg.original_max_pos, cfg.beta_fast, cfg.beta_slow)
        else:
            inv_freq, self.mscale = build_inv_freq(cfg.rope_dim, cfg.base), 1.0
        # buffer so it moves with .to(device) but is not a learned parameter
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """positions: (B, S) -> cos, sin each (B, S, rope_dim)."""
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        cos, sin = build_cos_sin(positions.reshape(-1), self.inv_freq)
        cos = cos.reshape(*positions.shape, -1) * self.mscale
        sin = sin.reshape(*positions.shape, -1) * self.mscale
        return cos, sin
