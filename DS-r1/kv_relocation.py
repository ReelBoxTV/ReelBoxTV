"""
Relocating a KV block in the context (position K -> position P) under
DeepSeek-R1 decoupled RoPE.
===========================================================================

WHY THIS IS CHEAP AND EXACT
---------------------------
The MLA KV cache stores, per token:
    * c_kv      : compressed latent (content). POSITION-INDEPENDENT — no RoPE.
                  Values are reconstructed from it, so values move for free.
    * k_rope    : the 64-d shared decoupled key, cached POST-RoPE, i.e. already
                  multiplied by R_K at the position K where it was produced.

RoPE rotations compose additively in angle:  R_a · R_b = R_{a+b}.
So to send a token from absolute position K to P we only need the DELTA:

    k_new = R_P · k_pre = R_{P-K} · (R_K · k_pre) = R_{P-K} · k_post

i.e. re-rotate the *cached* key by Δ = P - K.  The pre-RoPE value is never
needed.  A contiguous block translating rigidly shares a single Δ, so the whole
block is shifted by applying R_Δ to every key in it.  c_kv is untouched.

This module provides that operation and a test proving it equals a full
recompute at the new positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepseek_rope import (DeepseekRotaryEmbedding, RopeConfig, apply_rope,
                           build_cos_sin)


@dataclass
class MLAKVCache:
    """Minimal MLA cache. Shapes:
        c_kv   : (B, S, kv_lora_rank)      — position-independent content latent
        k_rope : (B, S, rope_dim)          — shared decoupled key, POST-RoPE
        pos    : (B, S) long               — absolute position each token sits at
    """
    c_kv: torch.Tensor
    k_rope: torch.Tensor
    pos: torch.Tensor


def _delta_cos_sin(delta: torch.Tensor, rope: DeepseekRotaryEmbedding):
    """cos/sin tables for a per-token rotation angle of `delta` positions."""
    # delta: (N,) signed integer shifts. build_cos_sin handles signed positions.
    cos, sin = build_cos_sin(delta.to(rope.inv_freq.dtype), rope.inv_freq)
    return cos * rope.mscale, sin * rope.mscale


def relocate_kv_block(cache: MLAKVCache, rope: DeepseekRotaryEmbedding,
                      k_start: int, length: int, p_start: int,
                      batch: int = 0) -> MLAKVCache:
    """Return a NEW cache with the block [k_start, k_start+length) moved so its
    first token now sits at absolute position p_start (rigid translation).

    Only the rope key is touched (re-rotated by Δ = p_start - k_start); c_kv is
    copied unchanged.  `pos` is updated to reflect the new absolute positions.
    """
    delta = p_start - k_start
    sl = slice(k_start, k_start + length)

    new_c_kv = cache.c_kv.clone()
    new_k_rope = cache.k_rope.clone()
    new_pos = cache.pos.clone()

    # one shared Δ for the whole rigidly-translated block
    d = torch.full((length,), float(delta), device=cache.k_rope.device)
    cos, sin = _delta_cos_sin(d, rope)                     # (length, rope_dim)

    # apply R_Δ to the cached POST-RoPE keys of this block
    blk = cache.k_rope[batch, sl].unsqueeze(0).unsqueeze(0)   # (1,1,length,dr)
    cos_b = cos.unsqueeze(0)                                  # (1,length,dr)
    sin_b = sin.unsqueeze(0)
    rotated = apply_rope(blk, cos_b, sin_b).squeeze(0).squeeze(0)
    new_k_rope[batch, sl] = rotated

    new_pos[batch, sl] = cache.pos[batch, sl] + delta
    return MLAKVCache(new_c_kv, new_k_rope, new_pos)


# ---------------------------------------------------------------------------
# Verification: relocating == recomputing RoPE at the destination positions.
# ---------------------------------------------------------------------------
def _test_relocation_is_exact():
    torch.manual_seed(0)
    B, S, dr = 1, 12, 64
    rope = DeepseekRotaryEmbedding(RopeConfig(rope_dim=dr))

    # raw (pre-RoPE) shared keys and a content latent
    k_pre = torch.randn(B, S, dr)
    c_kv = torch.randn(B, S, 32)
    orig_pos = torch.arange(S).unsqueeze(0)                 # 0..S-1

    # build the cache as it would exist after a normal forward (post-RoPE)
    cos, sin = rope(orig_pos)
    k_post = apply_rope(k_pre.unsqueeze(1), cos, sin).squeeze(1)
    cache = MLAKVCache(c_kv=c_kv, k_rope=k_post, pos=orig_pos.clone())

    # move block [3,7) so it starts at position 9  (Δ = +6)
    k_start, length, p_start = 3, 4, 9
    moved = relocate_kv_block(cache, rope, k_start, length, p_start)

    # GROUND TRUTH: recompute RoPE on the same pre-RoPE keys at NEW positions
    new_positions = torch.arange(p_start, p_start + length).unsqueeze(0)
    cos2, sin2 = rope(new_positions)
    gt = apply_rope(k_pre[:, k_start:k_start + length].unsqueeze(1),
                    cos2, sin2).squeeze(1)[0]

    got = moved.k_rope[0, k_start:k_start + length]
    assert torch.allclose(got, gt, atol=1e-5), (got - gt).abs().max()

    # c_kv must be byte-for-byte unchanged (content is position-free)
    assert torch.equal(moved.c_kv, cache.c_kv)
    # positions updated
    assert moved.pos[0, k_start:k_start + length].tolist() == list(range(9, 13))

    print("exactness OK: relocated rope key == recompute at destination "
          f"(max err {(got-gt).abs().max():.2e})")
    print("c_kv unchanged OK; positions updated OK")

    # bonus: relative-distance to a future query is preserved correctly.
    # a query at position Q sees the moved key at relative distance Q - P.
    q_pre = torch.randn(B, 1, 1, dr)
    Q = 11
    cq, sq = rope(torch.tensor([[Q]]))
    q_post = apply_rope(q_pre, cq, sq)                      # (1,1,1,dr)
    # score with the first moved token (now at position 9): should match a key
    # freshly roped at position 9.
    score_moved = (q_post[0, 0, 0] * moved.k_rope[0, k_start]).sum()
    cos9, sin9 = rope(torch.tensor([[9]]))
    fresh = apply_rope(k_pre[:, k_start:k_start + 1].unsqueeze(1),
                       cos9, sin9)[0, 0, 0]                 # -> (dr,)
    score_fresh = (q_post[0, 0, 0] * fresh).sum()
    assert torch.allclose(score_moved, score_fresh, atol=1e-5)
    print(f"query-score consistency OK: {score_moved:+.5f} == {score_fresh:+.5f}")


if __name__ == "__main__":
    _test_relocation_is_exact()
    print("\nKV block relocation verified.")
