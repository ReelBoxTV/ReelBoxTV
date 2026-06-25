"""
Self-contained NumPy verification of the RoPE math used in DeepSeek-R1.
Run:  python test_rope_math.py

It proves, numerically, the two facts the whole design rests on:

  (A) RELATIVE-POSITION property:
        <RoPE(q, m), RoPE(k, n)>  depends only on (m - n), not on m, n alone.
      This is *why* rotating only the rope_part still encodes position.

  (B) DECOUPLED-SCORE identity:
        the MLA score  q_nope·k_nope + RoPE(q_rope,m)·RoPE(k_rope,n)
      equals a plain attention score on the concatenated head where ONLY the
      rope_part is rotated — confirming the pre/post split is loss-free.
"""

import numpy as np


def inv_freq(d, base=10000.0):
    i = np.arange(0, d, 2, dtype=np.float64)
    return base ** (-(i / d))                      # (d/2,)


def rope(x, m, d, base=10000.0):
    """Rotate vector x (len d) at position m using 2D pair rotations.
    Uses the *interleaved-pair* form, the cleanest expression of R_m."""
    f = inv_freq(d, base)                          # (d/2,)
    ang = m * f                                     # (d/2,)
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[0::2], x[1::2]                       # even / odd = the 2D pairs
    y = np.empty_like(x)
    y[0::2] = x1 * c - x2 * s
    y[1::2] = x1 * s + x2 * c
    return y


def test_relative_position():
    rng = np.random.default_rng(0)
    d = 64
    q, k = rng.standard_normal(d), rng.standard_normal(d)
    # Fix the relative distance (m-n)=3, vary the absolute base position.
    scores = []
    for base_pos in [0, 10, 100, 1000]:
        m, n = base_pos + 3, base_pos
        scores.append(rope(q, m, d) @ rope(k, n, d))
    scores = np.array(scores)
    assert np.allclose(scores, scores[0], atol=1e-9), scores
    print(f"(A) relative-position OK: score(Δ=3) = {scores[0]:+.6f} "
          f"is constant across absolute positions {scores}")


def test_decoupled_score():
    rng = np.random.default_rng(1)
    dn, dr = 128, 64                                # nope / rope dims
    m, n = 7, 4
    q_nope, k_nope = rng.standard_normal(dn), rng.standard_normal(dn)
    q_rope, k_rope = rng.standard_normal(dr), rng.standard_normal(dr)

    # Path 1: decoupled score (what MLA computes)
    decoupled = q_nope @ k_nope + rope(q_rope, m, dr) @ rope(k_rope, n, dr)

    # Path 2: build full heads, rotate ONLY the rope slice, take one dot product
    q_full = np.concatenate([q_nope, rope(q_rope, m, dr)])
    k_full = np.concatenate([k_nope, rope(k_rope, n, dr)])
    full = q_full @ k_full

    assert np.allclose(decoupled, full, atol=1e-10)
    print(f"(B) decoupled-score OK: {decoupled:+.6f} == {full:+.6f}")


def test_rotation_is_orthogonal():
    """R_m preserves norms (it is a rotation): ||RoPE(x)|| == ||x||."""
    rng = np.random.default_rng(2)
    d = 64
    x = rng.standard_normal(d)
    for m in [0, 1, 5, 123]:
        assert np.allclose(np.linalg.norm(rope(x, m, d)), np.linalg.norm(x))
    print("(C) orthogonality OK: RoPE preserves vector norm at all positions")


if __name__ == "__main__":
    test_relative_position()
    test_decoupled_score()
    test_rotation_is_orthogonal()
    print("\nAll RoPE math checks passed.")
