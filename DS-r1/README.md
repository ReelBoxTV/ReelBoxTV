# DeepSeek-R1 Decoupled RoPE — Pre & Post RoPE, in Detail

This repo implements and explains the **decoupled Rotary Position Embedding
(RoPE)** used inside DeepSeek-R1's **Multi-head Latent Attention (MLA)**, with
an emphasis on the two distinct stages the title calls **pre-RoPE** and
**post-RoPE**.

| File | What it is |
|------|------------|
| `deepseek_rope.py` | RoPE primitives: frequency table, YaRN scaling, `rotate_half`, `apply_rope`, and an `nn.Module` that serves `cos/sin`. |
| `mla_attention.py` | A full DeepSeek-R1 MLA block showing exactly where pre-RoPE and post-RoPE tensors live. |
| `test_rope_math.py` | Dependency-light NumPy proof of the relative-position property and the decoupled-score identity. Run it: `python test_rope_math.py`. |

---

## 1. Standard RoPE — the rotation

RoPE injects **absolute** position through a rotation, chosen so that the
attention dot-product only ever sees the **relative** position.

For a head vector $x \in \mathbb{R}^{d}$ ($d$ even) at position $m$, split it
into $d/2$ consecutive 2-D pairs $(x_{2i}, x_{2i+1})$ and rotate the $i$-th pair
by angle $m\theta_i$:

$$
\theta_i = b^{-2i/d}, \qquad i = 0,\dots,\tfrac{d}{2}-1, \qquad b = 10000 .
$$

The whole operation is a block-diagonal **orthogonal** matrix $R_m$:

$$
R_m =
\begin{pmatrix}
R(m\theta_0) & & \\
 & \ddots & \\
 & & R(m\theta_{d/2-1})
\end{pmatrix},
\qquad
R(\phi)=\begin{pmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{pmatrix}.
$$

So $\text{RoPE}(x, m) = R_m\,x$.

### The one property that matters

Because each $R(\phi)$ is a planar rotation, $R_m^\top R_n = R_{n-m}$. Hence for
a query at $m$ and key at $n$:

$$
\big\langle R_m q,\; R_n k \big\rangle
= q^\top R_m^\top R_n\, k
= q^\top R_{n-m}\, k .
$$

The score **depends only on $m-n$**. That is the entire point of RoPE, and
`test_rope_math.py` (check **A**) verifies it numerically.

### `rotate_half` — the vectorised form used in code

Instead of materialising $R_m$, implementations precompute $\cos(m\theta_i)$ and
$\sin(m\theta_i)$ and use:

$$
\text{RoPE}(x) = x \odot \cos + \text{rotate\_half}(x) \odot \sin,
\qquad
\text{rotate\_half}([x_1; x_2]) = [-x_2;\, x_1].
$$

Here the vector is split into halves $x_1, x_2 \in \mathbb{R}^{d/2}$ (the
GPT-NeoX / HuggingFace layout). The `cos`/`sin` tables duplicate each angle so
they line up with `[first half | second half]` — see `build_cos_sin`.
(`test_rope_math.py` uses the mathematically-equivalent **interleaved** form,
which makes the 2-D pairs explicit.)

---

## 2. Why DeepSeek *decouples* RoPE

MLA's whole speed advantage comes from **low-rank KV compression**: instead of
caching a full key/value per head, it caches a small shared latent
$c^{KV}_t = W^{DKV} h_t \in \mathbb{R}^{512}$, and reconstructs keys/values on
the fly with $k^C = W^{UK} c^{KV}$, $v = W^{UV} c^{KV}$.

The nope (content) part of the score can then be **absorbed** at inference:

$$
(q^C)^\top k^C
= (W^{UQ} c^Q)^\top (W^{UK} c^{KV})
= (c^Q)^\top \underbrace{(W^{UQ})^\top W^{UK}}_{\text{precompute once}} c^{KV}.
$$

So you never need the per-head keys — just the latent $c^{KV}$.

**RoPE breaks this.** If you rotated the reconstructed key,
$R_m W^{UK} c^{KV}$, the position-dependent $R_m$ sits *between* $W^{UQ}$ and
$W^{UK}$ and cannot be folded into a position-independent constant. The absorb
trick dies.

**DeepSeek's fix:** carry position on a *separate, small* slice that is **not**
reconstructed from the compressed latent. Each head is split:

$$
q_t = [\,q_t^{C}\; ;\; q_t^{R}\,], \qquad
k_t = [\,k_t^{C}\; ;\; k_t^{R}\,],
$$

with dims $128$ (nope, $C$) $+\,64$ (rope, $R$) $=192$. Only the $R$ slice is
rotated, and the rope key $k^R$ is **shared across all heads** (one 64-dim
vector per token), so the cache stores just $c^{KV}\,(512) + k^R\,(64)$ per
token.

The score factorises additively:

$$
\boxed{\,s_{t,s} = \underbrace{(q_t^{C})^\top k_s^{C}}_{\text{content, absorbable}}
\;+\; \underbrace{(R_m q_t^{R})^\top (R_n k_s^{R})}_{\text{position, } = (q_t^R)^\top R_{n-m} k_s^R}\,}
$$

`test_rope_math.py` (check **B**) verifies this additive split is exactly equal
to a plain dot-product on the concatenated head where only the $R$-slice is
rotated.

---

## 3. Pre-RoPE vs Post-RoPE — the two stages

This is the crux of the title. In MLA there are two clearly separated moments:

| | tensor | how produced | rotated? |
|---|--------|--------------|----------|
| **PRE-RoPE** | `q_rope_pre` | $W^{QR} c^Q$ (per head, 64-d) | no — raw projection |
| | `k_rope_pre` | $W^{KR} h$ (shared, 64-d) | no — this is what the **KV cache stores** |
| **POST-RoPE** | `q_rope_post` | $R_m \cdot$ `q_rope_pre` | yes |
| | `k_rope_post` | $R_n \cdot$ `k_rope_pre` | yes |
| **(never roped)** | `q_nope`, `k_nope` | $W^{UQ}c^Q$, $W^{UK}c^{KV}$ | pre == post |

- **Pre-RoPE** is everything straight out of the linear layers, *before*
  multiplying in $R$. The KV cache holds the **pre-RoPE shared key**
  `k_rope_pre` together with the latent `c_kv` — both position-independent
  projections — which is what keeps the cache tiny and the absorb trick alive.
- **Post-RoPE** is obtained by applying $R_m$ / $R_n$ via `apply_rope`. This is
  the only place position enters. In code:

  ```python
  cos, sin = self.rope(positions)               # angles m*theta_i
  q_rope_post = apply_rope(q_rope_pre, cos, sin) # R_m · q_rope_pre
  k_rope_post = apply_rope(k_rope_pre, cos, sin) # R_n · k_rope_pre  (then broadcast to all heads)
  ```

The nope parts skip both stages entirely, which is *why* they remain
absorbable.

---

## 4. Full MLA forward (dimensions)

For DeepSeek-V3/R1 (`hidden=7168, heads=128`):

```
h (B,S,7168)
 ├─ q_a_proj  ─► c_q (B,S,1536) ─ RMSNorm ─ q_b_proj ─► (B,H,S,192)
 │                                   └► split ► q_nope (128) | q_rope_pre (64)
 │
 └─ kv_a_proj ─► (B,S, 512+64)
        └► split ► c_kv (512) ─RMSNorm─ kv_b_proj ─► k_nope (128) | value (128)
                 │
                 └► k_rope_pre (64, SHARED across heads)

RoPE:  q_rope_post = R_m q_rope_pre        k_rope_post = R_n k_rope_pre
query = [q_nope ; q_rope_post] (192)       key = [k_nope ; k_rope_post] (192)
scores = query·keyᵀ / sqrt(192)  ─softmax─►  · value ─► o_proj ─► (B,S,7168)
```

Note the softmax scale uses the **full** head dim $192 = 128 + 64$, and under
YaRN an extra temperature `mscale` multiplies `cos`/`sin` to stabilise the
softmax after the context window is stretched (see `build_inv_freq_yarn`).

---

## 5. YaRN long-context (optional)

DeepSeek-R1 extends context far beyond the pre-training window using **YaRN**,
which interpolates the rope angles per-frequency:

- high-frequency dims (small $i$) keep **extrapolating** (angle unchanged),
- low-frequency dims (large $i$) **interpolate** (angle divided by the scaling
  factor $s$),
- a smooth ramp blends the two, and attention logits are rescaled by
  $\text{mscale} = 0.1\ln s + 1$ to hold the softmax temperature steady.

Enable with `RopeConfig(use_yarn=True, scaling_factor=40, original_max_pos=4096)`.

---

## Run it

```bash
python test_rope_math.py     # NumPy proofs (A relative-pos, B decoupled, C orthogonality)
python mla_attention.py      # PyTorch MLA forward smoke test
```
