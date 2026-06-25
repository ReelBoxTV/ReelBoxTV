"""
DeepSeek-R1 Multi-head Latent Attention (MLA) with decoupled (pre/post) RoPE.
=============================================================================

This is the consumer of `deepseek_rope.py`.  It shows *where* the pre-RoPE and
post-RoPE tensors live inside a real attention block, and why the split exists.

Shapes / config below follow DeepSeek-V3 / R1:

    hidden_size        d   = 7168
    num_heads          n_h = 128
    qk_nope_head_dim         = 128   (NO rope  -> "nope")
    qk_rope_head_dim         = 64    (carries rope)
    qk_head_dim              = 192   (= nope + rope, the full query/key head)
    v_head_dim               = 128
    q_lora_rank              = 1536  (query down-projection rank)
    kv_lora_rank             = 512   (key/value down-projection rank)

The two RoPE "moments":

    PRE-RoPE  : q_rope_pre  = W_QR · c_q          (per-head,  shape ..., 64)
                k_rope_pre  = W_KR · h            (SHARED 1 head, shape ..., 64)
    POST-RoPE : q_rope_post = R_m · q_rope_pre
                k_rope_post = R_m · k_rope_pre

The nope parts (q_nope, k_nope) bypass RoPE entirely (pre == post).
Final scores:  s = q_nope·k_nope  +  q_rope_post·k_rope_post.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepseek_rope import DeepseekRotaryEmbedding, RopeConfig, apply_rope


@dataclass
class MLAConfig:
    hidden_size: int = 7168
    num_heads: int = 128
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    rope_base: float = 10000.0
    use_yarn: bool = False
    attn_dropout: float = 0.0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


class DeepseekMLA(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        H, dn, dr, dv = cfg.num_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim

        # ---- Query path: hidden -> low-rank latent c_q -> per-head q ----
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)   # W_DQ
        self.q_a_norm = nn.RMSNorm(cfg.q_lora_rank)
        # up-project c_q to BOTH the nope and rope parts of every head at once
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, H * (dn + dr), bias=False)     # W_UQ + W_QR

        # ---- Key/Value path: hidden -> [latent c_kv | shared k_rope_pre] ----
        # One fused down-projection emits the kv latent AND the single shared
        # decoupled key-rope vector (this is the only rope'd thing in the KV cache).
        self.kv_a_proj = nn.Linear(cfg.hidden_size, cfg.kv_lora_rank + dr, bias=False)  # W_DKV + W_KR
        self.kv_a_norm = nn.RMSNorm(cfg.kv_lora_rank)
        # up-project c_kv to per-head k_nope and value
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank, H * (dn + dv), bias=False)         # W_UK + W_UV

        self.o_proj = nn.Linear(H * dv, cfg.hidden_size, bias=False)                    # W_O

        self.rope = DeepseekRotaryEmbedding(RopeConfig(
            rope_dim=dr, base=cfg.rope_base, use_yarn=cfg.use_yarn))

        # softmax scale.  Note: full head dim is nope+rope.
        self.scale = (cfg.qk_head_dim) ** -0.5
        self.attn_dropout = cfg.attn_dropout

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """hidden: (B, S, d).  positions: (B, S).  Returns (B, S, d)."""
        cfg = self.cfg
        B, S, _ = hidden.shape
        H, dn, dr, dv = cfg.num_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim

        # ================= QUERY =================
        c_q = self.q_a_norm(self.q_a_proj(hidden))                 # (B,S,q_lora)
        q = self.q_b_proj(c_q).view(B, S, H, dn + dr)              # (B,S,H,192)
        q = q.transpose(1, 2)                                      # (B,H,S,192)
        q_nope, q_rope_pre = torch.split(q, [dn, dr], dim=-1)      # nope=(.,128) rope_pre=(.,64)

        # ================= KEY / VALUE =================
        kv = self.kv_a_proj(hidden)                                # (B,S,kv_lora+64)
        c_kv, k_rope_pre = torch.split(kv, [cfg.kv_lora_rank, dr], dim=-1)
        c_kv = self.kv_a_norm(c_kv)
        kv_b = self.kv_b_proj(c_kv).view(B, S, H, dn + dv).transpose(1, 2)  # (B,H,S,256)
        k_nope, value = torch.split(kv_b, [dn, dv], dim=-1)        # k_nope=(.,128) value=(.,128)

        # k_rope_pre is SHARED across heads -> a single head, broadcast later.
        k_rope_pre = k_rope_pre.view(B, 1, S, dr)                  # (B,1,S,64)

        # ================= RoPE :  PRE -> POST =================
        cos, sin = self.rope(positions)                           # (B,S,64) each
        q_rope_post = apply_rope(q_rope_pre, cos, sin)            # (B,H,S,64)
        k_rope_post = apply_rope(k_rope_pre, cos, sin)            # (B,1,S,64) -> broadcast

        # broadcast the shared key-rope to every head
        k_rope_post = k_rope_post.expand(B, H, S, dr)

        # ================= ASSEMBLE FULL HEADS =================
        query = torch.cat([q_nope, q_rope_post], dim=-1)          # (B,H,S,192)
        key = torch.cat([k_nope, k_rope_post], dim=-1)            # (B,H,S,192)

        # ================= ATTENTION =================
        # s_ts = q_nope·k_nope + q_rope_post·k_rope_post   (the decoupled score)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale  # (B,H,S,S)
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if self.attn_dropout > 0 and self.training:
            probs = F.dropout(probs, p=self.attn_dropout)

        out = torch.matmul(probs, value)                          # (B,H,S,128)
        out = out.transpose(1, 2).reshape(B, S, H * dv)           # (B,S,H*128)
        return self.o_proj(out)


if __name__ == "__main__":
    # tiny smoke test on a downscaled config
    torch.manual_seed(0)
    cfg = MLAConfig(hidden_size=256, num_heads=4, qk_nope_head_dim=16,
                    qk_rope_head_dim=8, v_head_dim=16, q_lora_rank=64, kv_lora_rank=32)
    mla = DeepseekMLA(cfg)
    x = torch.randn(2, 5, cfg.hidden_size)
    pos = torch.arange(5).unsqueeze(0).expand(2, 5)
    y = mla(x, pos)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    assert y.shape == x.shape
    print("OK — MLA forward with decoupled pre/post RoPE runs.")
