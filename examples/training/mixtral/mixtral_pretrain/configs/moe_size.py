#!/usr/bin/env python3
"""
Mixtral-style MoE parameter estimator with GQA support (num_key_value_heads).

Usage:
  python moe_params.py config.json
  cat config.json | python moe_params.py
  python moe_params.py --json '{"hidden_size":2048,...}'
  python moe_params.py config.json --json-out
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MoEConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    tie_word_embeddings: bool = False

    # assumptions / toggles
    include_bias: bool = False
    mlp_matrices_per_expert: int = 3   # Mixtral default: gate/up/down
    norms_per_layer: int = 2
    include_final_norm: bool = True


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: '{key}'")
    return d[key]


def load_json(args: argparse.Namespace) -> Dict[str, Any]:
    if args.json_str:
        return json.loads(args.json_str)

    if args.path:
        with open(args.path, "r", encoding="utf-8") as f:
            return json.load(f)

    data = sys.stdin.read().strip()
    if not data:
        raise ValueError("No input provided. Pass a file path, pipe JSON via stdin, or use --json.")
    return json.loads(data)


def build_cfg(raw: Dict[str, Any], args: argparse.Namespace) -> MoEConfig:
    return MoEConfig(
        vocab_size=int(_require(raw, "vocab_size")),
        hidden_size=int(_require(raw, "hidden_size")),
        intermediate_size=int(_require(raw, "intermediate_size")),
        num_hidden_layers=int(_require(raw, "num_hidden_layers")),
        num_attention_heads=int(_require(raw, "num_attention_heads")),
        num_key_value_heads=int(raw.get("num_key_value_heads", raw.get("num_attention_heads"))),
        num_local_experts=int(_require(raw, "num_local_experts")),
        num_experts_per_tok=int(_require(raw, "num_experts_per_tok")),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        include_bias=bool(args.include_bias),
        mlp_matrices_per_expert=int(args.mlp_mats),
        norms_per_layer=int(args.norms_per_layer),
        include_final_norm=not bool(args.no_final_norm),
    )


def count_params(cfg: MoEConfig) -> Dict[str, int]:
    d = cfg.hidden_size
    m = cfg.intermediate_size
    L = cfg.num_hidden_layers
    E = cfg.num_local_experts
    k = cfg.num_experts_per_tok
    V = cfg.vocab_size

    # --- Attention weights (GQA-aware) ---
    # head_dim = d / num_attention_heads
    if d % cfg.num_attention_heads != 0:
        raise ValueError(f"hidden_size ({d}) must be divisible by num_attention_heads ({cfg.num_attention_heads})")
    head_dim = d // cfg.num_attention_heads
    kv_dim = cfg.num_key_value_heads * head_dim  # output width of K and V projections

    # Wq: (d x d), Wk: (d x kv_dim), Wv: (d x kv_dim), Wo: (d x d)
    attn_w = (d * d) + (d * kv_dim) + (d * kv_dim) + (d * d)  # = 2d^2 + 2d*kv_dim

    # Optional biases: out dims are d, kv_dim, kv_dim, d
    attn_b = (d + kv_dim + kv_dim + d) if cfg.include_bias else 0
    attn_total = attn_w + attn_b

    # --- MoE FFN per expert ---
    # Mixtral default: 3 matrices => 3*d*m
    expert_w = cfg.mlp_matrices_per_expert * d * m

    if cfg.include_bias:
        # Common cases:
        # 3 mats: two (d->m) biases m each + one (m->d) bias d => 2*m + d
        # 2 mats: one (d->m) bias m + one (m->d) bias d => m + d
        if cfg.mlp_matrices_per_expert == 3:
            expert_b = 2 * m + d
        elif cfg.mlp_matrices_per_expert == 2:
            expert_b = m + d
        else:
            a = cfg.mlp_matrices_per_expert // 2
            b = cfg.mlp_matrices_per_expert - a
            expert_b = a * m + b * d
    else:
        expert_b = 0

    expert_total = expert_w + expert_b
    moe_ffn_total_per_layer = E * expert_total
    moe_ffn_active_per_layer = k * expert_total

    # --- Router ---
    router_w = d * E
    router_b = E if cfg.include_bias else 0
    router_total = router_w + router_b

    # --- Norms ---
    norm_total_per_layer = cfg.norms_per_layer * d  # RMSNorm weights only

    # --- Per-layer totals ---
    layer_total = attn_total + router_total + norm_total_per_layer + moe_ffn_total_per_layer
    layer_active = attn_total + router_total + norm_total_per_layer + moe_ffn_active_per_layer

    # --- Stack totals ---
    stack_total = L * layer_total
    stack_active = L * layer_active

    # --- Embedding + LM head ---
    embed = V * d
    lm_head = 0 if cfg.tie_word_embeddings else (d * V)
    lm_head_b = V if (cfg.include_bias and not cfg.tie_word_embeddings) else 0
    lm_head_total = lm_head + lm_head_b

    # Final norm
    final_norm = d if cfg.include_final_norm else 0

    total_params = stack_total + embed + lm_head_total + final_norm

    return {
        "head_dim": head_dim,
        "kv_dim": kv_dim,
        "attn_per_layer": attn_total,
        "router_per_layer": router_total,
        "norms_per_layer": norm_total_per_layer,
        "expert_params": expert_total,
        "moe_ffn_total_per_layer": moe_ffn_total_per_layer,
        "moe_ffn_active_per_layer": moe_ffn_active_per_layer,
        "layer_total": layer_total,
        "layer_active": layer_active,
        "stack_total": stack_total,
        "stack_active": stack_active,
        "embed": embed,
        "lm_head": lm_head_total,
        "final_norm": final_norm,
        "total_params": total_params,
        "activated_stack_only": stack_active,
        "activated_including_embed_and_head": stack_active + embed + lm_head_total + final_norm,
    }


def fmt(n: int) -> str:
    if n >= 10**9:
        return f"{n/1e9:.3f}B"
    if n >= 10**6:
        return f"{n/1e6:.3f}M"
    if n >= 10**3:
        return f"{n/1e3:.3f}K"
    return str(n)


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate total/activated params for Mixtral-style MoE configs (GQA-aware).")
    p.add_argument("path", nargs="?", help="Path to config JSON. If omitted, reads JSON from stdin.")
    p.add_argument("--json", dest="json_str", help="Raw JSON string (overrides path/stdin).")
    p.add_argument("--include-bias", action="store_true", help="Count biases in linear layers.")
    p.add_argument("--mlp-mats", type=int, default=3, help="Number of matrices per expert MLP (default: 3).")
    p.add_argument("--norms-per-layer", type=int, default=2, help="RMSNorm layers per transformer layer (default: 2).")
    p.add_argument("--no-final-norm", action="store_true", help="Do not count final norm params.")
    p.add_argument("--json-out", action="store_true", help="Print JSON output instead of a human summary.")
    args = p.parse_args()

    raw = load_json(args)
    cfg = build_cfg(raw, args)
    out = count_params(cfg)

    if args.json_out:
        print(json.dumps(out, indent=2))
        return 0

    print("=== Mixtral-style MoE param estimates (GQA-aware) ===")
    print(f"head_dim={out['head_dim']}, kv_dim={out['kv_dim']}")
    print(f"Total params: {fmt(out['total_params'])} ({out['total_params']:,})")
    print()
    print("Activated params (common definitions):")
    print(f"  - Transformer stack only: {fmt(out['activated_stack_only'])} ({out['activated_stack_only']:,})")
    print(f"  - Including embedding + lm_head: {fmt(out['activated_including_embed_and_head'])} ({out['activated_including_embed_and_head']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
