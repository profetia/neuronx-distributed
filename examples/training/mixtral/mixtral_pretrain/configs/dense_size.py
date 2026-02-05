#!/usr/bin/env python3
"""
Dense (non-MoE) parameter estimator with GQA support (num_key_value_heads).

Works well for LLaMA-like configs from Hugging Face.

Usage:
  python dense_params.py config.json
  cat config.json | python dense_params.py
  python dense_params.py --json '{"hidden_size":4096,...}'
  python dense_params.py config.json --json-out

Notes:
- Attention params assume separate Q, K, V, O projections.
- MLP params default to 3 matrices (gate/up/down), like LLaMA.
- Norm params assume RMSNorm weights only (no bias).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DenseConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    tie_word_embeddings: bool = False

    # assumptions / toggles
    include_bias: bool = False
    mlp_matrices: int = 3          # LLaMA default: gate/up/down
    norms_per_layer: int = 2       # typically: input + post-attn (or pre-mlp)
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


def build_cfg(raw: Dict[str, Any], args: argparse.Namespace) -> DenseConfig:
    # Some HF configs include "attention_bias": false/true. We keep CLI as the source of truth,
    # but you can uncomment the next two lines if you want to default from config:
    # cfg_attention_bias = bool(raw.get("attention_bias", False))
    # include_bias = bool(args.include_bias or cfg_attention_bias)
    include_bias = bool(args.include_bias)

    return DenseConfig(
        vocab_size=int(_require(raw, "vocab_size")),
        hidden_size=int(_require(raw, "hidden_size")),
        intermediate_size=int(_require(raw, "intermediate_size")),
        num_hidden_layers=int(_require(raw, "num_hidden_layers")),
        num_attention_heads=int(_require(raw, "num_attention_heads")),
        num_key_value_heads=int(raw.get("num_key_value_heads", raw.get("num_attention_heads"))),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        include_bias=include_bias,
        mlp_matrices=int(args.mlp_mats),
        norms_per_layer=int(args.norms_per_layer),
        include_final_norm=not bool(args.no_final_norm),
    )


def count_params(cfg: DenseConfig) -> Dict[str, int]:
    d = cfg.hidden_size
    m = cfg.intermediate_size
    L = cfg.num_hidden_layers
    V = cfg.vocab_size

    # --- Attention weights (GQA-aware) ---
    if d % cfg.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({d}) must be divisible by num_attention_heads ({cfg.num_attention_heads})"
        )
    head_dim = d // cfg.num_attention_heads
    kv_dim = cfg.num_key_value_heads * head_dim  # output width of K and V projections

    # Wq: (d x d), Wk: (d x kv_dim), Wv: (d x kv_dim), Wo: (d x d)
    attn_w = (d * d) + (d * kv_dim) + (d * kv_dim) + (d * d)  # 2d^2 + 2d*kv_dim

    # Optional biases: out dims are d, kv_dim, kv_dim, d
    attn_b = (d + kv_dim + kv_dim + d) if cfg.include_bias else 0
    attn_total = attn_w + attn_b

    # --- Dense MLP (not per-expert) ---
    # LLaMA gated MLP default: 3 matrices => 3*d*m
    mlp_w = cfg.mlp_matrices * d * m

    if cfg.include_bias:
        # Common cases:
        # 3 mats: two (d->m) biases m each + one (m->d) bias d => 2*m + d
        # 2 mats: one (d->m) bias m + one (m->d) bias d => m + d
        if cfg.mlp_matrices == 3:
            mlp_b = 2 * m + d
        elif cfg.mlp_matrices == 2:
            mlp_b = m + d
        else:
            # heuristic split: first half map to m, remainder map to d
            a = cfg.mlp_matrices // 2
            b = cfg.mlp_matrices - a
            mlp_b = a * m + b * d
    else:
        mlp_b = 0

    mlp_total = mlp_w + mlp_b

    # --- Norms ---
    norm_total_per_layer = cfg.norms_per_layer * d  # RMSNorm weights only

    # --- Per-layer totals ---
    layer_total = attn_total + mlp_total + norm_total_per_layer

    # --- Stack totals ---
    stack_total = L * layer_total

    # --- Embedding + LM head ---
    embed = V * d
    lm_head_w = 0 if cfg.tie_word_embeddings else (d * V)
    lm_head_b = V if (cfg.include_bias and not cfg.tie_word_embeddings) else 0
    lm_head_total = lm_head_w + lm_head_b

    # Final norm
    final_norm = d if cfg.include_final_norm else 0

    total_params = stack_total + embed + lm_head_total + final_norm

    return {
        "head_dim": head_dim,
        "kv_dim": kv_dim,
        "attn_per_layer": attn_total,
        "mlp_per_layer": mlp_total,
        "norms_per_layer": norm_total_per_layer,
        "layer_total": layer_total,
        "stack_total": stack_total,
        "embed": embed,
        "lm_head": lm_head_total,
        "final_norm": final_norm,
        "total_params": total_params,
        # For dense models, "activated stack" == full stack (no expert sparsity).
        "activated_stack_only": stack_total,
        "activated_including_embed_and_head": stack_total + embed + lm_head_total + final_norm,
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
    p = argparse.ArgumentParser(description="Estimate params for dense transformer configs (GQA-aware).")
    p.add_argument("path", nargs="?", help="Path to config JSON. If omitted, reads JSON from stdin.")
    p.add_argument("--json", dest="json_str", help="Raw JSON string (overrides path/stdin).")
    p.add_argument("--include-bias", action="store_true", help="Count biases in linear layers.")
    p.add_argument("--mlp-mats", type=int, default=3, help="Number of MLP matrices (default: 3 for LLaMA gated MLP).")
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

    print("=== Dense transformer param estimates (GQA-aware) ===")
    print(f"head_dim={out['head_dim']}, kv_dim={out['kv_dim']}")
    print(f"Total params: {fmt(out['total_params'])} ({out['total_params']:,})")
    print()
    print("Breakdown:")
    print(f"  - stack_total: {fmt(out['stack_total'])} ({out['stack_total']:,})")
    print(f"    - attn_per_layer:  {fmt(out['attn_per_layer'])} ({out['attn_per_layer']:,})")
    print(f"    - mlp_per_layer:   {fmt(out['mlp_per_layer'])} ({out['mlp_per_layer']:,})")
    print(f"    - norms_per_layer: {fmt(out['norms_per_layer'])} ({out['norms_per_layer']:,})")
    print(f"    - layer_total:     {fmt(out['layer_total'])} ({out['layer_total']:,})")
    print(f"  - embed:     {fmt(out['embed'])} ({out['embed']:,})")
    print(f"  - lm_head:   {fmt(out['lm_head'])} ({out['lm_head']:,})")
    print(f"  - final_norm:{fmt(out['final_norm'])} ({out['final_norm']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())