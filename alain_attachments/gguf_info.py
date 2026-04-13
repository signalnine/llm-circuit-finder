#!/usr/bin/env python3
"""
GGUF Model Info

Shows model metadata, layer count, per-layer tensor breakdown,
and size estimates.

Usage:
    python gguf_info.py /path/to/model.gguf
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from gguf import GGUFReader

BLK_PATTERN = re.compile(r'^blk\.(\d+)\.(.+)$')


def format_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def inspect(path: str):
    reader = GGUFReader(path, 'r')
    file_size = Path(path).stat().st_size

    # ── Metadata ──────────────────────────────────────────────
    arch = None
    block_count = None
    metadata = {}

    for field in reader.fields.values():
        if field.name.startswith('GGUF.'):
            continue
        try:
            val = field.contents()
        except Exception:
            val = "<unreadable>"
        metadata[field.name] = val

        if field.name == 'general.architecture':
            arch = val
        elif arch and field.name == f'{arch}.block_count':
            block_count = val

    print(f"File: {path}")
    print(f"Size: {format_bytes(file_size)}")
    print()

    # Print selected metadata
    interesting_keys = [
        'general.architecture', 'general.name', 'general.quantized_by',
    ]
    if arch:
        interesting_keys += [
            f'{arch}.block_count',
            f'{arch}.context_length',
            f'{arch}.embedding_length',
            f'{arch}.feed_forward_length',
            f'{arch}.attention.head_count',
            f'{arch}.attention.head_count_kv',
            f'{arch}.full_attention_interval',
            f'{arch}.ssm.conv_kernel',
            f'{arch}.ssm.state_size',
            f'{arch}.ssm.inner_size',
        ]

    print("Metadata:")
    for key in interesting_keys:
        if key in metadata:
            print(f"  {key} = {metadata[key]}")
    print()

    # ── Tensors ───────────────────────────────────────────────
    layers = defaultdict(list)       # layer_idx -> [(name, nbytes)]
    non_block = []                   # (name, nbytes)

    for tensor in reader.tensors:
        match = BLK_PATTERN.match(tensor.name)
        if match:
            layer_idx = int(match.group(1))
            suffix = match.group(2)
            layers[layer_idx].append((suffix, tensor.n_bytes))
        else:
            non_block.append((tensor.name, tensor.n_bytes))

    # ── Non-block tensors ─────────────────────────────────────
    non_block_total = sum(nb for _, nb in non_block)
    print(f"Non-block tensors: {len(non_block)} ({format_bytes(non_block_total)})")
    for name, nb in non_block:
        print(f"  {name:40s} {format_bytes(nb):>12}")
    print()

    # ── Per-layer summary ─────────────────────────────────────
    n_layers = len(layers)
    print(f"Layers: {n_layers}")
    print()

    # Detect layer types by tensor count
    type_groups = defaultdict(list)
    for idx in sorted(layers):
        tensor_names = sorted(t[0] for t in layers[idx])
        key = tuple(tensor_names)
        type_groups[key].append(idx)

    if len(type_groups) > 1:
        print(f"Layer types: {len(type_groups)} distinct configurations")
        for i, (tensors, indices) in enumerate(type_groups.items()):
            layer_size = sum(nb for _, nb in layers[indices[0]])
            # Show which layers
            ranges = _compress_ranges(indices)
            print(f"  Type {i+1}: {len(indices)} layers ({ranges})")
            print(f"    Size: {format_bytes(layer_size)} each")
            print(f"    Tensors: {', '.join(tensors)}")
        print()
    else:
        tensors = list(type_groups.keys())[0]
        sample_size = sum(nb for _, nb in layers[0])
        print(f"All layers have same structure: {len(tensors)} tensors, ~{format_bytes(sample_size)} each")
        print(f"  Tensors: {', '.join(tensors)}")
        print()

    # ── Per-layer size table ──────────────────────────────────
    print(f"{'Layer':>6}  {'Tensors':>7}  {'Size':>12}  {'Tensor names'}")
    print("-" * 80)

    block_total = 0
    for idx in sorted(layers):
        tensor_list = layers[idx]
        layer_bytes = sum(nb for _, nb in tensor_list)
        block_total += layer_bytes
        names = ", ".join(sorted(t[0] for t in tensor_list))
        print(f"{idx:>6}  {len(tensor_list):>7}  {format_bytes(layer_bytes):>12}  {names}")

    print("-" * 80)
    print(f"{'Total':>6}  {'':>7}  {format_bytes(block_total):>12}")
    print()

    # ── Summary ───────────────────────────────────────────────
    accounted = non_block_total + block_total
    overhead = file_size - accounted
    print(f"Summary:")
    print(f"  Non-block tensors: {format_bytes(non_block_total)}")
    print(f"  Block tensors:     {format_bytes(block_total)}")
    print(f"  GGUF overhead:     {format_bytes(overhead)}")
    print(f"  File total:        {format_bytes(file_size)}")

    if block_count and n_layers != block_count:
        print(f"\n  WARNING: metadata says {block_count} layers but found {n_layers} in tensors")


def _compress_ranges(indices):
    """Turn [0,1,2,5,6,9] into '0-2, 5-6, 9'."""
    if not indices:
        return ""
    sorted_idx = sorted(indices)
    ranges = []
    start = sorted_idx[0]
    end = start
    for i in sorted_idx[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = i
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)


def main():
    parser = argparse.ArgumentParser(description="Show GGUF model information")
    parser.add_argument("model", help="Path to GGUF file")
    args = parser.parse_args()
    inspect(args.model)


if __name__ == "__main__":
    main()
