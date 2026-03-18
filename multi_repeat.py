#!/usr/bin/env python3
"""
Multi-repeat GGUF surgery.

Duplicates a block of layers N times total (default 3 = original + 2 copies).

For (i=13, j=17, repeats=3) on a 40-layer model:
  0..16, 13..16, 13..16, 17..39  = 48 layers total
  The block 13-16 executes 3 times.

Usage:
    python multi_repeat.py \
        /path/to/model.gguf \
        /dev/shm/rys/triple_13_17.gguf \
        -i 13 -j 17 -n 3 -v
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

BLK_PATTERN = re.compile(r'^blk\.(\d+)\.(.+)$')


def get_field_value(reader, key):
    field = reader.get_field(key)
    if field is None:
        return None
    return field.contents()


def multi_repeat_layers(input_path, output_path, dup_start, dup_end, n_repeats, verbose=False):
    reader = GGUFReader(input_path, 'r')

    arch = get_field_value(reader, gguf.Keys.General.ARCHITECTURE)
    block_count_key = f'{arch}.block_count'
    orig_block_count = get_field_value(reader, block_count_key)

    n_block = dup_end - dup_start
    extra_copies = n_repeats - 1  # original pass counts as 1
    new_block_count = orig_block_count + (n_block * extra_copies)

    if verbose:
        print(f"Architecture: {arch}")
        print(f"Original layers: {orig_block_count}")
        print(f"Block: layers {dup_start}..{dup_end - 1} ({n_block} layers)")
        print(f"Repeats: {n_repeats}x (original + {extra_copies} copies)")
        print(f"New layer count: {new_block_count}")

    # Build layer map
    # Phase 1: original 0..dup_end-1
    # Phase 2..N: copies of dup_start..dup_end-1
    # Phase last: original dup_end..orig-1 (shifted)
    layer_map = {}

    # Phase 1: original layers up to dup_end
    for idx in range(dup_end):
        layer_map[idx] = idx

    # Phase 2+: extra copies
    offset = dup_end
    for copy in range(extra_copies):
        for k in range(n_block):
            layer_map[offset + k] = dup_start + k
        offset += n_block

    # Phase last: remaining original layers shifted
    for orig_idx in range(dup_end, orig_block_count):
        layer_map[orig_idx + (n_block * extra_copies)] = orig_idx

    assert len(layer_map) == new_block_count

    if verbose:
        path = [layer_map[i] for i in range(new_block_count)]
        print(f"Execution path ({len(path)} layers):")
        i = 0
        while i < len(path):
            run_start = path[i]
            run_end = run_start
            j = i + 1
            while j < len(path) and path[j] == run_end + 1:
                run_end = path[j]
                j += 1
            if run_start == run_end:
                print(f"  [{run_start}]")
            else:
                print(f"  [{run_start}..{run_end}]")
            i = j

    # Create writer
    writer = GGUFWriter(output_path, arch=arch, endianess=reader.endianess)

    alignment = get_field_value(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment

    # Copy metadata
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        if field.name == block_count_key:
            writer.add_key_value(field.name, new_block_count, val_type)
        else:
            val = field.contents()
            if val is not None:
                writer.add_key_value(field.name, val, val_type, sub_type=sub_type)

    # Organize tensors
    non_block_tensors = []
    block_tensors = {}
    for tensor in reader.tensors:
        match = BLK_PATTERN.match(tensor.name)
        if match:
            layer_idx = int(match.group(1))
            suffix = match.group(2)
            if layer_idx not in block_tensors:
                block_tensors[layer_idx] = []
            block_tensors[layer_idx].append((suffix, tensor))
        else:
            non_block_tensors.append(tensor)

    pre_block = [t for t in non_block_tensors if 'output' not in t.name]
    post_block = [t for t in non_block_tensors if 'output' in t.name]

    total_bytes = 0
    block_write_order = []

    for tensor in pre_block:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
        total_bytes += tensor.n_bytes

    for new_idx in range(new_block_count):
        orig_idx = layer_map[new_idx]
        for suffix, tensor in block_tensors[orig_idx]:
            new_name = f"blk.{new_idx}.{suffix}"
            writer.add_tensor_info(new_name, tensor.data.shape, tensor.data.dtype,
                                   tensor.data.nbytes, tensor.tensor_type)
            total_bytes += tensor.n_bytes
            block_write_order.append(tensor)

    for tensor in post_block:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
        total_bytes += tensor.n_bytes

    # Write
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    bar = tqdm(desc="Writing GGUF", total=total_bytes, unit="B", unit_scale=True)

    for tensor in pre_block:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)
    for tensor in block_write_order:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)
    for tensor in post_block:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)

    bar.close()
    writer.close()

    if verbose:
        size = Path(output_path).stat().st_size / (1024**3)
        print(f"Written: {output_path} ({size:.2f} GiB)")


def main():
    parser = argparse.ArgumentParser(description="Multi-repeat layer duplication")
    parser.add_argument("input", help="Input GGUF")
    parser.add_argument("output", help="Output GGUF")
    parser.add_argument("-i", "--dup-start", type=int, required=True)
    parser.add_argument("-j", "--dup-end", type=int, required=True)
    parser.add_argument("-n", "--repeats", type=int, default=3,
                        help="Total times the block executes (default: 3)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    multi_repeat_layers(args.input, args.output,
                        args.dup_start, args.dup_end,
                        args.repeats, args.verbose)


if __name__ == "__main__":
    main()
