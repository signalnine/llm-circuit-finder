#!/usr/bin/env python3
"""
GGUF Layer Duplication Surgery

Reads a GGUF model file, duplicates transformer layers i..j-1 so they
execute twice in the forward pass, and writes a new GGUF with the
modified layer structure.

For a model with N layers, configuration (i, j) produces:
  layers 0..j-1, then layers i..j-1 again, then layers j..N-1
  Total layers: N + (j - i)

Tensor naming convention: blk.{layer_idx}.{tensor_name}
Non-block tensors (token_embd, output_norm, output) are copied as-is.
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


def get_field_value(reader: GGUFReader, key: str):
    """Extract a scalar value from a reader field."""
    field = reader.get_field(key)
    if field is None:
        return None
    return field.contents()


def duplicate_layers(input_path: str, output_path: str, dup_start: int, dup_end: int, verbose: bool = False):
    """
    Create a new GGUF with layers dup_start..dup_end-1 duplicated.

    The new layer order is:
      Original layers 0..dup_end-1
      Duplicated layers dup_start..dup_end-1  (renumbered)
      Original layers dup_end..N-1            (renumbered)
    """
    reader = GGUFReader(input_path, 'r')

    arch = get_field_value(reader, gguf.Keys.General.ARCHITECTURE)
    if arch is None:
        raise ValueError("Could not read architecture from GGUF")

    block_count_key = f'{arch}.block_count'
    orig_block_count = get_field_value(reader, block_count_key)
    if orig_block_count is None:
        raise ValueError(f"Could not read {block_count_key} from GGUF")

    n_dup = dup_end - dup_start
    new_block_count = orig_block_count + n_dup

    if verbose:
        print(f"Architecture: {arch}")
        print(f"Original layers: {orig_block_count}")
        print(f"Duplicating layers {dup_start}..{dup_end - 1} ({n_dup} layers)")
        print(f"New layer count: {new_block_count}")

    if dup_start < 0 or dup_end > orig_block_count or dup_start >= dup_end:
        raise ValueError(
            f"Invalid duplication range ({dup_start}, {dup_end}) "
            f"for model with {orig_block_count} layers"
        )

    # Build layer mapping: new_idx -> original_layer_idx
    # Phase 1: original 0..dup_end-1 keep their indices
    # Phase 2: duplicates of dup_start..dup_end-1 get indices dup_end..dup_end+n_dup-1
    # Phase 3: original dup_end..N-1 shift up by n_dup
    layer_map = {}

    for orig_idx in range(dup_end):
        layer_map[orig_idx] = orig_idx

    for k in range(n_dup):
        layer_map[dup_end + k] = dup_start + k

    for orig_idx in range(dup_end, orig_block_count):
        layer_map[orig_idx + n_dup] = orig_idx

    if verbose:
        print("Layer mapping (new -> orig):")
        for new_idx in sorted(layer_map.keys()):
            tag = " [DUP]" if (dup_end <= new_idx < dup_end + n_dup) else ""
            print(f"  new {new_idx:3d} -> orig {layer_map[new_idx]:3d}{tag}")

    # Create writer
    writer = GGUFWriter(output_path, arch=arch, endianess=reader.endianess)

    alignment = get_field_value(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment

    # Copy metadata, overriding block_count
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None

        if field.name == block_count_key:
            writer.add_key_value(field.name, new_block_count, val_type)
            if verbose:
                print(f"Modified {field.name}: {orig_block_count} -> {new_block_count}")
        else:
            val = field.contents()
            if val is not None:
                writer.add_key_value(field.name, val, val_type, sub_type=sub_type)

    # Organize tensors by type
    non_block_tensors = []
    block_tensors = {}  # orig_layer_idx -> [(suffix, tensor), ...]

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

    # Split non-block tensors into pre-block and post-block
    pre_block = []
    post_block = []
    for t in non_block_tensors:
        if 'output' in t.name:
            post_block.append(t)
        else:
            pre_block.append(t)

    # Add tensor infos in order and build write queue
    total_bytes = 0
    block_write_order = []  # (new_name, original_tensor)

    for tensor in pre_block:
        writer.add_tensor_info(
            tensor.name, tensor.data.shape, tensor.data.dtype,
            tensor.data.nbytes, tensor.tensor_type
        )
        total_bytes += tensor.n_bytes

    for new_idx in range(new_block_count):
        orig_idx = layer_map[new_idx]
        if orig_idx not in block_tensors:
            print(f"WARNING: No tensors for original layer {orig_idx}", file=sys.stderr)
            continue
        for suffix, tensor in block_tensors[orig_idx]:
            new_name = f"blk.{new_idx}.{suffix}"
            writer.add_tensor_info(
                new_name, tensor.data.shape, tensor.data.dtype,
                tensor.data.nbytes, tensor.tensor_type
            )
            total_bytes += tensor.n_bytes
            block_write_order.append((new_name, tensor))

    for tensor in post_block:
        writer.add_tensor_info(
            tensor.name, tensor.data.shape, tensor.data.dtype,
            tensor.data.nbytes, tensor.tensor_type
        )
        total_bytes += tensor.n_bytes

    # Write file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    bar = tqdm(desc="Writing GGUF", total=total_bytes, unit="B", unit_scale=True)

    for tensor in pre_block:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)

    for _, tensor in block_write_order:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)

    for tensor in post_block:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)

    bar.close()
    writer.close()

    if verbose:
        print(f"Done. Written to {output_path}")
        out_size = Path(output_path).stat().st_size / (1024**3)
        print(f"Output size: {out_size:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(
        description="Duplicate layers in a GGUF model (RYS method)"
    )
    parser.add_argument("input", help="Input GGUF file path")
    parser.add_argument("output", help="Output GGUF file path")
    parser.add_argument("-i", "--dup-start", type=int, required=True,
                        help="First layer to duplicate (inclusive)")
    parser.add_argument("-j", "--dup-end", type=int, required=True,
                        help="Last layer to duplicate (exclusive)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    duplicate_layers(args.input, args.output, args.dup_start, args.dup_end, args.verbose)


if __name__ == "__main__":
    main()
