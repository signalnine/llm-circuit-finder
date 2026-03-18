#!/usr/bin/env python3
"""
Explicit Layer Path GGUF Surgery

You provide the exact sequence of layers the model should execute.
No ambiguous range notation — just list the layers.

Examples:
    # Normal 40-layer model (identity, for testing)
    python layer_path.py model.gguf out.gguf -p 0,1,2,...,39

    # Duplicate layers 13-16 once (same as RYS with i=13,j=17)
    python layer_path.py model.gguf out.gguf -p 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,13,14,15,16,17,18,...,39

    # Repeat layer 13 four times
    python layer_path.py model.gguf out.gguf -p 0,1,...,12,13,13,13,13,14,15,...,39

    # Triple-pass layers 13-16
    python layer_path.py model.gguf out.gguf -p 0,1,...,16,13,14,15,16,13,14,15,16,17,...,39

    # Shorthand: use .. to fill in sequential ranges
    python layer_path.py model.gguf out.gguf -p 0..16,13,14,15,16,13,14,15,16,17..39

Usage:
    python layer_path.py input.gguf output.gguf -p "0..16,13,14,15,16,17..39" -v
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


def parse_layer_path(path_str: str) -> list[int]:
    """
    Parse a layer path string into a list of layer indices.

    Supports:
        - Individual numbers: 0,1,2,13,13,14
        - Ranges with ..: 0..16 expands to 0,1,2,...,16 (inclusive)
        - Mixed: 0..12,13,13,13,14..39

    Whitespace is ignored.
    """
    path_str = path_str.replace(' ', '')
    layers = []

    for part in path_str.split(','):
        part = part.strip()
        if not part:
            continue

        if '..' in part:
            # Range: start..end (inclusive)
            pieces = part.split('..')
            if len(pieces) != 2:
                raise ValueError(f"Invalid range: '{part}'. Use 'start..end'")
            start = int(pieces[0])
            end = int(pieces[1])
            if start > end:
                raise ValueError(f"Invalid range: {start}..{end} (start > end)")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))

    return layers


def build_gguf_from_path(input_path: str, output_path: str,
                         layer_path: list[int], verbose: bool = False):
    """
    Create a new GGUF where the forward pass follows the given layer path.
    """
    reader = GGUFReader(input_path, 'r')

    arch = get_field_value(reader, gguf.Keys.General.ARCHITECTURE)
    block_count_key = f'{arch}.block_count'
    orig_block_count = get_field_value(reader, block_count_key)

    # Validate all layer indices
    for idx in layer_path:
        if idx < 0 or idx >= orig_block_count:
            raise ValueError(
                f"Layer {idx} out of range (model has {orig_block_count} layers, 0..{orig_block_count-1})"
            )

    new_block_count = len(layer_path)

    if verbose:
        print(f"Architecture: {arch}")
        print(f"Original layers: {orig_block_count}")
        print(f"New layer count: {new_block_count}")
        print(f"Layer path: {layer_path}")

        # Show which layers are repeated
        from collections import Counter
        counts = Counter(layer_path)
        repeated = {k: v for k, v in counts.items() if v > 1}
        if repeated:
            print(f"Repeated layers: {dict(sorted(repeated.items()))}")
        else:
            print("No repeated layers (just a reorder)")

    # layer_map: new_position -> original_layer_index
    layer_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(layer_path)}

    # Create writer
    writer = GGUFWriter(output_path, arch=arch, endianess=reader.endianess)

    alignment = get_field_value(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment

    # Copy metadata, override block count
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

    # Organize tensors by layer
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

    # Add tensor infos and build write order
    total_bytes = 0
    block_write_order = []

    for tensor in pre_block:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
        total_bytes += tensor.n_bytes

    for new_idx in range(new_block_count):
        orig_idx = layer_map[new_idx]
        if orig_idx not in block_tensors:
            raise ValueError(f"No tensors found for original layer {orig_idx}")
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
        out_size = Path(output_path).stat().st_size / (1024**3)
        print(f"Done. Output: {out_size:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(
        description="Build GGUF with explicit layer execution path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Duplicate layers 13-16 once (RYS style)
  %(prog)s model.gguf out.gguf -p "0..16,13,14,15,16,17..39"

  # Triple-pass layers 13-16
  %(prog)s model.gguf out.gguf -p "0..16,13,14,15,16,13,14,15,16,17..39"

  # Repeat just layer 13 four times
  %(prog)s model.gguf out.gguf -p "0..12,13,13,13,13,14..39"

  # Skip layer 5 entirely
  %(prog)s model.gguf out.gguf -p "0..4,6..39"
        """
    )
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("output", help="Output GGUF file")
    parser.add_argument("-p", "--path", required=True,
                        help="Layer execution path (e.g. '0..16,13,14,15,16,17..39')")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    layer_path = parse_layer_path(args.path)
    print(f"Model: {args.input}")
    print(f"Output: {args.output}")
    print(f"Layer path ({len(layer_path)} layers): {layer_path}")

    build_gguf_from_path(args.input, args.output, layer_path, args.verbose)


if __name__ == "__main__":
    main()
