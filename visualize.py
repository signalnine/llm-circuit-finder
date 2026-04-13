#!/usr/bin/env python3
"""
Visualize RYS sweep results.
Reads the JSONL output from sweep.py, prints ranked table and bar chart.
"""

import json
import sys


def load_results(path: str):
    baseline = None
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("is_baseline"):
                baseline = entry
            else:
                results.append(entry)
    return baseline, results


def print_ranked(baseline, results):
    """Print results ranked by combined delta."""
    if not baseline:
        print("No baseline found in results!")
        return

    for r in results:
        math_delta = r["math_score"] - baseline["math_score"]
        eq_delta = r["eq_score"] - baseline["eq_score"]
        r["math_delta"] = math_delta
        r["eq_delta"] = eq_delta
        r["combined"] = (math_delta * 100) + eq_delta

    ranked = sorted(results, key=lambda x: x["combined"], reverse=True)

    print(f"\nBaseline: math={baseline['math_score']:.4f}  eq={baseline['eq_score']:.2f}")
    print()
    print(f"{'Rank':>4} {'Config':>12} {'Dup':>4} "
          f"{'Math':>8} {'EQ':>8} "
          f"{'Math Δ':>9} {'EQ Δ':>8} {'Combined':>10}")
    print("-" * 80)

    for i, r in enumerate(ranked):
        config = f"({r['dup_start']},{r['dup_end']})"
        n_dup = r['dup_end'] - r['dup_start']

        if r["combined"] > 0:
            marker = "+"
        elif r["combined"] < -5:
            marker = "!"
        else:
            marker = " "

        print(f"{i+1:>4} {config:>12} {n_dup:>4} "
              f"{r['math_score']:>8.4f} {r['eq_score']:>8.2f} "
              f"{r['math_delta']:>+9.4f} {r['eq_delta']:>+8.2f} "
              f"{r['combined']:>+10.2f} {marker}")

    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        print()
        print(f"Best:  ({best['dup_start']},{best['dup_end']})  combined={best['combined']:+.2f}")
        print(f"Worst: ({worst['dup_start']},{worst['dup_end']})  combined={worst['combined']:+.2f}")


def print_bar_chart(baseline, results):
    """Print a horizontal bar chart sorted by start position."""
    if not baseline or not results:
        return

    for r in results:
        math_delta = r["math_score"] - baseline["math_score"]
        eq_delta = r["eq_score"] - baseline["eq_score"]
        r["combined"] = (math_delta * 100) + eq_delta

    ordered = sorted(results, key=lambda x: x["dup_start"])

    max_val = max(abs(r["combined"]) for r in ordered)
    if max_val == 0:
        max_val = 1

    half = 20
    print(f"\nCombined delta (baseline = |):")
    print(f"{'Config':>12} {'negative':<{half}}|{'positive':<{half}}")

    for r in ordered:
        config = f"({r['dup_start']},{r['dup_end']})"
        val = r["combined"]
        bar_len = int(abs(val) / max_val * half)

        if val >= 0:
            bar = " " * half + "|" + "#" * bar_len
        else:
            pad = half - bar_len
            bar = " " * pad + "=" * bar_len + "|"

        print(f"{config:>12} {bar} {val:+.2f}")


def print_heatmap(baseline, results):
    """Print a multi-probe heatmap from circuit_map.py results.

    Shows which layer ranges affect which probe types, enabling
    identification of task-specific circuits.
    """
    if not baseline or not results:
        print("Need baseline and results for heatmap!")
        return

    probes = ["math_score", "reasoning_score", "code_score", "swe_score"]
    probe_labels = {"math_score": "Math", "reasoning_score": "Reason",
                    "code_score": "Code", "swe_score": "SWE"}

    # Group by block_size for separate heatmaps
    by_bs = {}
    for r in results:
        bs = r.get("block_size", r.get("end", 0) - r.get("start", 0))
        if bs not in by_bs:
            by_bs[bs] = []
        by_bs[bs].append(r)

    for bs in sorted(by_bs.keys()):
        group = sorted(by_bs[bs], key=lambda x: x.get("start", 0))
        mode = group[0].get("mode", "dup")
        prefix = "del" if mode == "prune" else "dup"

        print(f"\n{'=' * 80}")
        print(f"  Block size {bs} ({prefix}) — delta vs baseline")
        print(f"{'=' * 80}")

        # Header
        print(f"{'Layer':>12}", end="")
        for p in probes:
            print(f"  {probe_labels[p]:>8}", end="")
        print(f"  {'Best':>8}")
        print("-" * (12 + len(probes) * 10 + 10))

        for r in group:
            start = r.get("start", r.get("dup_start", "?"))
            end = r.get("end", r.get("dup_end", "?"))
            label = f"{prefix}({start},{end})"
            print(f"{label:>12}", end="")

            deltas = {}
            best_probe = None
            best_delta = -999

            for p in probes:
                val = r.get(p, 0)
                base_val = baseline.get(p, 0)
                delta = val - base_val
                deltas[p] = delta

                if delta > best_delta:
                    best_delta = delta
                    best_probe = p

                # Color-code: positive = good, negative = bad
                if p in ("math_score",):
                    d_str = f"{delta:>+8.4f}"
                else:
                    d_str = f"{delta:>+8.2%}"
                print(f"  {d_str}", end="")

            print(f"  {probe_labels.get(best_probe, '?'):>8}")

        print()

    # ASCII circuit map (FINDINGS.md style)
    print(f"\n{'=' * 80}")
    print("  Circuit Map (layers where each probe improves by >2%)")
    print(f"{'=' * 80}")

    # Find the layer range across all results
    all_starts = [r.get("start", r.get("dup_start", 0)) for r in results]
    all_ends = [r.get("end", r.get("dup_end", 0)) for r in results]
    if not all_starts:
        return
    min_layer = min(all_starts)
    max_layer = max(all_ends)

    # For block_size=1 results, build per-layer per-probe delta
    bs1 = [r for r in results if r.get("block_size", 0) == 1]
    if not bs1:
        bs1 = by_bs.get(min(by_bs.keys()), [])

    # Build layer -> probe delta map
    layer_deltas = {}
    for r in bs1:
        start = r.get("start", r.get("dup_start", 0))
        for p in probes:
            delta = r.get(p, 0) - baseline.get(p, 0)
            if start not in layer_deltas:
                layer_deltas[start] = {}
            layer_deltas[start][p] = delta

    # Print scale
    scale_parts = []
    for i in range(min_layer, max_layer + 1, max(1, (max_layer - min_layer) // 10)):
        scale_parts.append(f"{i:<4}")
    print(f"{'Layer:':>10} {''.join(scale_parts)}")

    # Print per-probe line
    threshold = 0.02  # 2% improvement threshold
    for p in probes:
        label = probe_labels[p]
        line = ""
        for layer in range(min_layer, max_layer + 1):
            delta = layer_deltas.get(layer, {}).get(p, 0)
            if p == "math_score":
                # Math uses absolute delta, threshold at 0.02
                line += "#" if delta > 0.02 else ("!" if delta < -0.02 else ".")
            else:
                line += "#" if delta > threshold else ("!" if delta < -threshold else ".")
        print(f"{label:>10} {line}")

    print()
    print("  # = improves >2%    . = neutral    ! = hurts >2%")


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="Visualize sweep results")
    parser.add_argument("results", help="Path to results JSONL file")
    parser.add_argument("--heatmap", action="store_true",
                        help="Show multi-probe heatmap (for circuit_map.py output)")
    args = parser.parse_args()

    baseline, results = load_results(args.results)

    if args.heatmap:
        print_heatmap(baseline, results)
    else:
        print_ranked(baseline, results)
        print_bar_chart(baseline, results)
