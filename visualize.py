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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results.jsonl>")
        sys.exit(1)

    baseline, results = load_results(sys.argv[1])
    print_ranked(baseline, results)
    print_bar_chart(baseline, results)
