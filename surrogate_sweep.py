#!/usr/bin/env python3
"""
Surrogate Model Sweep — ML-guided search for optimal layer configurations.

Trains a surrogate model on existing JSONL sweep data to predict which configs
are promising, then evaluates only the top-N candidates. Inspired by David Ng's
XGBoost approach in RYS-II.

Active learning loop:
1. Load all existing JSONL results as training data
2. Train surrogate (Ridge → RandomForest → XGBoost)
3. Generate all candidate configs at stride=1
4. Predict scores, rank by predicted improvement
5. Evaluate top-K with real probes
6. Add results, retrain, repeat

Usage:
    # Train on existing data and evaluate top candidates
    python surrogate_sweep.py \
        --model /path/to/model.gguf \
        --llama-server /path/to/llama-server \
        --training-data qwen3-coder-codeswe-sweep.jsonl qwen3-coder-prune-sweep.jsonl \
        --mode dup \
        --top-k 10 \
        --iterations 3 \
        --results surrogate_results.jsonl

    # Predict-only mode (no evaluation, just rank candidates)
    python surrogate_sweep.py \
        --training-data *.jsonl \
        --predict-only \
        --top-k 20
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from sweep_lib import (
    DEFAULT_PORT, get_layer_count, create_modified_gguf, build_layer_path,
    wait_for_server, start_server, stop_server, dump_server_log,
    run_evaluation, load_jsonl, append_jsonl,
)


def load_training_data(jsonl_paths: list[str]) -> list[dict]:
    """Load and normalize training data from multiple JSONL files."""
    all_data = []
    for path in jsonl_paths:
        baseline, results = load_jsonl(path)
        if not baseline:
            continue
        for r in results:
            # Normalize both schema variants
            entry = {
                "baseline": baseline,
                "result": r,
            }
            # Extract mode/start/end from either schema
            if "mode" in r:
                entry["mode"] = r["mode"]
                entry["start"] = r["start"]
                entry["end"] = r["end"]
            elif "dup_start" in r and r.get("dup_start", -1) >= 0:
                entry["mode"] = "dup"
                entry["start"] = r["dup_start"]
                entry["end"] = r["dup_end"]
            elif "prune_start" in r and r.get("prune_start", -1) >= 0:
                entry["mode"] = "prune"
                entry["start"] = r["prune_start"]
                entry["end"] = r["prune_end"]
            else:
                continue

            # Compute deltas
            entry["code_delta"] = r.get("code_score", 0) - baseline.get("code_score", 0)
            entry["swe_delta"] = r.get("swe_score", 0) - baseline.get("swe_score", 0)
            entry["math_delta"] = r.get("math_score", 0) - baseline.get("math_score", 0)
            entry["reasoning_delta"] = r.get("reasoning_score", 0) - baseline.get("reasoning_score", 0)

            # Get n_layers from baseline or result
            n_layers = baseline.get("n_layers") or r.get("n_layers")
            if n_layers:
                entry["n_layers"] = n_layers

            all_data.append(entry)

    return all_data


def featurize(entries: list[dict], n_layers: int = None) -> np.ndarray:
    """Convert entries to feature matrix.

    Features:
        0: mode (0=prune, 1=dup)
        1: start
        2: end
        3: block_size
        4: start_frac (start / n_layers)
        5: end_frac (end / n_layers)
        6: mid_frac ((start + end) / (2 * n_layers))
    """
    X = []
    for e in entries:
        nl = e.get("n_layers", n_layers) or 48  # fallback
        start = e["start"]
        end = e["end"]
        bs = end - start
        X.append([
            0 if e["mode"] == "prune" else 1,
            start,
            end,
            bs,
            start / nl,
            end / nl,
            (start + end) / (2 * nl),
        ])
    return np.array(X)


def train_surrogate(X: np.ndarray, y: np.ndarray, model_type: str = "auto"):
    """Train a surrogate model. Returns (model, model_type_used, cv_score).

    Tries models in order of complexity, picks best CV score.
    """
    from sklearn.model_selection import cross_val_score

    models = {}

    # Always try Ridge (works with any data size)
    from sklearn.linear_model import Ridge
    models["ridge"] = Ridge(alpha=1.0)

    if len(X) >= 20:
        from sklearn.ensemble import RandomForestRegressor
        models["rf"] = RandomForestRegressor(
            n_estimators=50, max_depth=4, random_state=42
        )

    if len(X) >= 50:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            models["gbr"] = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42
            )
        except ImportError:
            pass

    if model_type != "auto" and model_type in models:
        models = {model_type: models[model_type]}

    best_model = None
    best_type = None
    best_score = -np.inf

    for name, model in models.items():
        cv = min(5, len(X))
        if cv < 2:
            cv = 2
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
            mean_score = scores.mean()
            print(f"  {name}: CV MSE = {-mean_score:.6f} (±{scores.std():.6f})")
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_type = name
        except Exception as e:
            print(f"  {name}: failed ({e})")

    if best_model is not None:
        best_model.fit(X, y)
        print(f"  Selected: {best_type} (CV MSE = {-best_score:.6f})")

    return best_model, best_type, -best_score


def generate_all_candidates(n_layers: int, mode: str,
                            block_sizes: list[int] = None,
                            start_min: int = 4) -> list[dict]:
    """Generate all possible candidate configs at stride=1."""
    if block_sizes is None:
        block_sizes = list(range(1, 6))  # 1 through 5

    candidates = []
    for bs in block_sizes:
        for start in range(start_min, n_layers - bs - 2 + 1):
            end = start + bs
            candidates.append({
                "mode": mode,
                "start": start,
                "end": end,
            })
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Surrogate Model Sweep")
    parser.add_argument("--model", help="Path to input GGUF model (not needed for predict-only)")
    parser.add_argument("--llama-server", help="Path to llama-server binary")
    parser.add_argument("--training-data", nargs="+", required=True,
                        help="JSONL files with existing sweep results")
    parser.add_argument("--results", default="surrogate_results.jsonl")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mode", choices=["dup", "prune"], default="dup",
                        help="Surgery mode for new candidates (default: dup)")
    parser.add_argument("--target", default="code_delta",
                        choices=["code_delta", "swe_delta", "math_delta", "reasoning_delta"],
                        help="Target metric to optimize (default: code_delta)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top candidates to evaluate per iteration")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of active learning iterations")
    parser.add_argument("--predict-only", action="store_true",
                        help="Only predict and rank, don't evaluate")
    parser.add_argument("--model-type", default="auto",
                        choices=["auto", "ridge", "rf", "gbr"],
                        help="Surrogate model type (default: auto)")
    parser.add_argument("--tmpdir", default="/dev/shm/surrogate")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass to llama-server (must be last)")
    args = parser.parse_args()

    if not args.predict_only and (not args.model or not args.llama_server):
        parser.error("--model and --llama-server required unless --predict-only")

    # Load training data
    print("Loading training data...", flush=True)
    training = load_training_data(args.training_data)
    print(f"  {len(training)} data points from {len(args.training_data)} files")

    if len(training) < 5:
        print("ERROR: Need at least 5 data points for surrogate training", file=sys.stderr)
        sys.exit(1)

    # Determine n_layers
    n_layers = None
    if args.model:
        n_layers, arch = get_layer_count(args.model)
        print(f"  Target model: {arch}, {n_layers} layers")
    else:
        # Infer from training data
        for t in training:
            if "n_layers" in t:
                n_layers = t["n_layers"]
                break
        if n_layers is None:
            n_layers = max(t["end"] for t in training) + 4
        print(f"  Inferred n_layers: {n_layers}")

    # Filter training data to matching mode
    mode_training = [t for t in training if t["mode"] == args.mode]
    print(f"  {len(mode_training)} entries match mode '{args.mode}'")

    if len(mode_training) < 3:
        print(f"  WARNING: Very few entries for mode '{args.mode}', using all data")
        mode_training = training

    # Featurize
    X = featurize(mode_training, n_layers)
    y = np.array([t[args.target] for t in mode_training])

    print(f"\nTarget: {args.target}")
    print(f"  Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    print(f"  Best observed: {y.max():.4f}, Worst: {y.min():.4f}")

    # Generate all candidates
    candidates = generate_all_candidates(n_layers, args.mode)
    # Remove already-seen configs
    seen = {(t["mode"], t["start"], t["end"]) for t in training}
    candidates = [c for c in candidates if (c["mode"], c["start"], c["end"]) not in seen]
    print(f"\n{len(candidates)} unseen candidates")

    if not candidates:
        print("No unseen candidates remaining!")
        sys.exit(0)

    # Active learning loop
    all_new_results = []
    results_path = Path(args.results) if not args.predict_only else None

    for iteration in range(args.iterations):
        print(f"\n{'=' * 60}")
        print(f"  Iteration {iteration + 1}/{args.iterations}")
        print(f"  Training set: {len(X)} points, Candidates: {len(candidates)}")
        print(f"{'=' * 60}")

        # Train surrogate
        print("\nTraining surrogate model...")
        model, model_type, cv_mse = train_surrogate(X, y, args.model_type)
        if model is None:
            print("ERROR: All models failed to train", file=sys.stderr)
            break

        # Predict on candidates
        X_cand = featurize(candidates, n_layers)
        y_pred = model.predict(X_cand)

        # Rank by predicted score (descending)
        ranking = np.argsort(y_pred)[::-1]

        print(f"\nTop {args.top_k} predicted candidates:")
        print(f"  {'Rank':>4} {'Config':>16} {'Predicted':>10}")
        print(f"  {'-' * 34}")
        for i, idx in enumerate(ranking[:args.top_k]):
            c = candidates[idx]
            prefix = "del" if c["mode"] == "prune" else "dup"
            label = f"{prefix}({c['start']},{c['end']})"
            print(f"  {i+1:>4} {label:>16} {y_pred[idx]:>+10.4f}")

        if args.predict_only:
            # Also show bottom 5
            print(f"\nBottom 5 predicted candidates:")
            for i, idx in enumerate(ranking[-5:]):
                c = candidates[idx]
                prefix = "del" if c["mode"] == "prune" else "dup"
                label = f"{prefix}({c['start']},{c['end']})"
                print(f"  {len(ranking)-4+i:>4} {label:>16} {y_pred[idx]:>+10.4f}")
            continue

        # Evaluate top-K
        tmpdir = Path(args.tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)

        # Load baseline for this model
        baseline, existing = load_jsonl(str(results_path)) if results_path and results_path.exists() else (None, [])

        if not args.skip_baseline and baseline is None:
            print("\n>>> Running BASELINE evaluation...")
            proc = start_server(args.llama_server, str(args.model), args.port, args.server_args)
            try:
                if not wait_for_server(args.port):
                    print("ERROR: Server failed to start", file=sys.stderr)
                    dump_server_log(proc)
                    stop_server(proc)
                    sys.exit(1)
                eval_result = run_evaluation(args.port)
                baseline = {"is_baseline": True, **eval_result,
                            "timestamp": datetime.now().isoformat()}
                append_jsonl(str(results_path), baseline)
            finally:
                stop_server(proc)

        top_indices = ranking[:args.top_k]
        evaluated = []

        for rank, idx in enumerate(top_indices):
            config = candidates[idx]
            prefix = "del" if config["mode"] == "prune" else "dup"
            label = f"{prefix}({config['start']},{config['end']})"
            print(f"\n>>> [{rank+1}/{args.top_k}] Evaluating {label} "
                  f"(predicted {args.target}={y_pred[idx]:+.4f})...")

            path_str = build_layer_path(config["mode"], config["start"],
                                        config["end"], n_layers)
            modified_path = tmpdir / f"surr_{config['mode']}_{config['start']}_{config['end']}.gguf"

            if not create_modified_gguf(str(args.model), str(modified_path), path_str):
                continue

            proc = start_server(args.llama_server, str(modified_path), args.port, args.server_args)
            try:
                if not wait_for_server(args.port):
                    print(f"  ERROR: Server failed", file=sys.stderr)
                    dump_server_log(proc)
                    continue

                eval_result = run_evaluation(args.port)
                entry = {
                    **config, **eval_result,
                    "predicted": float(y_pred[idx]),
                    "iteration": iteration + 1,
                    "timestamp": datetime.now().isoformat(),
                }

                actual_delta = eval_result.get("code_score", 0) - (baseline or {}).get("code_score", 0)
                entry["actual_code_delta"] = actual_delta

                append_jsonl(str(results_path), entry)
                all_new_results.append(entry)
                evaluated.append((config, eval_result, y_pred[idx]))

                print(f"  Predicted: {y_pred[idx]:+.4f}, Actual {args.target}: {actual_delta:+.4f}")

            finally:
                stop_server(proc)
                if modified_path.exists():
                    modified_path.unlink()

        # Update training data with new results
        if evaluated and baseline:
            for config, eval_result, _ in evaluated:
                new_entry = {
                    "mode": config["mode"],
                    "start": config["start"],
                    "end": config["end"],
                    "n_layers": n_layers,
                    "code_delta": eval_result.get("code_score", 0) - baseline.get("code_score", 0),
                    "swe_delta": eval_result.get("swe_score", 0) - baseline.get("swe_score", 0),
                    "math_delta": eval_result.get("math_score", 0) - baseline.get("math_score", 0),
                    "reasoning_delta": eval_result.get("reasoning_score", 0) - baseline.get("reasoning_score", 0),
                }
                mode_training.append(new_entry)

            X = featurize(mode_training, n_layers)
            y = np.array([t[args.target] for t in mode_training])

            # Remove evaluated candidates
            eval_keys = {(c["mode"], c["start"], c["end"]) for c, _, _ in evaluated}
            candidates = [c for c in candidates
                          if (c["mode"], c["start"], c["end"]) not in eval_keys]

        # Calibration report
        if evaluated:
            print(f"\n  Calibration (iteration {iteration + 1}):")
            preds = [p for _, _, p in evaluated]
            actuals = [e.get("code_score", 0) - (baseline or {}).get("code_score", 0)
                       for _, e, _ in evaluated]
            pred_rank = np.argsort(np.argsort(preds)[::-1])
            actual_rank = np.argsort(np.argsort(actuals)[::-1])

            if len(preds) > 1:
                rank_corr = np.corrcoef(pred_rank, actual_rank)[0, 1]
                print(f"  Rank correlation: {rank_corr:.3f}")
            mse = np.mean((np.array(preds) - np.array(actuals)) ** 2)
            print(f"  MSE (predicted vs actual): {mse:.6f}")

    # Final summary
    if all_new_results:
        print(f"\n{'=' * 60}")
        print(f"  Surrogate sweep complete — {len(all_new_results)} new evaluations")
        print(f"{'=' * 60}")
        for r in sorted(all_new_results, key=lambda x: x.get("actual_code_delta", 0), reverse=True):
            prefix = "del" if r["mode"] == "prune" else "dup"
            label = f"{prefix}({r['start']},{r['end']})"
            actual = r.get("actual_code_delta", 0)
            pred = r.get("predicted", 0)
            print(f"  {label:>16}  predicted={pred:+.4f}  actual={actual:+.4f}  "
                  f"iter={r.get('iteration', '?')}")


if __name__ == "__main__":
    main()
