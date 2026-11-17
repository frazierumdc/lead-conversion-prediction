"""End-to-end run: conversion analysis, four models, scores on the held-out test set.

    python -m leadscore.pipeline --data data/raw/leads.csv \
        --out results/metrics.json --check results/reference_metrics.json
    python -m leadscore.pipeline --synthetic --out results/demo_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from leadscore.analysis import conversion_by_channel
from leadscore.data import load_leads
from leadscore.evaluate import feature_importance, score
from leadscore.features import encode, split
from leadscore.synthetic import make_leads
from leadscore.train import default_forest, default_tree, tuned_forest, tuned_tree


def run(df: pd.DataFrame, n_jobs: int = -1) -> dict[str, Any]:
    """Run the whole analysis and return every number the README reports."""
    channels = conversion_by_channel(df)
    x, y = encode(df)
    x_train, x_test, y_train, y_test = split(x, y)

    tree = default_tree(x_train, y_train)
    pruned, tree_params = tuned_tree(x_train, y_train)
    forest = default_forest(x_train, y_train)
    tuned, forest_params = tuned_forest(x_train, y_train, n_jobs=n_jobs)

    models = {
        "decision_tree_default": tree,
        "decision_tree_tuned": pruned,
        "random_forest_default": forest,
        "random_forest_tuned": tuned,
    }
    return {
        "rows": len(df),
        "conversion_rate": round(float(y.mean()), 4),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "n_features": int(x.shape[1]),
        "channel_conversion": {
            name: {"leads": int(r.leads), "converted": int(r.converted), "rate": round(r.rate, 3)}
            for name, r in channels.iterrows()
        },
        "best_params": {
            "decision_tree_tuned": tree_params,
            "random_forest_tuned": forest_params,
        },
        "test_scores": {
            name: score(y_test, m.predict(x_test)).rounded() for name, m in models.items()
        },
        "train_accuracy": {
            name: round(score(y_train, m.predict(x_train)).accuracy, 2)
            for name, m in models.items()
        },
        "top_features_random_forest_tuned": [
            str(f) for f in feature_importance(tuned, list(x.columns)).head(5).index
        ],
    }


def compare(metrics: dict[str, Any], reference: dict[str, Any], tol: float = 0.011) -> list[str]:
    """List differences between a fresh run and the reference (original) run."""
    problems: list[str] = []
    for key, expected in reference.items():
        if key.startswith("_"):
            continue
        actual = metrics.get(key)
        if isinstance(expected, dict) and isinstance(actual, dict):
            problems += [f"{key}.{p}" for p in compare(actual, expected, tol)]
        elif isinstance(expected, float) and isinstance(actual, int | float):
            if abs(actual - expected) > tol:
                problems.append(f"{key}: expected {expected}, got {actual}")
        elif actual != expected:
            problems.append(f"{key}: expected {expected}, got {actual}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lead-conversion pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="path to the leads CSV")
    source.add_argument("--synthetic", action="store_true", help="use generated demo data")
    parser.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--check", type=Path, help="reference metrics to compare against")
    args = parser.parse_args(argv)

    df = make_leads() if args.synthetic else load_leads(args.data)
    metrics = run(df)
    if args.synthetic:
        metrics["_note"] = "Synthetic demo data. These numbers say nothing about real leads."

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["test_scores"], indent=2))
    print(f"Wrote {args.out}")

    if args.check:
        problems = compare(metrics, json.loads(args.check.read_text()))
        if problems:
            print("Differences from the reference run:\n  " + "\n  ".join(problems))
            return 1
        print("Matches the reference run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
