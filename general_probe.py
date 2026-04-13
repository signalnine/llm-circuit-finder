#!/usr/bin/env python3
"""General-purpose probe loader.

Loads category JSON files (e.g. probes/general_general.json,
probes/general_languages.json) and provides simple substring-match
scoring. Each category file has shape:

    {"name": "...", "max_tokens": 400,
     "questions": [{"prompt": "...", "answer": "..."}]}
"""

import glob
import json
import os
import re


def load_probe_files(probe_dir: str) -> list[dict]:
    """Load every general_*.json file in probe_dir.

    Returns a list of category dicts with each question normalized to
    include a per-question max_tokens (falling back to the category's).
    """
    categories = []
    if not probe_dir or not os.path.isdir(probe_dir):
        return categories
    for path in sorted(glob.glob(os.path.join(probe_dir, "general_*.json"))):
        with open(path) as f:
            data = json.load(f)
        default_mt = int(data.get("max_tokens", 256))
        questions = []
        for q in data.get("questions", []):
            questions.append({
                "prompt": q["prompt"],
                "answer": q["answer"],
                "max_tokens": int(q.get("max_tokens", default_mt)),
            })
        categories.append({
            "name": data.get("name", os.path.splitext(os.path.basename(path))[0]),
            "questions": questions,
        })
    return categories


def extract_answer(response: str) -> str:
    """Strip <think>...</think> and trim whitespace."""
    if response is None:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned.strip()


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[\s,;]+", s.lower()) if t]


def score_general_response(expected: str, response: str) -> float:
    """Token-overlap score in [0, 1].

    Computes |expected_tokens ∩ response_tokens| / |expected_tokens|.
    Order is ignored; duplicates counted once. Simple and robust enough
    for short category probes (lists, orderings).
    """
    if response is None:
        return 0.0
    exp = set(_tokens(expected))
    if not exp:
        return 0.0
    got = set(_tokens(extract_answer(response)))
    return len(exp & got) / len(exp)
