#!/usr/bin/env python3
"""
Math Probe for RYS Layer Duplication Sweep

Hard arithmetic questions where the model must guess the answer
without chain-of-thought. Scored with Ng's partial-credit function.
"""

import json
import math
import random


def calculate_score(actual, estimate):
    """
    Ng's partial-credit scoring function.
    Pads shorter answers, penalizes proportionally.
    Returns 0-1 score.
    """
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(estimate))
    except (ValueError, OverflowError):
        return 0.0

    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")
    padding_size = max_length - min(len(actual_str), len(estimate_str))

    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)

    if max(actual_int, estimate_int) == 0:
        return 0.0

    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor

    return max(0.0, min(score, 1.0))


def generate_math_questions(seed=42):
    """
    Generate hard arithmetic questions with known answers.
    Mix of operations to test different numeric intuition.
    Returns list of (question_text, correct_answer) tuples.
    """
    rng = random.Random(seed)
    questions = []

    # Cube roots of large numbers (compute perfect cubes, ask for root)
    for _ in range(4):
        root = rng.randint(20000, 50000)
        cube = root ** 3
        questions.append((
            f"What is the cube root of {cube}? "
            f"Answer with just the number, no explanation.",
            root
        ))

    # Large multiplications
    for _ in range(4):
        a = rng.randint(100000, 999999)
        b = rng.randint(100000, 999999)
        product = a * b
        questions.append((
            f"What is {a} multiplied by {b}? "
            f"Answer with just the number, no explanation.",
            product
        ))

    # Square roots of large numbers (perfect squares)
    for _ in range(4):
        root = rng.randint(50000, 200000)
        square = root ** 2
        questions.append((
            f"What is the square root of {square}? "
            f"Answer with just the number, no explanation.",
            root
        ))

    # Mixed: cube root multiplied by a number
    for _ in range(4):
        root = rng.randint(100, 999)
        cube = root ** 3
        multiplier = rng.randint(10, 99)
        answer = root * multiplier
        questions.append((
            f"What is the cube root of {cube}, multiplied by {multiplier}? "
            f"Answer with just the number, no explanation.",
            answer
        ))

    return questions


def parse_number_from_response(response: str) -> int | None:
    """
    Extract the first integer from a model response.
    Handles common LLM quirks: commas in numbers, trailing text, etc.
    """
    import re

    # Clean up common formatting
    text = response.strip()

    # Try to find a number (possibly with commas)
    # Match negative or positive integers, possibly with commas
    patterns = [
        r'[-+]?[\d,]+',  # numbers with optional commas
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Take the first/longest match
            num_str = max(matches, key=len)
            num_str = num_str.replace(',', '')
            try:
                return int(num_str)
            except ValueError:
                continue

    return None


def score_math_response(question_answer: int, response: str) -> float:
    """Score a single math response."""
    parsed = parse_number_from_response(response)
    if parsed is None:
        return 0.0
    return calculate_score(question_answer, parsed)


# Pre-generated questions for consistency across runs
MATH_QUESTIONS = generate_math_questions(seed=42)


if __name__ == "__main__":
    # Print the questions and answers for verification
    print("Math Probe Questions:")
    print("=" * 60)
    for i, (q, a) in enumerate(MATH_QUESTIONS):
        print(f"\n[{i+1}] {q}")
        print(f"    Answer: {a}")

    # Test the scoring function
    print("\n\nScoring tests:")
    print(f"  Exact match:     {calculate_score(4302459, 4302459):.4f}")
    print(f"  Missing digit:   {calculate_score(4302459, 430245):.4f}")
    print(f"  One digit off:   {calculate_score(123456789, 123356789):.4f}")
    print(f"  Way off:         {calculate_score(4302459, 9999999):.4f}")
    print(f"  Zero vs nonzero: {calculate_score(4302459, 0):.4f}")
