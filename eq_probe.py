#!/usr/bin/env python3
"""
EQ Probe for RYS Layer Duplication Sweep

Simplified EQ-Bench-style scenarios. Each presents a social situation
and asks the model to rate the intensity of 4 emotions (0-10).
Scored by comparing to reference answers using EQ-Bench v2 scoring.

The reference answers were crafted to represent reasonable human consensus.
What matters for the sweep is relative performance between configs,
not absolute calibration.
"""

import json
import re
import math


# Each scenario: situation text, 4 emotions to rate, reference scores (0-10)
EQ_SCENARIOS = [
    {
        "id": "eq01",
        "situation": (
            "David just found out his closest friend has been spreading rumors about him "
            "at work for months. When confronted, the friend laughed it off and said "
            "\"everyone does it, don't be so sensitive.\" David has been quiet all evening."
        ),
        "emotions": ["Anger", "Betrayal", "Sadness", "Confusion"],
        "reference": [7, 9, 6, 4]
    },
    {
        "id": "eq02",
        "situation": (
            "Maria spent six months preparing a presentation for the board. During the "
            "meeting, her manager took credit for all her ideas and presented them as his "
            "own. Afterward, he thanked her for 'helping out with the slides.'"
        ),
        "emotions": ["Frustration", "Resentment", "Humiliation", "Determination"],
        "reference": [8, 8, 5, 6]
    },
    {
        "id": "eq03",
        "situation": (
            "After 20 years of marriage, Tom's wife tells him she wants a divorce. "
            "She says she still cares about him but hasn't been happy for years. "
            "Tom had no idea anything was wrong."
        ),
        "emotions": ["Shock", "Grief", "Self-doubt", "Denial"],
        "reference": [9, 7, 6, 7]
    },
    {
        "id": "eq04",
        "situation": (
            "A young doctor must tell parents that their child's surgery was successful "
            "but there will be permanent limitations. The parents are already celebrating "
            "in the waiting room."
        ),
        "emotions": ["Relief", "Anxiety", "Empathy", "Professional_detachment"],
        "reference": [4, 7, 8, 5]
    },
    {
        "id": "eq05",
        "situation": (
            "Chen receives a prestigious award at a ceremony. As he walks to the stage, "
            "he sees his estranged father in the audience - the man who abandoned the "
            "family when Chen was twelve."
        ),
        "emotions": ["Pride", "Anger", "Longing", "Anxiety"],
        "reference": [7, 5, 6, 6]
    },
    {
        "id": "eq06",
        "situation": (
            "A retired teacher learns that a former student, who she failed years ago "
            "and who dropped out, has become extremely successful. The student publicly "
            "credits 'proving my teacher wrong' as their motivation."
        ),
        "emotions": ["Guilt", "Pride", "Defensiveness", "Amusement"],
        "reference": [5, 4, 6, 3]
    },
    {
        "id": "eq07",
        "situation": (
            "Sophie finds out she's been accepted to her dream university on the same "
            "day her best friend receives a rejection from the same school. Her friend "
            "calls to congratulate her, voice cracking."
        ),
        "emotions": ["Joy", "Guilt", "Empathy", "Awkwardness"],
        "reference": [7, 6, 8, 7]
    },
    {
        "id": "eq08",
        "situation": (
            "A firefighter rescues a child from a burning building. Weeks later, he "
            "wakes up screaming from nightmares about the ones he couldn't save in "
            "a previous fire. His partner asks if he's okay."
        ),
        "emotions": ["Satisfaction", "Trauma", "Vulnerability", "Shame"],
        "reference": [3, 8, 7, 5]
    },
    {
        "id": "eq09",
        "situation": (
            "An elderly woman's family surprises her with a birthday party. She smiles "
            "and thanks everyone, but keeps glancing at an empty chair - where her "
            "late husband always sat."
        ),
        "emotions": ["Gratitude", "Grief", "Loneliness", "Warmth"],
        "reference": [7, 7, 6, 6]
    },
    {
        "id": "eq10",
        "situation": (
            "A manager must lay off a team member who is also a close friend and a "
            "single parent. The company requires it due to budget cuts. HR is waiting "
            "for the paperwork."
        ),
        "emotions": ["Guilt", "Dread", "Helplessness", "Resentment"],
        "reference": [8, 8, 7, 5]
    },
    {
        "id": "eq11",
        "situation": (
            "James finds old love letters in the attic from his wife, written to "
            "someone else before they met. The letters are passionate and describe "
            "a depth of feeling he's not sure she's ever expressed toward him."
        ),
        "emotions": ["Jealousy", "Insecurity", "Curiosity", "Sadness"],
        "reference": [6, 7, 5, 4]
    },
    {
        "id": "eq12",
        "situation": (
            "A teenager confesses to her mother that she's been self-harming. The mother, "
            "who is a psychologist, realizes she completely missed the signs despite her "
            "professional training."
        ),
        "emotions": ["Fear", "Guilt", "Love", "Self_criticism"],
        "reference": [8, 8, 9, 7]
    },
    {
        "id": "eq13",
        "situation": (
            "A war veteran returns home after two years to find his dog waiting on the "
            "porch, much older and thinner. The dog recognizes him immediately and "
            "limps over, tail wagging."
        ),
        "emotions": ["Joy", "Guilt", "Tenderness", "Sorrow"],
        "reference": [8, 5, 9, 4]
    },
    {
        "id": "eq14",
        "situation": (
            "During a job interview, the candidate realizes the interviewer is someone "
            "she bullied badly in high school. The interviewer clearly recognizes her "
            "but proceeds professionally."
        ),
        "emotions": ["Shame", "Anxiety", "Admiration", "Regret"],
        "reference": [7, 8, 4, 7]
    },
    {
        "id": "eq15",
        "situation": (
            "A father watches his daughter's wedding, knowing he has a terminal diagnosis "
            "he hasn't shared with the family. He chose to wait until after the wedding "
            "to tell them."
        ),
        "emotions": ["Joy", "Grief", "Protectiveness", "Isolation"],
        "reference": [6, 8, 8, 7]
    },
    {
        "id": "eq16",
        "situation": (
            "Two siblings meet for the first time as adults after being separated in "
            "foster care as children. They look alike but have lived completely different "
            "lives. One is wealthy, the other struggles financially."
        ),
        "emotions": ["Wonder", "Resentment", "Hope", "Grief"],
        "reference": [7, 3, 7, 6]
    },
]


def build_eq_prompt(scenario: dict) -> str:
    """Build the prompt for a single EQ scenario."""
    emotions_str = ", ".join(scenario["emotions"])
    return (
        f"Read the following situation and rate the emotional intensity that "
        f"the main character is likely feeling for each of the listed emotions. "
        f"Rate each emotion from 0 (not feeling it at all) to 10 (extremely intense).\n\n"
        f"Situation: {scenario['situation']}\n\n"
        f"Rate these emotions: {emotions_str}\n\n"
        f"Respond ONLY with the four numbers separated by commas, in the same order. "
        f"Example format: 5, 3, 8, 2\n"
        f"Do not include any other text."
    )


def parse_eq_response(response: str, n_emotions: int = 4) -> list[float] | None:
    """Extract emotion ratings from model response."""
    # Try to find comma-separated numbers
    numbers = re.findall(r'(\d+(?:\.\d+)?)', response)

    if len(numbers) < n_emotions:
        return None

    try:
        # Take the first n_emotions numbers found
        ratings = [float(numbers[i]) for i in range(n_emotions)]
        # Clamp to 0-10
        ratings = [max(0.0, min(10.0, r)) for r in ratings]
        return ratings
    except (ValueError, IndexError):
        return None


def score_eq_response(reference: list[int], predicted: list[float]) -> float:
    """
    EQ-Bench v2 style scoring.
    Differences 1-4 from reference are scaled down on a curve.
    Differences 5-10 count 1:1.
    Score 0 = random, 100 = perfect match.
    """
    if predicted is None or len(predicted) != len(reference):
        return 0.0

    total_penalty = 0.0
    max_possible_penalty = 10.0 * len(reference)  # worst case: all off by 10

    for ref, pred in zip(reference, predicted):
        diff = abs(ref - pred)
        # Scale down small differences (EQ-Bench v2 approach)
        if diff <= 4:
            # Quadratic scaling: diff^2 / 4 so diff=4 -> penalty=4
            penalty = (diff ** 2) / 4.0
        else:
            # Linear for larger diffs, continuous at diff=4 (penalty=4)
            penalty = diff
        total_penalty += penalty

    # Normalize: 0 penalty = score 100, max penalty = score ~0
    score = max(0.0, 100.0 * (1.0 - total_penalty / max_possible_penalty))
    return score


# Convenience
EQ_PROMPTS = [(s, build_eq_prompt(s)) for s in EQ_SCENARIOS]


if __name__ == "__main__":
    print(f"EQ Probe: {len(EQ_SCENARIOS)} scenarios")
    print("=" * 60)

    for i, scenario in enumerate(EQ_SCENARIOS):
        print(f"\n[{scenario['id']}] Emotions: {scenario['emotions']}")
        print(f"    Reference: {scenario['reference']}")
        prompt = build_eq_prompt(scenario)
        print(f"    Prompt length: {len(prompt)} chars")

    # Test scoring
    print("\n\nScoring tests:")
    print(f"  Perfect match:    {score_eq_response([7, 9, 6, 4], [7, 9, 6, 4]):.1f}")
    print(f"  All off by 1:     {score_eq_response([7, 9, 6, 4], [8, 8, 7, 5]):.1f}")
    print(f"  All off by 3:     {score_eq_response([7, 9, 6, 4], [4, 6, 3, 1]):.1f}")
    print(f"  All off by 5:     {score_eq_response([7, 9, 6, 4], [2, 4, 1, 0]):.1f}")
    print(f"  Worst case:       {score_eq_response([7, 9, 6, 4], [0, 0, 0, 10]):.1f}")
    print(f"  Unparseable:      {score_eq_response([7, 9, 6, 4], None):.1f}")
