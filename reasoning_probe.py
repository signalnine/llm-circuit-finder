#!/usr/bin/env python3
"""
Reasoning Probe for RYS Layer Duplication Sweep

BBH-style questions: causal judgement, date understanding, logical
deduction, navigation, plus basic math word problems.
All answers are short (Yes/No, a letter, or a number).
"""

import re


# ─── Causal Judgement (Yes/No) ─────────────────────────────────────

CAUSAL_QUESTIONS = [
    {
        "prompt": (
            "How would a typical person answer each of the following questions about causation?\n"
            "Frank T., had an ongoing dispute with his neighbor over a stretch of land and one day "
            "decided to shoot his neighbor in the body. Frank T. had no experience with guns, his "
            "hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet "
            "bounced off a large boulder several feet away and hit the neighbor's body, causing "
            "significant injury. Did Frank T. intentionally shoot his neighbor in the body?\n"
            "Options:\n- Yes\n- No\n\n"
            "Think step by step, then answer Yes or No."
        ),
        "answer": "no",
    },
    {
        "prompt": (
            "How would a typical person answer each of the following questions about causation?\n"
            "Suzy and Billy are working on a project that is very important for our nation's security. "
            "The boss tells them both: \"Be sure that you are here at exactly 9 am. It is absolutely "
            "essential that you arrive at that time.\" Both Billy and Suzy arrive at 9 am. As it happens, "
            "there was a motion detector installed in the room where they arrived. The motion detector "
            "was set up to be triggered if at least one person appeared in the room at the same time. "
            "So the motion detector went off. Did Billy cause the motion detector to go off?\n"
            "Options:\n- Yes\n- No\n\n"
            "Think step by step, then answer Yes or No."
        ),
        "answer": "yes",
    },
    {
        "prompt": (
            "How would a typical person answer each of the following questions about causation?\n"
            "George and his sister Lena reunite at their parents' house for Thanksgiving. Whereas "
            "George just got into medical school, Lena is unhappy in her marriage and recently lost "
            "her job. Over the course of the day, George and Lena get into a number of heated arguments. "
            "Later in the afternoon they play a game of darts. They split the first two games, and the "
            "third game is close until the end. Who will win comes down to George's last shot. If he "
            "hits a high point region, he wins; if he hits a low point region, Lena wins. George thinks "
            "of the difficult time Lena is having, and he really wants to let her win. He aims the dart "
            "at the low point region. He sets up his shot and the dart lands in the low point region. "
            "After his shot, Lena wins the game and is very happy. Did George hit the low point region "
            "intentionally?\n"
            "Options:\n- Yes\n- No\n\n"
            "Think step by step, then answer Yes or No."
        ),
        "answer": "yes",
    },
]

# ─── Date Understanding (multiple choice letter) ──────────────────

DATE_QUESTIONS = [
    {
        "prompt": (
            "Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?\n"
            "Options:\n"
            "(A) 12/14/2026\n(B) 12/14/1950\n(C) 12/14/2007\n"
            "(D) 12/14/1937\n(E) 07/14/1938\n(F) 12/14/1988\n\n"
            "Think step by step, then give your answer as a single letter."
        ),
        "answer": "d",
    },
    {
        "prompt": (
            "Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?\n"
            "Options:\n"
            "(A) 09/04/2018\n(B) 11/11/2018\n(C) 08/25/2018\n"
            "(D) 11/02/2018\n(E) 11/04/2018\n\n"
            "Think step by step, then give your answer as a single letter."
        ),
        "answer": "b",
    },
    {
        "prompt": (
            "Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. "
            "What is the date tomorrow in MM/DD/YYYY?\n"
            "Options:\n"
            "(A) 01/11/1961\n(B) 01/03/1963\n(C) 01/18/1961\n"
            "(D) 10/14/1960\n(E) 01/03/1982\n(F) 12/03/1960\n\n"
            "Think step by step, then give your answer as a single letter."
        ),
        "answer": "b",
    },
]

# ─── Logical Deduction (multiple choice letter) ───────────────────

LOGIC_QUESTIONS = [
    {
        "prompt": (
            "The following paragraphs each describe a set of three objects arranged in a fixed order. "
            "The statements are logically consistent within each paragraph.\n"
            "In a golf tournament, there were three golfers: Amy, Eli, and Eve. "
            "Eve finished above Amy. Eli finished below Amy.\n"
            "Options:\n"
            "(A) Amy finished last\n(B) Eli finished last\n(C) Eve finished last\n\n"
            "Think step by step, then give your answer as a single letter."
        ),
        "answer": "b",
    },
    {
        "prompt": (
            "The following paragraphs each describe a set of three objects arranged in a fixed order. "
            "The statements are logically consistent within each paragraph.\n"
            "On a shelf, there are three books: a white book, a green book, and an orange book. "
            "The green book is to the right of the white book. The orange book is the rightmost.\n"
            "Options:\n"
            "(A) The white book is the leftmost\n(B) The green book is the leftmost\n"
            "(C) The orange book is the leftmost\n\n"
            "Think step by step, then give your answer as a single letter."
        ),
        "answer": "a",
    },
    {
        "prompt": (
            "The following paragraphs each describe a set of three objects arranged in a fixed order. "
            "The statements are logically consistent within each paragraph.\n"
            "On a shelf, there are three books: a red book, a gray book, and a white book. "
            "The white book is to the left of the gray book. The red book is the second from the left.\n"
            "Options:\n"
            "(A) The red book is the leftmost\n(B) The gray book is the leftmost\n"
            "(C) The white book is the leftmost\n\n"
            "Think step by step, then give your answer as a single letter."
        ),
        "answer": "c",
    },
]

# ─── Navigation (Yes/No) ──────────────────────────────────────────

NAV_QUESTIONS = [
    {
        "prompt": (
            "If you follow these instructions, do you return to the starting point? "
            "Turn left. Turn around. Turn left. Take 7 steps. Take 2 steps. Take 4 steps. Take 8 steps.\n"
            "Options:\n- Yes\n- No\n\n"
            "Think step by step, then answer Yes or No."
        ),
        "answer": "no",
    },
    {
        "prompt": (
            "If you follow these instructions, do you return to the starting point? "
            "Turn around. Take 1 step. Take 6 steps. Turn around. Take 6 steps. Take 9 steps. Take 1 step.\n"
            "Options:\n- Yes\n- No\n\n"
            "Think step by step, then answer Yes or No."
        ),
        "answer": "no",
    },
    {
        "prompt": (
            "If you follow these instructions, do you return to the starting point? "
            "Always face forward. Take 2 steps right. Take 9 steps left. Take 7 steps right.\n"
            "Options:\n- Yes\n- No\n\n"
            "Think step by step, then answer Yes or No."
        ),
        "answer": "yes",
    },
]

# ─── GSM8K Word Problems (number answer) ──────────────────────────

GSM_QUESTIONS = [
    {
        "prompt": (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
            "After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n\n"
            "Solve step by step. End with 'The answer is [NUMBER]'."
        ),
        "answer": "6",
    },
    {
        "prompt": (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n\n"
            "Solve step by step. End with 'The answer is [NUMBER]'."
        ),
        "answer": "5",
    },
    {
        "prompt": (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n\n"
            "Solve step by step. End with 'The answer is [NUMBER]'."
        ),
        "answer": "39",
    },
    {
        "prompt": (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
            "How many lollipops did Jason give to Denny?\n\n"
            "Solve step by step. End with 'The answer is [NUMBER]'."
        ),
        "answer": "8",
    },
    {
        "prompt": (
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
            "How many toys does he have now?\n\n"
            "Solve step by step. End with 'The answer is [NUMBER]'."
        ),
        "answer": "9",
    },
]

# ─── All questions with types ─────────────────────────────────────

ALL_REASONING = []
for q in CAUSAL_QUESTIONS:
    ALL_REASONING.append({**q, "type": "causal"})
for q in DATE_QUESTIONS:
    ALL_REASONING.append({**q, "type": "date"})
for q in LOGIC_QUESTIONS:
    ALL_REASONING.append({**q, "type": "logic"})
for q in NAV_QUESTIONS:
    ALL_REASONING.append({**q, "type": "nav"})
for q in GSM_QUESTIONS:
    ALL_REASONING.append({**q, "type": "gsm"})

# Alias for sweep.py import
REASONING_QUESTIONS = ALL_REASONING


def extract_final_answer(response: str) -> str:
    """Extract the final answer from a CoT response."""
    match = re.search(r'the answer is\s+(.+?)[\.\!\n\r]', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'\b(yes|no)\b', response.split('\n')[-1], re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'\(?([A-F])\)?', response.split('\n')[-1])
    if match:
        return match.group(1).strip()
    return response.strip().split('\n')[-1].strip()


def score_reasoning_response(question: dict, response: str) -> float:
    """Score a reasoning question. Returns 0 or 1."""
    if response is None:
        return 0.0

    final = extract_final_answer(response).lower().strip()
    correct = question["answer"].lower().strip()

    final = re.sub(r'[^a-z0-9]', '', final)
    correct = re.sub(r'[^a-z0-9]', '', correct)

    if final == correct:
        return 1.0

    if correct in ("yes", "no"):
        last_line = response.strip().split('\n')[-1].lower()
        if correct in last_line and (correct == "yes") != ("no" in last_line):
            return 1.0

    return 0.0


if __name__ == "__main__":
    print(f"Reasoning Probe: {len(ALL_REASONING)} questions")
    print(f"  Causal:  {len(CAUSAL_QUESTIONS)}")
    print(f"  Date:    {len(DATE_QUESTIONS)}")
    print(f"  Logic:   {len(LOGIC_QUESTIONS)}")
    print(f"  Nav:     {len(NAV_QUESTIONS)}")
    print(f"  GSM:     {len(GSM_QUESTIONS)}")

    print("\nSample questions:")
    for q in ALL_REASONING[:2]:
        print(f"\n  [{q['type']}] answer={q['answer']}")
        print(f"  {q['prompt'][:80]}...")
