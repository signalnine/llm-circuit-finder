#!/usr/bin/env python3
"""
Code Probe for RYS Layer Duplication Sweep (Brutal Edition)

Designed to baseline around 50-60% for strong coding models.
Tasks require precise handling of tricky constraints, unusual algorithms,
and compositions that trip up pattern-matching.
"""

import re


def extract_code(response: str) -> str:
    """Extract code from a model response, handling markdown fences."""
    m = re.search(r"```(?:python|typescript|javascript|ts|js)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def run_python_tests(code: str, tests: list[dict]) -> float:
    """Execute Python code and run test cases. Returns fraction passed (0.0-1.0)."""
    passed = 0
    for test in tests:
        try:
            namespace = {}
            exec(code, namespace)
            result = eval(test["call"], namespace)
            if result == test["expected"]:
                passed += 1
        except Exception:
            pass
    return passed / len(tests) if tests else 0.0


CODE_TASKS = [
    # 1. Serialize/deserialize a binary tree — models often get the None handling wrong
    {
        "id": "serialize_tree",
        "prompt": (
            "Write Python code defining a class `TreeNode` with attributes `val`, `left`, `right` "
            "(defaulting to None), and two functions:\n"
            "- `serialize(root)` — convert a binary tree to a string\n"
            "- `deserialize(s)` — convert the string back to the identical binary tree\n"
            "They must be inverses: deserialize(serialize(tree)) must produce an identical tree. "
            "Handle None nodes correctly. Return ONLY the class and functions, no explanation."
        ),
        "tests": [
            {
                "call": "(lambda: (t := TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5))), deserialize(serialize(t)).val))()",
                "expected": 1,
            },
            {
                "call": "(lambda: (t := TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5))), deserialize(serialize(t)).right.left.val))()",
                "expected": 4,
            },
            {"call": "deserialize(serialize(None))", "expected": None},
            {
                "call": "(lambda: (t := TreeNode(1, None, TreeNode(2, None, TreeNode(3))), deserialize(serialize(t)).right.right.val))()",
                "expected": 3,
            },
            {"call": "deserialize(serialize(TreeNode(42))).val", "expected": 42},
            {
                "call": "deserialize(serialize(TreeNode(-1, TreeNode(-2), None))).left.val",
                "expected": -2,
            },
            {
                "call": "deserialize(serialize(TreeNode(0, TreeNode(0), TreeNode(0)))).right.val",
                "expected": 0,
            },
        ],
    },
    # 2. Count smaller numbers after self — merge sort with index tracking
    {
        "id": "count_smaller_after",
        "prompt": (
            "Write a Python function `count_smaller(nums)` that returns a list where each element "
            "is the count of smaller elements to the RIGHT of that element in nums. "
            "Must be O(n log n). Use merge sort with index tracking. "
            "Example: count_smaller([5,2,6,1]) => [2,1,1,0]. "
            "count_smaller([2,0,1]) => [2,0,0]. "
            "Return ONLY the function, no explanation."
        ),
        "tests": [
            {"call": "count_smaller([5,2,6,1])", "expected": [2,1,1,0]},
            {"call": "count_smaller([2,0,1])", "expected": [2,0,0]},
            {"call": "count_smaller([-1])", "expected": [0]},
            {"call": "count_smaller([-1,-1])", "expected": [0,0]},
            {"call": "count_smaller([1,2,3,4])", "expected": [0,0,0,0]},
            {"call": "count_smaller([4,3,2,1])", "expected": [3,2,1,0]},
            {"call": "count_smaller([26,78,27,100,33,67,90,23,66,5,38,7,35,23,52,22,83,51,98,69,81,32,21,1,27,81])", "expected": [10,17,10,22,12,14,17,6,13,2,8,2,6,2,8,1,11,5,13,8,9,3,2,0,0,0]},
        ],
    },
    # 4. Skyline problem — coordinate compression + sweep line
    {
        "id": "skyline",
        "prompt": (
            "Write a Python function `get_skyline(buildings)` where buildings is a list of "
            "[left, right, height] and returns a list of [x, height] key points that form "
            "the skyline silhouette. Key points are sorted by x. "
            "When multiple buildings end/start at the same x, report the correct height. "
            "Example: get_skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]) => "
            "[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]. "
            "Return ONLY the function, no explanation."
        ),
        "tests": [
            {
                "call": "get_skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]])",
                "expected": [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]],
            },
            {
                "call": "get_skyline([[0,2,3],[2,5,3]])",
                "expected": [[0,3],[5,0]],
            },
            {"call": "get_skyline([[1,2,1]])", "expected": [[1,1],[2,0]]},
            {
                "call": "get_skyline([[1,2,1],[1,2,2],[1,2,3]])",
                "expected": [[1,3],[2,0]],
            },
            {"call": "get_skyline([])", "expected": []},
        ],
    },
    # 5. Implement a calculator that handles +, -, *, /, parentheses, and unary minus
    {
        "id": "calculator",
        "prompt": (
            "Write a Python function `calculate(s)` that evaluates a math expression string. "
            "Support +, -, *, / (integer division truncating toward zero), parentheses, "
            "unary minus, and spaces. "
            "Examples: calculate('3+2*2') => 7, calculate(' 3/2 ') => 1, "
            "calculate('(1+(4+5+2)-3)+(6+8)') => 23, calculate('-(3+2)') => -5. "
            "Return ONLY the function, no explanation."
        ),
        "tests": [
            {"call": "calculate('3+2*2')", "expected": 7},
            {"call": "calculate(' 3/2 ')", "expected": 1},
            {"call": "calculate(' 3+5 / 2 ')", "expected": 5},
            {"call": "calculate('(1+(4+5+2)-3)+(6+8)')", "expected": 23},
            {"call": "calculate('-(3+2)')", "expected": -5},
            {"call": "calculate('2*(3+4)')", "expected": 14},
            {"call": "calculate('14-3/2')", "expected": 13},
            {"call": "calculate('(-7)*3')", "expected": -21},
        ],
    },
    # 6. Burst balloons — interval DP, counterintuitive formulation
    {
        "id": "burst_balloons",
        "prompt": (
            "Write a Python function `max_coins(nums)` that returns the maximum coins you can "
            "collect by bursting all balloons. If you burst balloon i, you get "
            "nums[i-1] * nums[i] * nums[i+1] coins (neighbors multiply). "
            "After bursting, the neighbors change. Treat boundaries as virtual balloons with value 1. "
            "Example: max_coins([3,1,5,8]) => 167. "
            "Return ONLY the function, no explanation."
        ),
        "tests": [
            {"call": "max_coins([3,1,5,8])", "expected": 167},
            {"call": "max_coins([1,5])", "expected": 10},
            {"call": "max_coins([1])", "expected": 1},
            {"call": "max_coins([7,9,8,0,7,1,3,5,5,2])", "expected": 1654},
            {"call": "max_coins([35,16,83])", "expected": 51940},
            {"call": "max_coins([])", "expected": 0},
        ],
    },
    # 7. Minimum number of refueling stops — greedy with heap
    {
        "id": "min_refuel_stops",
        "prompt": (
            "Write a Python function `min_refuel(target, start_fuel, stations)` where target is "
            "the distance to reach, start_fuel is initial fuel, and stations is a list of "
            "[position, fuel] sorted by position. Return the minimum number of refueling stops "
            "to reach the target, or -1 if impossible. You can only refuel at stations. "
            "Example: min_refuel(100, 10, [[10,60],[20,30],[30,30],[60,40]]) => 2. "
            "Return ONLY the function, no explanation."
        ),
        "tests": [
            {"call": "min_refuel(100, 10, [[10,60],[20,30],[30,30],[60,40]])", "expected": 2},
            {"call": "min_refuel(1, 1, [])", "expected": 0},
            {"call": "min_refuel(100, 1, [[10,100]])", "expected": -1},
            {"call": "min_refuel(100, 50, [[25,25],[50,50]])", "expected": 1},
            {"call": "min_refuel(100, 100, [])", "expected": 0},
            {"call": "min_refuel(100, 10, [])", "expected": -1},
        ],
    },
]


def score_code_response(task: dict, response: str) -> float:
    """Score a code response by extracting code and running tests."""
    if response is None:
        return 0.0
    code = extract_code(response)
    if not code:
        return 0.0
    return run_python_tests(code, task["tests"])


if __name__ == "__main__":
    print(f"Code Probe (Brutal): {len(CODE_TASKS)} tasks, "
          f"{sum(len(t['tests']) for t in CODE_TASKS)} total test cases")
    print("=" * 60)
    for task in CODE_TASKS:
        print(f"\n[{task['id']}] {len(task['tests'])} tests")
        print(f"  {task['prompt'][:80]}...")
