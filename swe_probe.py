#!/usr/bin/env python3
"""
SWE Agentic Probe for RYS Layer Surgery

Tests model ability to reason about software engineering tool use:
- Diff reading and application
- Git operations
- Error diagnosis from build/test output
- Multi-file reasoning
- Command generation for file/search operations

Scored deterministically by checking extracted answers against known correct ones.
"""

import re
import json


def strip_thinking(response: str) -> str:
    """Strip <think>...</think> blocks from thinking models (e.g., Qwen3.5)."""
    if response is None:
        return ""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def extract_answer(response: str, marker: str = "ANSWER:") -> str:
    """Extract the answer after a marker."""
    if response is None:
        return ""
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith(marker.upper()):
            return line[len(marker):].strip()
        # Also try without colon
        if line.upper().startswith(marker.rstrip(":").upper()):
            rest = line[len(marker.rstrip(":")):].strip()
            if rest.startswith(":"):
                rest = rest[1:].strip()
            return rest
    # Fallback: try to find it in code blocks
    m = re.search(r"```\s*\n?(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def normalize(s: str) -> str:
    """Normalize whitespace and case for comparison."""
    return " ".join(s.lower().split())


def check_contains_all(response: str, required: list[str]) -> float:
    """Check what fraction of required strings appear in response."""
    if not response:
        return 0.0
    resp_lower = response.lower()
    found = sum(1 for r in required if r.lower() in resp_lower)
    return found / len(required)


SWE_TASKS = [
    # 1. Read a unified diff and identify what changed
    {
        "id": "diff_comprehension",
        "prompt": (
            "Read this unified diff and answer the questions below.\n\n"
            "```diff\n"
            "--- a/src/auth.ts\n"
            "+++ b/src/auth.ts\n"
            "@@ -15,7 +15,9 @@ export async function validateToken(token: string): Promise<User | null> {\n"
            "   const decoded = jwt.verify(token, SECRET_KEY);\n"
            "   const user = await db.users.findById(decoded.sub);\n"
            "-  return user;\n"
            "+  if (!user || user.deletedAt) {\n"
            "+    return null;\n"
            "+  }\n"
            "+  return user;\n"
            " }\n"
            "```\n\n"
            "Questions (answer each on its own line, prefixed with the number):\n"
            "1. What file was modified?\n"
            "2. How many lines were added?\n"
            "3. How many lines were removed?\n"
            "4. What bug or issue does this fix? (one sentence)\n"
            "5. Could this be a breaking change? (yes/no)\n"
        ),
        "check": lambda resp: (
            (1.0 if "src/auth.ts" in resp else 0.0) +
            (1.0 if any(x in resp for x in ["3 lines", "three lines", "3 added", "+3"]) else 0.0) +
            (1.0 if any(x in resp for x in ["1 line", "one line", "1 removed", "-1"]) else 0.0) +
            (1.0 if any(x in resp.lower() for x in ["deleted", "soft-delete", "soft delete"]) else 0.0) +
            (1.0 if ("5" in resp and "no" in resp.lower().split("5")[-1][:50]) else 0.0)
        ) / 5,
    },
    # 2. Git bisect scenario — requires reasoning about commit history
    {
        "id": "git_bisect_reasoning",
        "prompt": (
            "A test started failing somewhere in the last 8 commits. Here's the git log:\n\n"
            "```\n"
            "a1b2c3d (HEAD) refactor: extract validation logic\n"
            "e4f5g6h fix: handle null in user service\n"
            "i7j8k9l feat: add rate limiting middleware\n"
            "m0n1o2p chore: update dependencies\n"
            "q3r4s5t fix: correct timezone handling\n"
            "u6v7w8x feat: add caching layer\n"
            "y9z0a1b refactor: rename database columns\n"
            "c2d3e4f feat: add user preferences (LAST KNOWN GOOD)\n"
            "```\n\n"
            "You run `git bisect` and test at each midpoint. The results are:\n"
            "- Test at m0n1o2p: PASS\n"
            "- Test at e4f5g6h: FAIL\n"
            "- Test at i7j8k9l: FAIL\n\n"
            "Answer each question:\n"
            "1. Which commit introduced the bug? (give the hash)\n"
            "2. How many tests did bisect need to find it? (the 3 above)\n"
            "3. What's the minimum number of tests bisect would need for 8 commits? (log2)\n"
            "4. Based on the commit message, what likely category of change caused the bug?\n"
        ),
        "check": lambda resp: (
            # i7j8k9l is the first failing commit (m0n1o2p passes, i7j8k9l fails)
            (1.0 if "i7j8k9l" in resp else 0.0) +
            # 3 tests
            (1.0 if "3" in resp else 0.0) +
            # log2(8) = 3
            (1.0 if "3" in resp else 0.0) +
            # rate limiting middleware
            (1.0 if any(x in resp.lower() for x in ["rate limit", "middleware", "rate-limit"]) else 0.0)
        ) / 4,
    },
    # 3. Diagnose a build error
    {
        "id": "build_error_diagnosis",
        "prompt": (
            "Here's a TypeScript build error. What's wrong and how do you fix it?\n\n"
            "```\n"
            "src/services/payment.ts:42:5 - error TS2322: Type 'string | undefined' is not assignable to type 'string'.\n"
            "  Type 'undefined' is not assignable to type 'string'.\n"
            "\n"
            "42     const customerId: string = user.stripeCustomerId;\n"
            "       ~~~~~~~~~~\n"
            "\n"
            "src/services/payment.ts:58:30 - error TS2345: Argument of type 'number' is not assignable to parameter of type 'string'.\n"
            "\n"
            "58     await chargeCustomer(amount, currency);\n"
            "                            ~~~~~~\n"
            "```\n\n"
            "For each error, provide:\n"
            "1. Line number\n"
            "2. The fix (one line of corrected code)\n"
        ),
        "check": lambda resp: (
            # Error 1: line 42, needs optional check or assertion
            (1.0 if "42" in resp else 0.0) +
            (1.0 if any(x in resp for x in [
                "string | undefined",
                "?.stripeCustomerId",
                "stripeCustomerId!",
                "stripeCustomerId ?? ",
                "if (!user.stripeCustomerId",
                "as string",
            ]) else 0.0) +
            # Error 2: line 58, wrong argument order
            (1.0 if "58" in resp else 0.0) +
            (1.0 if any(x in resp for x in [
                "currency, amount",
                "amount.toString()",
                "String(amount)",
                "`${amount}`",
                "swap", "wrong order", "reversed",
            ]) else 0.0)
        ) / 4,
    },
    # 4. Multi-file merge conflict with subtle interaction
    {
        "id": "multi_file_conflict",
        "prompt": (
            "Two developers made changes to the same codebase. Dev A renamed "
            "`getUserName()` to `getDisplayName()` across the codebase. Dev B "
            "added a new function that calls `getUserName()`. After merging Dev A's "
            "changes first, Dev B's code compiles but the function call is wrong.\n\n"
            "Dev B's new code (already merged, now has a bug):\n"
            "```typescript\n"
            "// src/notifications.ts\n"
            "import { getUserName } from './user';\n"
            "\n"
            "export function sendWelcomeEmail(userId: string) {\n"
            "  const name = getUserName(userId);\n"
            "  return sendEmail(userId, `Welcome, ${name}!`);\n"
            "}\n"
            "```\n\n"
            "But `getUserName` no longer exists — it was renamed to `getDisplayName`.\n"
            "The build passes because there's a deprecated compatibility shim in user.ts:\n"
            "```typescript\n"
            "/** @deprecated Use getDisplayName instead */\n"
            "export const getUserName = getDisplayName;\n"
            "```\n\n"
            "Answer:\n"
            "1. Does this code have a runtime bug? (yes/no)\n"
            "2. What tool/command would find all remaining usages of the deprecated function?\n"
            "3. Write the corrected import and function call for notifications.ts\n"
            "4. Should the compatibility shim be removed immediately? Why or why not?\n"
        ),
        "check": lambda resp: (
            # 1. No runtime bug (the shim works), but it's a code smell
            (1.0 if any(x in resp.lower() for x in ["no runtime", "no,", "works", "shim"]) else 0.0) +
            # 2. grep for getUserName
            (1.0 if any(x in resp for x in ["grep", "rg", "search", "find"]) and "getUserName" in resp else 0.0) +
            # 3. Must have getDisplayName in the fix
            (1.0 if "getDisplayName" in resp and "import" in resp else 0.0) +
            # 4. Not immediately — need to update all callers first
            (1.0 if any(x in resp.lower() for x in ["not immediately", "after", "update all", "other callers", "breaking", "gradual"]) else 0.0)
        ) / 4,
    },
    # 5. Flaky test diagnosis — requires understanding of async/race conditions
    {
        "id": "flaky_test_diagnosis",
        "prompt": (
            "This test passes ~80% of the time and fails ~20%. Diagnose the flakiness.\n\n"
            "```typescript\n"
            "test('should process all items in queue', async () => {\n"
            "  const queue = new TaskQueue();\n"
            "  const results: string[] = [];\n"
            "\n"
            "  queue.enqueue(() => results.push('a'));\n"
            "  queue.enqueue(() => results.push('b'));\n"
            "  queue.enqueue(() => results.push('c'));\n"
            "\n"
            "  queue.processAll();  // starts processing async\n"
            "\n"
            "  expect(results).toEqual(['a', 'b', 'c']);\n"
            "});\n"
            "```\n\n"
            "Answer:\n"
            "1. Why is this test flaky? (one sentence)\n"
            "2. Write the corrected test (must await the processing)\n"
            "3. What general category of bug is this? (e.g., race condition, memory leak, etc.)\n"
        ),
        "check": lambda resp: (
            # 1. async/race - processAll is async but not awaited
            (1.0 if any(x in resp.lower() for x in ["await", "async", "not waiting", "race", "before.*complete"]) else 0.0) +
            # 2. Must have await in the fix
            (1.0 if "await" in resp and "processAll" in resp else 0.0) +
            # 3. Race condition
            (1.0 if "race" in resp.lower() else
             0.5 if any(x in resp.lower() for x in ["timing", "async", "concurren"]) else 0.0)
        ) / 3,
    },
    # 6. CI pipeline failure — requires reading YAML + error together
    {
        "id": "ci_pipeline_debug",
        "prompt": (
            "This GitHub Actions workflow is failing. The error is:\n"
            "```\n"
            "Error: Process completed with exit code 1.\n"
            "npm ERR! Missing script: \"test:ci\"\n"
            "```\n\n"
            "The workflow file:\n"
            "```yaml\n"
            "name: CI\n"
            "on: [push, pull_request]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-node@v4\n"
            "        with:\n"
            "          node-version: 20\n"
            "      - run: npm ci\n"
            "      - run: npm run test:ci\n"
            "      - run: npm run lint\n"
            "```\n\n"
            "The package.json scripts:\n"
            "```json\n"
            "{\n"
            "  \"scripts\": {\n"
            "    \"test\": \"vitest run\",\n"
            "    \"test:watch\": \"vitest\",\n"
            "    \"test:coverage\": \"vitest run --coverage\",\n"
            "    \"lint\": \"eslint src/\"\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "Answer:\n"
            "1. What's the exact problem?\n"
            "2. Give TWO different ways to fix it (one changes the workflow, one changes package.json)\n"
            "3. Which fix is better for maintainability and why?\n"
        ),
        "check": lambda resp: (
            # 1. test:ci script doesn't exist
            (1.0 if any(x in resp.lower() for x in ["test:ci", "missing script", "doesn't exist", "does not exist", "not defined"]) else 0.0) +
            # 2a. Change workflow to use "npm test" or "npm run test"
            (1.0 if any(x in resp for x in ["npm test", "npm run test\"", "npm run test\n", "run: npm test"]) else
             0.5 if "change" in resp.lower() and "workflow" in resp.lower() else 0.0) +
            # 2b. Add test:ci script to package.json
            (1.0 if any(x in resp for x in ["test:ci", "add.*script", "package.json"]) and "vitest" in resp else
             0.5 if "add" in resp.lower() and "script" in resp.lower() else 0.0) +
            # 3. Adding the script is better (single source of truth for CI config)
            (1.0 if any(x in resp.lower() for x in ["package.json", "single source", "convention", "explicit", "ci-specific"]) else 0.0)
        ) / 4,
    },
    # 7. Generate correct grep/find commands
    {
        "id": "code_search_commands",
        "prompt": (
            "Write the exact shell command for each task. One command per line, no explanation.\n\n"
            "1. Find all TypeScript files that contain 'TODO' or 'FIXME' comments\n"
            "2. Find all files larger than 1MB in the src/ directory\n"
            "3. Count the number of times 'console.log' appears across all .ts files\n"
            "4. Find all exported functions in src/utils/ that start with 'validate'\n"
            "5. List all .ts files modified in the last 24 hours\n"
        ),
        "check": lambda resp: (
            # 1: grep for TODO/FIXME in ts files
            (1.0 if re.search(r"grep.*-[rRn].*TODO.*FIXME|grep.*-[rRn].*FIXME.*TODO|grep.*-E.*(TODO|FIXME)", resp.split("\n")[0] if resp else "") else
             0.5 if re.search(r"grep.*(TODO|FIXME)", resp) else 0.0) +
            # 2: find with -size
            (1.0 if re.search(r"find\s+src/?\s.*-size\s+\+1[Mm]", resp) else 0.0) +
            # 3: grep -c or grep | wc for console.log
            (1.0 if re.search(r"grep.*-[rc].*console\.log.*\.ts|grep.*console\.log.*\.ts.*wc", resp) else
             0.5 if "console.log" in resp and ("grep" in resp or "rg" in resp) else 0.0) +
            # 4: grep for export function validate in src/utils
            (1.0 if re.search(r"grep.*export.*function\s+validate.*src/utils|grep.*validate.*src/utils", resp) else
             0.5 if "validate" in resp and "src/utils" in resp else 0.0) +
            # 5: find with -mtime or -newer
            (1.0 if re.search(r"find.*-mtime\s+(-1|0)|find.*-newer|find.*-mmin", resp) and ".ts" in resp else 0.0)
        ) / 5,
    },
    # 8. Environment-specific bug — requires reasoning about dev vs prod differences
    {
        "id": "env_specific_bug",
        "prompt": (
            "A feature works locally but fails in production with this error:\n"
            "```\n"
            "Error: connect ECONNREFUSED 127.0.0.1:6379\n"
            "    at TCPConnectWrap.afterConnect [as oncomplete] (net.js:1141:16)\n"
            "```\n\n"
            "The code:\n"
            "```typescript\n"
            "import Redis from 'ioredis';\n"
            "\n"
            "const redis = new Redis({\n"
            "  host: process.env.REDIS_HOST || '127.0.0.1',\n"
            "  port: parseInt(process.env.REDIS_PORT || '6379'),\n"
            "});\n"
            "```\n\n"
            "Production docker-compose.yml:\n"
            "```yaml\n"
            "services:\n"
            "  app:\n"
            "    build: .\n"
            "    environment:\n"
            "      - NODE_ENV=production\n"
            "      - DATABASE_URL=postgres://...\n"
            "  redis:\n"
            "    image: redis:7-alpine\n"
            "    ports:\n"
            "      - '6379:6379'\n"
            "```\n\n"
            "Answer:\n"
            "1. Why does it work locally but not in production?\n"
            "2. What's the exact fix to docker-compose.yml?\n"
            "3. What's the exact fix to the app environment variables?\n"
            "4. Is the port mapping (6379:6379) necessary for service-to-service communication? (yes/no)\n"
        ),
        "check": lambda resp: (
            # 1. REDIS_HOST not set, defaults to 127.0.0.1 which doesn't work in Docker
            (1.0 if any(x in resp.lower() for x in ["redis_host", "not set", "127.0.0.1", "localhost", "different container", "network"]) else 0.0) +
            # 2/3. Need to add REDIS_HOST=redis to environment
            (1.0 if "REDIS_HOST" in resp and "redis" in resp else 0.0) +
            (1.0 if any(x in resp for x in ["REDIS_HOST=redis", "REDIS_HOST: redis", "host: redis"]) else 0.0) +
            # 4. No — Docker services communicate over internal network, port mapping is for external access
            (1.0 if any(x in resp.lower() for x in ["no", "not necessary", "internal", "not needed"]) else 0.0)
        ) / 4,
    },
    # 9. Memory leak diagnosis — requires understanding of closures and event listeners
    {
        "id": "memory_leak_diagnosis",
        "prompt": (
            "This Express server's memory usage grows over time. Find the leak.\n\n"
            "```typescript\n"
            "const express = require('express');\n"
            "const EventEmitter = require('events');\n"
            "const app = express();\n"
            "const bus = new EventEmitter();\n"
            "\n"
            "app.get('/stream/:userId', (req, res) => {\n"
            "  const userId = req.params.userId;\n"
            "  res.setHeader('Content-Type', 'text/event-stream');\n"
            "  res.setHeader('Cache-Control', 'no-cache');\n"
            "\n"
            "  const handler = (data: any) => {\n"
            "    if (data.userId === userId) {\n"
            "      res.write(`data: ${JSON.stringify(data)}\\n\\n`);\n"
            "    }\n"
            "  };\n"
            "\n"
            "  bus.on('notification', handler);\n"
            "\n"
            "  req.on('close', () => {\n"
            "    console.log(`Client ${userId} disconnected`);\n"
            "  });\n"
            "});\n"
            "```\n\n"
            "Answer:\n"
            "1. What's the memory leak? (one sentence)\n"
            "2. Write the fix (the corrected close handler)\n"
            "3. What Node.js warning would eventually appear if this isn't fixed?\n"
        ),
        "check": lambda resp: (
            # 1. Event listener is never removed on disconnect
            (1.0 if any(x in resp.lower() for x in [
                "listener.*never.*removed", "not removed", "removelistener",
                "off(", "never.*clean", "never.*unsubscri", "handler.*not.*removed",
                "listener.*leak", "not.*unregister"
            ]) else 0.0) +
            # 2. Must include removeListener or off in the fix
            (1.0 if any(x in resp for x in [
                "removeListener", ".off(", "bus.off", "bus.removeListener"
            ]) and "handler" in resp else 0.0) +
            # 3. MaxListenersExceeded warning
            (1.0 if any(x in resp.lower() for x in [
                "maxlisteners", "max listeners", "possible memory leak",
                "too many listeners", "listener.*exceed"
            ]) else 0.0)
        ) / 3,
    },
    # 10. Generate a correct sed/awk command for a refactoring task
    {
        "id": "refactoring_command",
        "prompt": (
            "Write the exact command(s) for each refactoring task:\n\n"
            "1. Rename all occurrences of the function `getUserById` to `findUserById` "
            "across all .ts files in src/ (including imports)\n"
            "2. Add 'use strict'; as the first line of every .js file in the project\n"
            "3. Remove all console.log statements from .ts files in src/ (entire lines)\n"
        ),
        "check": lambda resp: (
            # 1: sed or find+sed to rename getUserById -> findUserById
            (1.0 if re.search(r"sed.*s[/|]getUserById[/|]findUserById[/|]", resp) or
                    re.search(r"replace.*getUserById.*findUserById", resp) else
             0.5 if "getUserById" in resp and "findUserById" in resp and ("sed" in resp or "replace" in resp) else 0.0) +
            # 2: sed to insert 'use strict' at line 1
            (1.0 if re.search(r"sed.*1i.*use strict|sed.*0a.*use strict", resp) else
             0.5 if "use strict" in resp and "sed" in resp else 0.0) +
            # 3: sed or grep -v to remove console.log lines
            (1.0 if re.search(r"sed.*/console\.log/d|grep\s+-v.*console\.log", resp) else
             0.5 if "console.log" in resp and ("sed" in resp or "grep" in resp or "remove" in resp) else 0.0)
        ) / 3,
    },
]


def score_swe_response(task: dict, response: str) -> float:
    """Score an SWE task response."""
    if response is None:
        return 0.0
    response = strip_thinking(response)
    return task["check"](response)


if __name__ == "__main__":
    print(f"SWE Agentic Probe: {len(SWE_TASKS)} tasks")
    print("=" * 60)
    for task in SWE_TASKS:
        print(f"\n[{task['id']}]")
        print(f"  {task['prompt'][:80]}...")
