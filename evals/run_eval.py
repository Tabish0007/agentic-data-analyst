import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import run_agent, learn
from agent.router import Router
from metrics import score

TASKS = "evals/tasks/tasks.jsonl"
HISTORY = "results/history"

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(TASKS) if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    router = Router()
    rows, t0 = [], time.time()
    for i, t in enumerate(tasks):
        res = run_agent(t["question"], router)
        passed, detail = score(t["question"], res.answer or "", t["answer"])
        rows.append({"id": t["id"], "passed": passed, "detail": detail,
                     "answer": res.answer, "gold": t["answer"], "steps": res.steps,
                     "prompt_tokens": res.prompt_tokens,
                     "completion_tokens": res.completion_tokens})
        print(f"[{i + 1}/{len(tasks)}] {t['id']} {'PASS' if passed else 'FAIL'} ({detail})")

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "big_model": os.getenv("AGENT_BIG_MODEL", "gpt-4o"),
        "small_model": os.getenv("AGENT_SMALL_MODEL", "gpt-4o-mini"),
        "verifier": "fine-tuned" if os.getenv("VERIFIER_BASE_URL") else "heuristic",
        "router": "ml" if os.path.exists("models/router.joblib") else "heuristic",
        "n": len(rows),
        "success": sum(r["passed"] for r in rows) / len(rows),
        "avg_steps": sum(r["steps"] for r in rows) / len(rows),
        "avg_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / len(rows),
        "wall_seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    os.makedirs(HISTORY, exist_ok=True)
    out = os.path.join(HISTORY, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsuccess={summary['success']:.1%}  avg_steps={summary['avg_steps']:.1f}  "
          f"avg_tokens={summary['avg_tokens']:.0f}  -> {out}")

if __name__ == "__main__":
    main()
