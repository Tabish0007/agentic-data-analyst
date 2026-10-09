import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import run_agent
from agent.router import Router
from metrics import score


def mk(t, answer, trace, correct):
    return {"question": t["question"], "answer": str(answer),
            "trace": trace[-6000:], "verdict": bool(correct)}

def mutate(gold):
    if isinstance(gold, (int, float)):
        return round(gold * random.uniform(0.8, 1.2), 2)
    return "none"

def main(epochs: int = 3, out: str = "data_verifier/train.jsonl") -> None:
    tasks = [json.loads(l) for l in open("evals/tasks/tasks.jsonl") if l.strip()]
    router = Router()
    examples = []
    for _ in range(epochs):
        for t in tasks:
            res = run_agent(t["question"], router)
            passed, _ = score(t["question"], res.answer or "", t["answer"])
            examples.append(mk(t, res.answer or "(no answer)", "\n".join(res.trace), passed))
            examples.append(mk(t, mutate(t["answer"]), "\n".join(res.trace), False))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {len(examples)} examples -> {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="data_verifier/train.jsonl")
    a = ap.parse_args()
    main(a.epochs, a.out)
