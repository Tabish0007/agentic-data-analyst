import json
import os
import re

from config import CFG

def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", (s or "").lower()))

def retrieve(question: str, top_k: int = 3) -> list:
    if not os.path.exists(CFG.memory_path):
        return []
    q = _tokens(question)
    hits = []
    for line in open(CFG.memory_path):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ov = len(q & _tokens(e["question"])) / max(1, len(q | _tokens(e["question"])))
        if ov > 0.35:
            hits.append((ov, e))
    hits.sort(key=lambda x: -x[0])
    return [e for _, e in hits[:top_k]]

def remember(question: str, error_summary: str, fix_hint: str) -> None:
    os.makedirs(os.path.dirname(CFG.memory_path) or ".", exist_ok=True)
    with open(CFG.memory_path, "a") as f:
        f.write(json.dumps({"question": question,
                            "error_summary": error_summary[:300],
                            "fix_hint": fix_hint}) + "\n")
