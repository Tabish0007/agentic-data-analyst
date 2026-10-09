import json
import os
import sys

import joblib
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import question_vector

def main() -> None:
    tasks = [json.loads(l) for l in open("evals/tasks/tasks.jsonl") if l.strip()]
    X = [question_vector(t["question"]) for t in tasks]
    y = [1 if t.get("complexity") == "complex" else 0 for t in tasks]
    m = LogisticRegression(max_iter=1000)
    m.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(m, "models/router.joblib")
    acc = float((m.predict(X) == y).mean())
    print(f"router saved -> models/router.joblib (train acc {acc:.2f} on {len(X)} tasks)")

if __name__ == "__main__":
    main()
