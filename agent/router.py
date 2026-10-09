import os
import re

import joblib

from config import CFG
from features import question_vector

COMPLEX_HINTS = re.compile(
    r"(why|explain|trend|versus|vs\.?|compare|over time|by quarter|by month|growth|decline|each (month|quarter))", re.I)

def heuristic(question: str) -> str:
    return ("complex" if COMPLEX_HINTS.search(question)
            or question.count(",") >= 2 or len(question.split()) > 22 else "simple")

class Router:
    def __init__(self, path: str = CFG.router_path):
        self.model = joblib.load(path) if os.path.exists(path) else None

    def classify(self, question: str) -> str:
        if self.model is not None:
            pred = self.model.predict([question_vector(question)])[0]
            return "complex" if pred == 1 else "simple"
        return heuristic(question)
