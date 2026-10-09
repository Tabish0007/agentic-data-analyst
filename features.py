import re

FEATURE_KEYS = [
    "n_words", "n_clauses", "has_time_series",
    "has_comparison", "has_two_years", "has_explain", "long",
]

def question_features(q: str) -> dict:
    q = q.lower()
    return {
        "n_words": len(q.split()),
        "n_clauses": q.count(",") + q.count(" and "),
        "has_time_series": bool(re.search(r"(quarter|month|over time|trend|each (month|quarter|year))", q)),
        "has_comparison": bool(re.search(r"(compare|versus|vs|which .* had|largest|highest|best|most)", q)),
        "has_two_years": len(set(re.findall(r"20\d\d", q))) > 1,
        "has_explain": bool(re.search(r"(why|explain|reason|cause)", q)),
        "long": len(q) > 90,
    }

def question_vector(q: str) -> list:
    f = question_features(q)
    return [int(f[k]) for k in FEATURE_KEYS]
