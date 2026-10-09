import re


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in re.findall(r"-?\d[\d,]*\.?\d*", text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", (s or "").lower()).strip()


def score(question: str, answer: str, gold, tol: float = 0.005) -> tuple:
    """gold is a number (relative tolerance) or a string (token / substring match)."""
    if isinstance(gold, (int, float)):
        cands = extract_numbers(answer)
        if not cands:
            return False, "no number in answer"
        best = min(abs(c - gold) / max(abs(gold), 1e-9) for c in cands)
        return best <= tol, f"best_rel_err={best:.4f}"

    a, g = _norm(answer), _norm(str(gold))
    if not g:
        return False, "empty gold"
    if len(g) <= 3 and " " not in g:
        return bool(re.search(rf"\b{re.escape(g)}\b", a)), "text-token"
    return g in a, "text-contains"