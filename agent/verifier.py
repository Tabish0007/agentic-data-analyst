"""Grounding-first numerical verifier.

Hard invariant: every number the agent reports that the question did NOT
supply must appear in the observation of the LAST code run.

Order of authority:
  1. grounding gate   (deterministic, cannot be overruled)
  2. verifier model    (may veto, may not un-reject)
"""
import json
import re

from config import CFG
from .llm import chat
from metrics import extract_numbers

VERIFIER_SYSTEM = (
    "You are a strict numerical auditor. Given a question, a candidate answer, and the "
    "code/execution trace that produced it, decide if the answer is correct. "
    "Respond with ONLY JSON: {\"verdict\": true|false, \"reason\": \"...\", \"confidence\": 0.0-1.0}"
)

_MARK = "[python]"
# Cut the observation at the next trace marker so the model's own narration
# can never be mistaken for interpreter output.
_MARKERS = re.compile(r"^\s*\[(?:step|observation|assistant|final|verifier|python|error)\b",
                      re.M | re.I)
_TOL = 0.01

_GIVE_UP = re.compile(r"(i don'?t know|cannot (be )?(determined|computed|answered)|could not|no data)", re.I)
_NEEDS_NUM = re.compile(
    r"(how much|how many|what (is|was) the (total|average|median|percentage|percent|ratio|revenue))",
    re.I)


def _observation(trace: str) -> str:
    """Exactly what the interpreter printed in the LAST code run."""
    i = trace.rfind(_MARK)
    if i == -1:
        return ""
    rest = trace[i + len(_MARK):]
    stop = _MARKERS.search(rest)
    if stop:
        rest = rest[:stop.start()]
    return rest.strip()


def _close(a: float, b: float, tol: float = _TOL) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def _claimed_numbers(question: str, answer: str):
    """Answer numbers the question did not already supply (years, filters, N, ...)."""
    qnums = extract_numbers(question)
    return [n for n in extract_numbers(answer)
            if not any(_close(n, q) for q in qnums)]


def _heuristic(question: str, answer: str, trace: str) -> tuple:
    a = (answer or "").strip()
    if not a or _GIVE_UP.search(a):
        return False, "answer missing or gives up"

    obs = _observation(trace)
    if obs.startswith("ERROR"):
        return False, "last python run errored"

    claims = _claimed_numbers(question, a)
    if claims:
        if not obs or obs.startswith("(no output)"):
            return False, "last code run printed nothing, but the answer asserts a number"
        snums = extract_numbers(obs)
        ungrounded = [n for n in claims if not any(_close(n, s) for s in snums)]
        if ungrounded:
            return False, (f"ungrounded number(s) {ungrounded!r} do not appear in the last "
                           "code output. Re-run the computation WITH print() and report the "
                           "printed value verbatim.")
        reason = f"grounding pass ({len(claims)} claimed number(s) verified against stdout)"
    else:
        reason = "grounding pass (no numbers claimed)"

    if _NEEDS_NUM.search(question) and not re.search(r"\d", a):
        return False, "question expects a number but the answer has none"
    return True, reason


def check(question: str, answer: str, trace: str) -> tuple:
    # 1. deterministic gate first — a model must not be able to un-reject this
    ok, reason = _heuristic(question, answer, trace)
    if not ok:
        return False, reason

    # 2. optional model audit; veto-only
    if CFG.verifier_base_url:
        prompt = f"Question: {question}\nCandidate answer: {answer}\nTrace:\n{trace[-4000:]}"
        try:
            content, _ = chat(
                [{"role": "system", "content": VERIFIER_SYSTEM},
                 {"role": "user", "content": prompt}],
                model=CFG.verifier_model or "verifier",
                base_url=CFG.verifier_base_url, max_tokens=300)
            m = re.search(r"\{.*\}", content, re.S)
            data = json.loads(m.group(0) if m else content)
            if not bool(data.get("verdict")):
                return False, f"model veto: {data.get('reason', '')} ({reason})"
            return True, f"{reason}; model agreed"
        except Exception as e:
            print(f"[verifier] model check failed ({e}); heuristic verdict stands")

    return True, reason