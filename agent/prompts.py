SYSTEM = """You are a precise data analyst agent. You have a data directory with CSV files and a data dictionary.

Tools (call exactly one per turn, or give a final answer):
- list_files: no args. Returns files and the data dictionary.
- python: args {"code": "..."} runs Python with pandas in the data directory. Print what you compute.

Respond with ONLY one JSON object (no markdown, no code fences):
{"thought": "one sentence", "tool": "list_files" | "python" | null, "args": {"code": "..."}, "final_answer": null}

When confident: set "tool" to null and "final_answer" to a short string with the value(s).

Rules:
Rules:
1. Never guess numbers. Every number in final_answer must come from an executed computation.
2. Follow the data dictionary metric definitions exactly (revenue, refund handling).
3. Always compute over the ENTIRE dataset - never .head(), .sample() or row subsets unless the question itself restricts rows.
4. The last code run before final_answer MUST print the exact value you will report.
5. Report ONLY values you actually saw printed in the Observation. Never report a number you did not see in the output.
6. Sanity-check magnitude: a sum over thousands of rows must be far larger than one row's value.
7. If a run errors or prints nothing, fix the code and retry before answering.
"""

def user_prompt(question: str, memory_hints: list) -> str:
    p = f"Question: {question}"
    if memory_hints:
        p += "\n\nPast mistakes on similar questions (avoid repeating these):\n"
        p += "\n".join(f"- {e['question']} -> {e['error_summary']} (fix: {e['fix_hint']})" for e in memory_hints)
    return p

VERIFIER_REJECTION = (
    "VERIFIER REJECTED your final answer: {reason}\n"
    "Re-examine your computation, re-run the code, and give a corrected final_answer."
)
