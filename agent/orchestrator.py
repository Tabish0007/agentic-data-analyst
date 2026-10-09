import json
import re

from config import CFG
from . import memory
from . import tools
from . import verifier
from .llm import chat
from .prompts import SYSTEM, user_prompt, VERIFIER_REJECTION
from .router import Router


class AgentResult:
    def __init__(self):
        self.answer = None
        self.ok = False
        self.steps = 0
        self.trace: list = []
        self.prompt_tokens = 0
        self.completion_tokens = 0


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text[:200]}")
    return json.loads(m.group(0), strict=False)


def run_agent(question: str, router: Router) -> AgentResult:
    res = AgentResult()
    step_model = (CFG.small_model if router.classify(question) == "simple"
                  else CFG.big_model)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(question, memory.retrieve(question))},
    ]
    for step in range(1, CFG.max_steps + 1):
        res.steps = step
        content, (pt, ct) = chat(messages, model=step_model)
        res.prompt_tokens += pt
        res.completion_tokens += ct
        messages.append({"role": "assistant", "content": content})
        try:
            action = _extract_json(content)
        except (ValueError, json.JSONDecodeError) as e:
            messages.append({"role": "user",
                             "content": f"Malformed JSON ({e}). Reply with valid JSON only."})
            continue

        res.trace.append(f"[step {step}] {action.get('thought', '')}")

        if action.get("final_answer") and not action.get("tool"):
            answer = str(action["final_answer"])
            ok, reason = verifier.check(question, answer, "\n".join(res.trace))
            if ok:
                res.answer = answer
                res.ok = True
                res.trace.append(f"FINAL: {answer}")
                return res
            res.trace.append(f"REJECTED: {reason}")
            print(f"[rejected] {reason}")
            messages.append({"role": "user",
                             "content": VERIFIER_REJECTION.format(reason=reason)})
            step_model = CFG.big_model
            continue

        tool = action.get("tool")
        try:
            if tool == "list_files":
                obs = tools.list_files(CFG.data_dir)
            elif tool == "python":
                obs = tools.run_python(str(action.get("args", {}).get("code", "")), CFG.data_dir)
            else:
                obs = f"ERROR: unknown tool '{tool}'. Use list_files or python."
        except Exception as e:
            obs = f"ERROR: tool call failed: {e}"
        res.trace.append(f"[{tool}]\n{obs[:1500]}")
        messages.append({"role": "user", "content": f"Observation:\n{obs[:4000]}"})
    return res


def learn(res: AgentResult, question: str) -> None:
    if res.ok:
        return
    last_err = next((l for l in reversed(res.trace) if "ERROR" in l),
                    "gave up before a verified answer")
    memory.remember(question, last_err,
                    "check the data dictionary and ensure code runs without error before answering")
