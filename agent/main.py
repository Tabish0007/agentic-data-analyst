import argparse

from config import CFG
from .orchestrator import run_agent, learn
from .router import Router


def _ask(question: str, router: Router, verbose: bool = False) -> None:
    res = run_agent(question, router)
    if verbose:
        print("---- trace ----")
        for line in res.trace:
            print(line[:1200])
        print("---- end trace ----")
    print(f"\nanswer: {res.answer}")
    print(f"ok={res.ok} steps={res.steps} "
          f"tokens=({res.prompt_tokens} in / {res.completion_tokens} out)")
    learn(res, question)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic data analyst")
    ap.add_argument("question", nargs="?", help="question to answer (omit for REPL)")
    ap.add_argument("--data", default=None, help="override data dir")
    ap.add_argument("--verbose", action="store_true", help="print the full agent trace")
    args = ap.parse_args()
    if args.data:
        CFG.data_dir = args.data
    router = Router()
    if args.question:
        _ask(args.question, router, verbose=args.verbose)
        return
    print("Agentic data analyst. Type a question (Ctrl-D to quit).")
    while True:
        try:
            line = input("q> ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip():
            _ask(line.strip(), router, verbose=True)


if __name__ == "__main__":
    main()
