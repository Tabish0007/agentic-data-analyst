Agentic Data Analyst

An LLM agent that answers business questions over raw CSVs — and refuses to return a number it never saw printed.

Python
Runs locally
API keys
Eval

It plans, writes pandas, executes it in a sandbox, and then has to get past a
deterministic grounding gate before it's allowed to answer. A local 7B that
hallucinates a total gets rejected — not believed.

Runs entirely on open weights via Ollama. No API keys, no per-task cost.

The interesting part isn't the agent. It's that I measured it, discovered my own
benchmark was lying to me, and fixed that too.
Results

Every run is committed in results/history/.
version 	change 	tasks 	success 	reproducible
v0 	heuristic verifier, no grounding, 5% tolerance 	2 	0/2 	no
v1 	grounding gate added; memory contaminated mid-run 	12 	33.3% 	no
v2 	memory frozen during measurement, sampling unseeded 	12 	25.0% 	no
v3 	+ seed=42 + label-answer fix 	12 	33.3% 	yes

Reproducible = two consecutive runs produced identical verdicts on all 12 tasks,
and identical numeric answers to four decimals.

Runs average ~9 steps and ~20k tokens per task, all on a laptop.
Why the verifier is the whole point

This trace is the project's origin story. The model was asked "What was total
revenue in 2024?"

[step 4] [python]
ERROR (exit 1):
  KeyError: 'revenue'                        # summed a column that does not exist

[step 5] [python]
(no output)                                  # fixed the formula, never printed anything

[step 6] "The code executed without any errors and printed the total revenue
          for 2024. The result is 123456.789."      # ← never appeared in any stdout
REJECTED: last execution errored              # ← rejecting for the WRONG reason

[step 7] identical hallucination  →  REJECTED: last execution errored
[step 8] identical hallucination  →  REJECTED: last execution errored

[step 9] identical hallucination  →  FINAL: 123456.789      # ← passed

Three failures, all of them real:

    Ungrounded numbers. 123456.789 was fabricated — no execution ever printed it.
    Position-dependent heuristics. The verdict flipped between step 8 and step 9
    with no change to the code: the stale KeyError had simply scrolled out of the
    heuristic's 2000-character window. It was judging text position, not correctness.
    No rule binding answers to output. Nothing required the reported value to
    exist in the last run's stdout.

The fix is a deterministic rule, not a bigger model:

    Every number the answer asserts that the question did not supply must appear in the
    observation of the last code run — and the observation is read only from the
    interpreter's output, never from the model's own narration.

Same question, same local 7B, after the fix:

[python]
2297600.2957
FINAL: 2297600.2957          ok=True   steps=5

Independently recomputed gold answer: 2297600.3 — an exact match, not a
plausible-looking one. This is the difference between a demo and an instrument.
Then the benchmark itself turned out to be broken

Two consecutive runs of the identical configuration scored 33.3% and 25.0%.

Two causes, both in my code:

    Self-contaminating measurement. The eval called learn() inside its own task
    loop, so each task's prompt carried the failure hints of every earlier task. Task 11
    scored differently inside the eval than it did standalone — same question, same model.
    Unstable sampling. Greedy decoding on a local runtime isn't bit-stable across
    runs, so near-tied tokens flipped between runs.

Fixes: memory is now frozen during measurement (the agent still learns when you use
it interactively), and sampling is pinned with seed=42. Two consecutive runs now
produce identical verdicts on every task.

A benchmark you can't re-run is worth less than no benchmark, because it launders
noise into confidence. That lesson is the project.
Architecture

flowchart TD
    Q[Business question] --> RT[Router: simple vs complex]
    RT --> LLM[Local LLM — qwen2.5:7b]
    LLM -->|tool: list_files| DD[Data dictionary + schema]
    LLM -->|tool: python| SB[Sandboxed pandas execution]
    DD --> LLM
    SB -->|stdout| LLM
    LLM -->|final_answer| G{Grounding gate}
    G -->|number absent from stdout| LLM
    G -->|grounded| A[Verified answer]
    A --> OUT[Answer]

agent/
  orchestrator.py   plan -> act -> verify loop (hand-rolled, no framework)
  verifier.py       deterministic grounding gate (hard) + optional model audit (veto-only)
  router.py         complexity classifier -> model selection (heuristic or trained)
  memory.py         failure memory: past failures retrieved into later prompts
  prompts.py        system rules: never guess numbers, print before answering
  tools.py          list_files (data dictionary), run_python (sandboxed subprocess)
evals/run_eval.py   gold-answer scoring, versioned results in results/history/
scripts/            deterministic seeded dataset + eval-set builder (gold computed, never typed)
training/           scaffolds for the verifier/router fine-tuning phase

Domain rule enforced in the prompt and the data dictionary:
revenue = units × unit_price × (1 − discount), and revenue questions exclude
refunded orders unless the question says otherwise.
Quickstart

brew install ollama
ollama pull qwen2.5:7b

cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_data.py     # 5,000-row seeded dataset + data dictionary
python scripts/build_eval_set.py    # 12 gold-labeled tasks

python -m agent.main "What was total revenue in 2024?" --verbose
python evals/run_eval.py            # ~30 min on a local 7B

.env just points the OpenAI-compatible client at your local Ollama server. There is
no key to obtain.
Current failure taxonomy

Failures are stable across runs, which is what makes them worth fixing:
task 	failure 	class
t01 	refund rule ignored — number is grounded but computed under the wrong definition 	semantic
t03 	average-order-value filter wrong by ~20%, every run 	semantic
t05 	asserts 0.0 for a label question 	model capability
t07–t11 	code errors or prints nothing → gate blocks → no answer returned 	print discipline

The gate is deliberately conservative: no answer beats a wrong answer. That's why
several failures surface as no-answer rather than a confident wrong number.

Note the class that matters most: the gate can prove a number was printed. It cannot
prove the definition was right. Catching t01 and t03 requires a learned verifier —
which is exactly what the roadmap targets.
Honest limitations

    The sandbox is a subprocess with a timeout, not a container.
    12 tasks is a pilot, not a benchmark; the builder computes gold from the data, so
    scaling it is mechanical, but the committed numbers are on 12.
    A local 7B is weak at strict-JSON, printing discipline and multi-step arithmetic.
    That's the baseline this project measures against, not a bug to hide.
    seed=42 makes outcomes reproducible; token-level sampling still drifts slightly.

Roadmap (planned, not yet done)

    Generate verifier training data from the eval set — gold answers give free labels
    (correct traces → verdict: true, perturbed gold → verdict: false).
    LoRA fine-tune a small verifier on (question, answer, trace) → verdict; serve it
    and measure the delta against v3.
    Train the complexity router and report the success/cost trade-off curve.
    Measure retrieval over past failures as a delta, not a vibe.

What this demonstrates

    Building an agent loop from scratch, without a framework, where the control flow is
    visible and debuggable
    Designing a deterministic verification rule instead of trusting a model to grade
    its own output — including the exact case where that rule can't help
    Building an eval harness with computed (never hand-typed) gold answers, versioned
    results, and a documented failure taxonomy
    Reproducibility engineering: seeding, measurement isolation, and the discipline to
    distrust a number that moves between identical runs
    Local open-weights inference: quantised model serving, sandboxed code execution,
    token accounting
