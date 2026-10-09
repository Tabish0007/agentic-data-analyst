# Agentic Data Analyst

Agent that answers business questions over raw CSVs: plans, writes and runs
Python in a sandbox, a verifier model audits every final answer before it is
returned, and failures are stored as retrieved memory for future questions.

Components
- Orchestrator: plan -> act -> VERIFY loop (hand-rolled, no framework)
- Verifier: 7B model auditing (question, answer, trace); LoRA SFT via training/
- Router: trained task-complexity classifier (simple -> small model, complex -> big)
- Failure memory: failed traces retrieved into the prompt by similarity
- Eval harness: gold-labeled tasks, versioned runs in results/history/

Quickstart
  1. cp .env.example .env   (add your key)
  2. pip install -r requirements.txt
  3. python scripts/generate_data.py && python scripts/build_eval_set.py
  4. python -m agent.main "What was total revenue in 2024?"
  5. python evals/run_eval.py

The experiment (the resume part)
  v1 = baseline eval (heuristic verifier + heuristic router) -> commit results/history/
  v2 = + fine-tuned verifier served via vLLM (training/train_verifier.py)
  v3 = + trained router (training/train_router.py)
  Cost per task = avg tokens * provider per-token price.

Resume bullet template
  Agentic Data Analyst - agent answering business questions over raw data with a
  verify-in-the-loop design. Built a gold-labeled eval harness; LoRA-fine-tuned a
  7B answer-verifier and a task-complexity router, lifting task success _% -> _%
  at _x lower cost per task.
