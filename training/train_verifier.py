import json

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
VERIFIER_SYSTEM = (
    "You are a strict numerical auditor. Given a question, a candidate answer, and the "
    "code/execution trace that produced it, decide if the answer is correct. "
    "Respond with ONLY JSON: {\"verdict\": true|false, \"reason\": \"...\", \"confidence\": 0.0-1.0}"
)

def build_messages(ex):
    user = (f"Question: {ex['question']}\nCandidate answer: {ex['answer']}\n"
            f"Trace:\n{ex['trace']}")
    assistant = json.dumps({"verdict": ex["verdict"], "reason": ""})
    return {
        "prompt": [{"role": "system", "content": VERIFIER_SYSTEM},
                   {"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": assistant}],
    }

def main() -> None:
    data = [json.loads(l) for l in open("data_verifier/train.jsonl") if l.strip()]
    ds = Dataset.from_list([build_messages(e) for e in data])

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
        device_map="auto")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    args = SFTConfig(output_dir="models/verifier-lora",
                     max_seq_length=4096, num_train_epochs=2,
                     per_device_train_batch_size=1, gradient_accumulation_steps=8,
                     learning_rate=2e-5, logging_steps=20, save_strategy="epoch",
                     report_to="none", bf16=True, gradient_checkpointing=True)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model("models/verifier-lora")
    tok.save_pretrained("models/verifier-lora")
    print("saved -> models/verifier-lora")

if __name__ == "__main__":
    main()
