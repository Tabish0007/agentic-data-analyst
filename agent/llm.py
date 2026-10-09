from openai import OpenAI
from config import CFG

def get_client(base_url: str | None = None) -> OpenAI:
    if base_url:
        return OpenAI(base_url=base_url, api_key="local")
    return OpenAI(**({"api_key": CFG.api_key} if CFG.api_key else {}))

def chat(messages, model, temperature=None, base_url=None, max_tokens=2048):
    resp = get_client(base_url).chat.completions.create(
        model=model, messages=messages,
        temperature=CFG.temperature if temperature is None else temperature,
        max_tokens=max_tokens,
        seed=42,
    )
    return resp.choices[0].message.content, (resp.usage.prompt_tokens, resp.usage.completion_tokens)
