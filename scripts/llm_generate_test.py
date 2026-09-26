import json
import time
import urllib.request
from pathlib import Path


# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:8b"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = PROJECT_ROOT / "tests" / "vlm_example.json"


# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = """
Ты — голос робота AstraMind, который смотрит через камеру
на человека и рассказывает о том, что видит.

Придумай естественное описание на русском языке,
которое робот сможет произнести вслух.

Правила:
- используй только информацию из JSON;
- описывай человека естественно, а не как список характеристик;
- можно упомянуть другие наблюдаемые особенности, если это помогает сделать фразу живее;
- можно начать с фразы вроде «О, вижу перед собой прекрасного мужчину или девушку...»;
- не перечисляй поля JSON;
- не упоминай JSON, нейросеть или анализ изображения;
- текст должен звучать живо, мило и естественно;
- 2–3 предложения;
- верни только готовый текст для произнесения.
""".strip()


# =========================
# OLLAMA
# =========================

def call_ollama(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.25,
            "num_predict": 100
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result


# =========================
# MAIN
# =========================

def main():
    observations = json.loads(
        JSON_PATH.read_text(encoding="utf-8")
    )

    json_text = json.dumps(
        observations,
        ensure_ascii=False,
        indent=2
    )

    prompt = (
        SYSTEM_PROMPT
        + "\n\nНаблюдения компьютерного зрения:\n"
        + json_text
    )

    print("=== LLM ===")
    print(f"Model: {MODEL}")
    print()

    start = time.perf_counter()

    result = call_ollama(prompt)

    elapsed = time.perf_counter() - start

    text = result["message"]["content"].strip()

    print("=== OUTPUT ===")
    print()
    print(text)
    print()

    print(f"Time: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()
