#!/usr/bin/env python3

"""
AstraMind - VLM test

Отправляет одну фотографию в локальную Qwen3-VL
через Ollama и получает структурированное описание.

Пока:
- нет YOLO;
- нет LLM;
- нет MacBook;
- нет ESP32.

Только:
JPEG -> Qwen3-VL -> JSON
"""

import base64
import json
import urllib.request
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path("/tmp/astramind_vlm_test.jpg")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MODEL_NAME = "qwen3-vl:4b"

RESULT_PATH = Path("/tmp/astramind_vlm_result.json")


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Проанализируй изображение человека.

Верни ТОЛЬКО JSON без markdown и без дополнительного текста.

Используй строго следующие поля:

{
  "person": true,
  "appearance": "",
  "clothing": "",
  "expression": "",
  "hair": "",
  "facial_hair": "",
  "pose": ""
}

Правила:

- person: true, если человек действительно виден.
- appearance: видимое описание внешности, например "мужчина" или "женщина".
- clothing: кратко опиши видимую одежду и её цвет.
- expression: опиши видимое выражение лица, например "улыбается", "нейтральное выражение".
- hair: опиши только то, что действительно видно.
- facial_hair: например "борода", "усы", "нет заметной растительности", или "не видно".
- pose: краткое описание положения человека.

Не придумывай информацию, которой нет на изображении.
Не указывай возраст, профессию, характер или другие свойства,
которые нельзя надёжно определить по фотографии.
"""


# ============================================================
# SEND IMAGE TO OLLAMA
# ============================================================

def analyze_image(image_path: Path) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_base64 = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_base64],
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:

        response_data = json.loads(
            response.read().decode("utf-8")
        )

    content = response_data["message"]["content"]

    return json.loads(content)


# ============================================================
# MAIN
# ============================================================

def main():
    print("AstraMind: sending image to Qwen3-VL...")

    result = analyze_image(IMAGE_PATH)

    RESULT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("VLM result:")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(f"Saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()