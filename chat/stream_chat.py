"""
Streaming chat: print assistant tokens as they arrive (OpenAI-compatible endpoint).
Configuration via .env: AI_GRID_KEY, BASE_URL, OSS_MODEL (or set STREAM_CHAT_MODEL to use another).
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("STREAM_CHAT_MODEL") or os.getenv("OSS_MODEL", "gpt-oss-120b")
client = OpenAI(api_key=AI_GRID_KEY, base_url=BASE_URL)


def stream_chat(user_message: str) -> str:
    """
    Send a user message and stream the assistant reply; print each delta and return full content.
    """
    full = []
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_message}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full.append(delta)
            print(delta, end="", flush=True)
    print()
    return "".join(full)


def main():
    """Stream a sample question and print the reply token-by-token."""
    prompt = "In 2–3 short sentences, what is machine learning?"
    print("User:", prompt)
    print("Assistant: ", end="", flush=True)
    stream_chat(prompt)


if __name__ == "__main__":
    main()
