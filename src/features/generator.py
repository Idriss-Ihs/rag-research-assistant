"""
generator.py
-------------
Flexible RAG answer generator with multiple backends:
- OpenAI
- Gemini (Google)
- Ollama (local open-source)
"""

import os
import json
import yaml
import textwrap
import subprocess
import requests
from pathlib import Path
from src.utils.logger import setup_logger
from src.features.retriever import FaissRetriever, load_config

# Optional import for OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def call_openai(prompt, model, temperature):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise and concise assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def call_gemini(prompt, model, temperature):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }

    resp = requests.post(endpoint, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(data, indent=2)


def call_ollama(prompt, model):
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"⚠️ Ollama failed: {e}"


def generate_answer(query: str, retriever=None):
    cfg = load_config()
    backend = cfg["rag"].get("backend", "ollama")
    model = cfg["rag"].get("llm_model", "mistral")
    temperature = cfg["rag"].get("temperature", 0.3)

    logger = setup_logger("generator", f"{cfg['paths']['logs']}/generator.log")
    retriever = retriever or FaissRetriever(cfg, logger)

    logger.info(f"[Backend: {backend}] Generating answer for query: {query}")
    results = retriever.search(query, mmr=True)

    context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in results)
    prompt = f"""
You are a research assistant with access to retrieved scientific documents.
Answer the question **only** using the context below.
If unsure, respond with: "Not enough information in the sources."

Question:
{query}

Context:
{context}

Answer:
""".strip()

    try:
        if backend == "openai":
            answer = call_openai(prompt, model, temperature)
        elif backend == "gemini":
            answer = call_gemini(prompt, model, temperature)
        else:
            answer = call_ollama(prompt, model)

        logger.info("Answer generation completed successfully.")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        answer = f"⚠️ Generation error: {e}"

    return answer, results


if __name__ == "__main__":
    q = input("Enter your question: ")
    ans, refs = generate_answer(q)
    print("\n=== ANSWER ===\n")
    print(ans)
    print("\n--- Sources ---")
    for r in refs:
        print(f"{r['source']} (chunk {r['chunk_id']})")
