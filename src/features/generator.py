"""
generator.py
-------------
Takes a user query, retrieves context via FAISS, and generates an answer using an LLM.
"""

import yaml
from pathlib import Path
from src.utils.logger import setup_logger
from src.features.retriever import FaissRetriever, load_config
import textwrap
import subprocess

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def generate_answer(query: str, retriever=None):
    cfg = load_config()
    logger = setup_logger("generator", f"{cfg['paths']['logs']}/generator.log")
    retriever = retriever or FaissRetriever(cfg, logger)

    logger.info(f"Generating answer for query: {query}")
    results = retriever.search(query, mmr=True)

    context = "\n\n".join(
        [f"[{r['source']}] {r['text']}" for r in results]
    )

    prompt = f"""
You are an intelligent research assistant.  
Answer the user's question **using only** the information provided below.  
If the answer is not explicitly stated, say "Not enough information in the sources."

Question:
{query}

Context:
{context}

Answer:
""".strip()

    # if OpenAI:
    #     client = OpenAI()
    #     response = client.chat.completions.create(
    #         model=cfg["rag"].get("llm_model", "gpt-4o-mini"),
    #         messages=[
    #             {"role": "system", "content": "You are a precise and concise assistant."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=cfg["rag"].get("temperature", 0.3)
    #     )
    #     answer = response.choices[0].message.content.strip()
    # else:
    #     # fallback (no API) → display retrieved context summary
    #     answer = "⚠️ OpenAI not installed/configured. Retrieved context only:\n" + textwrap.shorten(context, 1500)

    # logger.info("Answer generation completed.")
    # return answer, results
        # Use Ollama for generation
    model_name = cfg["rag"].get("llm_model", "mistral")
    logger.info(f"Running Ollama model: {model_name}")

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        answer = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Ollama execution failed: {e}")
        answer = f"⚠️ Error: Ollama failed — {e}"

    logger.info("Answer generation completed.")
    return answer, results



if __name__ == "__main__":
    q = input("Enter your question: ")
    ans, refs = generate_answer(q)
    print("\n=== ANSWER ===\n")
    print(ans)
    print("\n--- Sources ---")
    for r in refs:
        print(f"{r['source']}  (chunk {r['chunk_id']})")
