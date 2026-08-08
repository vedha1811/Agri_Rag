import os
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1"
)

MODEL_ID = "Meta-Llama-3.3-70B-Instruct"

MODEL_NAME = "Llama 3.1 8B (Sambanova)"

def generate_answer(query, context_chunks):

    context = "\n\n".join([
        f"[Source: {m['filename']}, Page {m['page']}]\n{doc}"
        for _, doc, m in context_chunks
    ])

    prompt = f"""
You are an expert agricultural advisor for Indian farmers.

Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:

        start = time.time()

        r = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=400
        )

        elapsed = round(time.time() - start, 2)

        answer = r.choices[0].message.content.strip()

        tokens = r.usage.completion_tokens

        print(f"  [{MODEL_NAME}] {elapsed}s | {tokens} tokens")

        return {
            "answer": answer,
            "model": MODEL_NAME,
            "time_s": elapsed,
            "tokens": tokens
        }

    except Exception as e:

        print(f"  [{MODEL_NAME}] ERROR: {e}")

        return None