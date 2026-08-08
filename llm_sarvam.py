import os
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("SARVAM_API_KEY"),
    base_url="https://api.sarvam.ai/v1"
)

MODEL_ID = "sarvam-m"

MODEL_NAME = "Sarvam-M"

def generate_answer(query, context_chunks):

    context = "\n\n".join([
        f"[Source: {m['filename']}, Page {m['page']}]\n{doc}"
        for _, doc, m in context_chunks
    ])

    prompt = f"""
You are an expert agricultural advisor for Indian farmers.

Use ONLY the context below.

Give practical agricultural advice with:
- crop name
- disease/pest
- treatment
- timing
- quantities

If answer not found in context, say:
"I don't have enough information on this topic."

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
            max_tokens=400,
            temperature=0.3
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