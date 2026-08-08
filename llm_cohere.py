import os, time
import cohere
from dotenv import load_dotenv

load_dotenv()

MODEL_ID   = "command-r-08-2024"
MODEL_NAME = "Command-R 08-2024 (Cohere)"


def generate_answer(query, context_chunks):

    client = cohere.ClientV2(
        api_key=os.getenv("CO_API_KEY")
    )

    context = "\n\n".join([
        f"[Source: {m['filename']}, Page {m['page']}]\n{doc}"
        for _, doc, m in context_chunks
    ])

    prompt = f"""You are an expert agricultural advisor for Indian farmers.

Use ONLY the context below to answer the question.

Give practical advice with crop name, quantities, timing and steps.

If not in context, say "I don't have enough information on this topic."

CONTEXT:
{context}

FARMER'S QUESTION:
{query}

ANSWER:
"""

    try:

        start = time.time()

        r = client.chat(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=400,
        )

        elapsed = round(time.time() - start, 2)

        answer = r.message.content[0].text.strip()

        tokens = (
            r.usage.tokens.output_tokens
            if r.usage else 0
        )

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