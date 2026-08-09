"""
LLM 5: Cerebras — LLaMA 3.1 8B
Get key: https://cloud.cerebras.ai
Setup  : pip install cerebras-cloud-sdk
         export CEREBRAS_API_KEY="your-key"
Speed  : ~0.5s — fastest of all 5 providers
"""
import os, time
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
load_dotenv()

client     = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
MODEL_ID   = "gpt-oss-120b"
MODEL_NAME = "GPT-OSS 120B (Cerebras)"
def generate_answer(query, context_chunks):

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

FARMER'S QUESTION: {query}

ANSWER:"""

    start = time.time()

    try:

        r = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=400,
            temperature=0.3,
        )

        elapsed = round(time.time() - start, 2)
        content = r.choices[0].message.content

        if not content:
           print(f"  [{MODEL_NAME}] No final content returned")
           return None

        answer = content.strip()
        tokens = r.usage.completion_tokens

        print(
            f"  [{MODEL_NAME}] {elapsed}s | {tokens} tokens"
        )

        return {
            "answer": answer,
            "model": MODEL_NAME,
            "time_s": elapsed,
            "tokens": tokens
        }

    except Exception as error:

        error_message = str(error)

        print(
            f"  [{MODEL_NAME}] ERROR: {error_message}"
        )

        # Retry only for rate-limit errors
        if "429" in error_message:

            print(
                "  [Cerebras] Retrying after traffic delay..."
            )

            time.sleep(3)

            try:

                r = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )

                elapsed = round(time.time() - start, 2)
                answer = r.choices[0].message.content.strip()
                tokens = r.usage.completion_tokens

                return {
                    "answer": answer,
                    "model": MODEL_NAME,
                    "time_s": elapsed,
                    "tokens": tokens
                }

            except Exception as retry_error:

                print(
                    f"  [{MODEL_NAME}] RETRY ERROR: "
                    f"{retry_error}"
                )

        return None
