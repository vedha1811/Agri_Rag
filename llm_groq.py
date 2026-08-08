"""
LLM 1: Groq — LLaMA 3.1 8B Instant
Provider : Groq Cloud
Model    : llama-3.1-8b-instant
Speed    : ~1 second
Cost     : Free
Get key  : https://console.groq.com
Setup    : pip install groq
           export GROQ_API_KEY="your-key"
"""
import os, time
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_ID  = "llama-3.1-8b-instant"
MODEL_NAME = "LLaMA 3.1 8B (Groq)"

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

    try:
        start = time.time()
        r = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        elapsed = round(time.time() - start, 2)
        answer  = r.choices[0].message.content.strip()
        tokens  = r.usage.completion_tokens
        print(f"  [{MODEL_NAME}] {elapsed}s | {tokens} tokens")
        return {"answer": answer, "model": MODEL_NAME, "time_s": elapsed, "tokens": tokens}
    except Exception as e:
        print(f"  [{MODEL_NAME}] ERROR: {e}")
        return None