import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_answer(query, context_chunks):
    """Generate answer using free Groq API (LLaMA 3) — fast & no quota issues."""

    context = "\n\n".join([
        f"[Source: {meta['filename']}, Page {meta['page']}]\n{doc}"
        for _, doc, meta in context_chunks
    ])

    prompt = f"""You are an expert agricultural advisor for Indian farmers.
Use ONLY the context below to answer the farmer's question.
Give practical, specific advice. Include crop name, quantities, timing, and actionable steps.
If the answer is not in the context, say "I don't have enough information on this topic."
Keep your answer clear and concise — easy for a farmer to understand.

CONTEXT:
{context}

FARMER'S QUESTION: {query}

ANSWER:"""

    try:
        print(f"  Calling Groq API (LLaMA 3)...")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        print(f"  Answer received ({len(answer)} chars)")
        return answer

    except Exception as e:
        print(f"  Groq ERROR: {e}")
        return None