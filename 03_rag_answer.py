from retriever import retrieve   # reuse retrieval from step 2

# ── This uses a free local LLM via Ollama ───────────────
# Install Ollama from https://ollama.com then run:
#   ollama pull mistral
import requests, json

def ask_ollama(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def rag_answer(query, crop_filter=None):
    # Step 1: Retrieve relevant chunks
    results = retrieve(query, crop_filter=crop_filter, top_k=10, rerank_top=5)

    # Step 2: Build context from top chunks
    context_parts = []
    for score, doc, meta in results:
        context_parts.append(
            f"[Source: {meta['filename']}, Page {meta['page']}]\n{doc}"
        )
    context = "\n\n".join(context_parts)

    # Step 3: Build the prompt
    prompt = f"""You are an expert agricultural advisor for Indian farmers.
Use ONLY the context below to answer the question. 
If the answer isn't in the context, say "I don't have enough information on this."
Always mention which crop and what action the farmer should take.

CONTEXT:
{context}

FARMER'S QUESTION: {query}

ANSWER:"""

    # Step 4: Generate answer
    answer = ask_ollama(prompt)
    return answer, results


# ── Run it ───────────────────────────────────────────────
if __name__ == "__main__":
    query = "My cotton leaves have yellow spots. What should I do?"

    print(f"Farmer Query: {query}\n")
    answer, sources = rag_answer(query, crop_filter="cotton")

    print("Answer:")
    print(answer)
    print("\nSources used:")
    for _, doc, meta in sources:
        print(f"  - {meta['filename']}, page {meta['page']}")