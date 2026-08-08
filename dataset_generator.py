"""
dataset_generator.py
─────────────────────────────────────────────────────────────────
Generates a reference Q&A dataset from your ChromaDB chunks.
Uses Groq (free, fast) to create ground-truth answers.

These ground-truth answers are what ROUGE and BLEU scores are
calculated AGAINST in evaluate.py.

Run:
    python dataset_generator.py

Output:
    reference_dataset.json   ← used by evaluate.py for metrics
    reference_dataset.csv    ← human-readable version

Requirements:
    pip install groq
    export GROQ_API_KEY="your-key"   (free at console.groq.com)
"""

import json
import csv
import time
import os
import random
from retriever import retrieve, collection
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────
OUTPUT_JSON = "reference_dataset.json"
OUTPUT_CSV  = "reference_dataset.csv"

# How many Q&A pairs to generate per question type
PAIRS_PER_TYPE = 3   # 3 types × 10 crops × 3 = ~90 pairs

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# ──────────────────────────────────────────────────────────────


# ─── Test questions covering your crops and document topics ───
TEST_QUESTIONS = [
    # (question, crop_filter, question_type)

    # Cereal crops
    ("What is the recommended seed rate for rice per hectare?",            "rice",      "factual"),
    ("What are the major diseases that affect rice crop?",                 "rice",      "factual"),
    ("How to control stem borer in paddy crop?",                          "rice",      "howto"),
    ("What is the optimum temperature for rice cultivation?",             "rice",      "factual"),
    ("What fertilizer schedule should be followed for irrigated rice?",   "rice",      "howto"),

    ("What is the seed rate for wheat per hectare?",                      "wheat",     "factual"),
    ("How to manage yellow rust in wheat?",                               "wheat",     "howto"),
    ("What are the critical irrigation stages in wheat?",                 "wheat",     "factual"),
    ("What is the recommended sowing time for wheat in rabi season?",     "wheat",     "factual"),
    ("How to control aphids in wheat crop?",                              "wheat",     "howto"),

    ("What is the sowing time for maize in kharif season?",               "maize",     "factual"),
    ("How to control fall armyworm in maize?",                            "maize",     "howto"),
    ("What are the symptoms of downy mildew in maize?",                   "maize",     "diagnostic"),
    ("What is the spacing recommended for maize cultivation?",            "maize",     "factual"),

    # Pulses
    ("What is the seed rate for blackgram per hectare?",                  "blackgram", "factual"),
    ("What are the symptoms of yellow mosaic virus in blackgram?",        "blackgram", "diagnostic"),
    ("How to manage yellow mosaic virus in blackgram?",                   "blackgram", "howto"),

    ("What is the recommended variety of greengram for kharif season?",   "greengram", "factual"),
    ("How to control leaf spot disease in greengram?",                    "greengram", "howto"),

    ("What is the sowing time for redgram in India?",                     "redgram",   "factual"),
    ("How to manage wilt disease in pigeonpea?",                          "redgram",   "howto"),

    ("What is the seed rate for soyabean per hectare?",                   "soyabean",  "factual"),
    ("How to control soybean mosaic virus?",                              "soyabean",  "howto"),

    # Oilseeds
    ("What is the irrigation schedule for groundnut kharif crop?",        "groundnut", "factual"),
    ("How to control tikka leaf spot disease in groundnut?",              "groundnut", "howto"),
    ("What is the recommended spacing for groundnut crop?",               "groundnut", "factual"),

    ("What fertilizer dose should be applied to sunflower crop?",         "sunflower", "factual"),
    ("How to manage downy mildew in sunflower?",                          "sunflower", "howto"),

    ("What is the seed treatment recommended for sesame?",                "sesame",    "factual"),

    # Fibre crops
    ("When should cotton be sown in India?",                              "cotton",    "factual"),
    ("What fertilizer should be applied to cotton at 30 DAS?",            "cotton",    "factual"),
    ("How to control bollworms in cotton crop?",                          "cotton",    "howto"),
    ("What are the symptoms of leaf curl virus in cotton?",               "cotton",    "diagnostic"),
    ("What is the recommended plant spacing for cotton?",                 "cotton",    "factual"),

    # Vegetables
    ("What pests commonly affect brinjal crop?",                          "brinjal",   "factual"),
    ("How to manage shoot and fruit borer in brinjal?",                   "brinjal",   "howto"),
    ("What diseases affect tomato plants?",                               "tomato",    "factual"),
    ("How to control leaf curl virus in chilli?",                         "chilli",    "howto"),
    ("What are the symptoms of powdery mildew in chilli?",                "chilli",    "diagnostic"),

    # Millets
    ("What is the sowing time for sorghum jowar crop?",                   "sorghum",   "factual"),
    ("How to manage shoot fly in sorghum?",                               "sorghum",   "howto"),

    # Horticultural
    ("What are the common pests of banana crop?",                         "banana",    "factual"),
    ("How to manage sigatoka leaf spot in banana?",                       "banana",    "howto"),
    ("What is the fertilizer schedule for mango?",                        "mango",     "factual"),
    ("How much water does sugarcane need per irrigation?",                "sugarcane", "factual"),
    ("What is the spacing recommended for coconut plantation?",           "coconut",   "factual"),

    # General IPM
    ("What is Integrated Pest Management in agriculture?",                "ipm",       "factual"),
    ("How to use neem-based pesticides in crop protection?",              "ipm",       "howto"),
]


# ─── Groq reference answer generator ─────────────────────────
def generate_reference_answer(question, crop, chunks):
    """
    Generate a concise, factual reference answer using Groq.
    This becomes the 'ground truth' for ROUGE/BLEU comparison.
    """
    if not GROQ_API_KEY:
        print("  ⚠ GROQ_API_KEY not set — using chunk text as reference")
        # Fallback: use first chunk as reference
        if chunks:
            return chunks[0][1][:400]
        return ""

    client = Groq(api_key=GROQ_API_KEY)

    context = "\n\n".join([
        f"[Source: {m['filename']}, Page {m['page']}]\n{doc}"
        for _, doc, m in chunks[:3]
    ])

    prompt = f"""You are an agricultural expert. Answer the question below using ONLY the provided context.
Write a factual, complete answer in 3-5 sentences. Include specific quantities, timing, and crop names.
This answer will serve as the reference/ground-truth for evaluation.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (factual, 3-5 sentences):"""

    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.1,   # low temp = more factual
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Groq error: {e}")
        if chunks:
            return chunks[0][1][:400]
        return ""


# ─── Main generation loop ─────────────────────────────────────
def build_dataset():
    print("=" * 60)
    print("  Reference Dataset Generator")
    print(f"  Total questions : {len(TEST_QUESTIONS)}")
    print(f"  ChromaDB chunks : {collection.count()}")
    print("=" * 60)

    dataset = []

    for i, (question, crop, q_type) in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] {question[:55]}...")
        print(f"  Crop: {crop} | Type: {q_type}")

        # Retrieve relevant chunks
        chunks = retrieve(question, crop_filter=crop)

        if not chunks:
            # Try without crop filter
            chunks = retrieve(question, crop_filter=None)
            print(f"  ⚠ No crop-filtered results, using unfiltered ({len(chunks)} chunks)")

        if not chunks:
            print(f"  ✗ No chunks found — skipping")
            continue

        # Generate reference answer
        reference = generate_reference_answer(question, crop, chunks)

        if not reference or len(reference.strip()) < 20:
            print(f"  ✗ Reference answer too short — skipping")
            continue

        # Store
        entry = {
            "id":              i + 1,
            "question":        question,
            "crop":            crop,
            "question_type":   q_type,
            "reference_answer": reference,
            "source_docs":     list(set(m["filename"] for _, _, m in chunks[:3])),
            "num_chunks":      len(chunks),
        }
        dataset.append(entry)
        print(f"  ✓ Reference answer generated ({len(reference.split())} words)")
        print(f"    Preview: {reference[:100]}...")

        time.sleep(1.2)  # Groq rate limit

    # ── Save JSON ──────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved JSON → {OUTPUT_JSON}  ({len(dataset)} entries)")

    # ── Save CSV ───────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "question", "crop", "question_type",
            "reference_answer", "source_docs", "num_chunks"
        ])
        writer.writeheader()
        for row in dataset:
            row["source_docs"] = ", ".join(row["source_docs"])
            writer.writerow(row)
    print(f"✅ Saved CSV  → {OUTPUT_CSV}")

    # ── Summary ────────────────────────────────────────────────
    from collections import Counter
    type_counts = Counter(d["question_type"] for d in dataset)
    crop_counts = Counter(d["crop"] for d in dataset)

    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY — {len(dataset)} Q&A pairs")
    print(f"{'='*60}")
    print("  By question type:")
    for t, c in type_counts.most_common():
        print(f"    {t:<15}: {c}")
    print("  By crop (top 10):")
    for crop, c in crop_counts.most_common(10):
        print(f"    {crop:<15}: {c}")
    print(f"{'='*60}")
    print("\nNext step: Run  python evaluate.py  to compute ROUGE/BLEU scores.")

    return dataset


if __name__ == "__main__":
    build_dataset()