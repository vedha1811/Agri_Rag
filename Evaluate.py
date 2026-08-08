


import csv
import json
import time
import os
import math
from collections import defaultdict

# ── NLP metric libraries ──────────────────────────────────────
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠ rouge-score not installed. Run: pip install rouge-score")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    # Download required NLTK data silently
    for pkg in ["punkt", "wordnet", "punkt_tab", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠ nltk not installed. Run: pip install nltk")

from retriever import retrieve

# ── LLM Imports ───────────────────────────────────────────────
import llm_groq
import llm_cohere
import llm_sarvam
import llm_cerebras
import llm_sambanova

# ── Models to evaluate ────────────────────────────────────────
LLMS = [
    ("Groq — LLaMA 3.1 8B",       llm_groq),
    ("Cohere — Command-R",         llm_cohere),
    ("Cerebras — LLaMA 3.1 8B",   llm_cerebras),
    ("Sarvam AI — Sarvam-M",      llm_sarvam),
    #("Sambanova — LLaMA 3.3 70B", llm_sambanova),
]

REFERENCE_FILE = "reference_dataset.json"

# ─────────────────────────────────────────────────────────────
#  METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    ROUGE = Recall-Oriented Understudy for Gisting Evaluation
    Range: 0.0 to 1.0 (higher = better)

    ROUGE-1: How many words in the reference appear in the answer?
    ROUGE-2: How many 2-word pairs match?
    ROUGE-L: How long is the longest common word sequence?
    """
    if not ROUGE_AVAILABLE or not hypothesis or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True   # treats "growing" and "grows" as same
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(hypothesis: str, reference: str) -> dict:
    """
    Compute BLEU-1, BLEU-2, BLEU-4 scores.
    BLEU = Bilingual Evaluation Understudy
    Range: 0.0 to 1.0 (higher = better)

    BLEU-1: 1-gram precision (individual word matches)
    BLEU-2: 2-gram precision (consecutive word pair matches)
    BLEU-4: 4-gram precision (standard BLEU, strictest)

    Note: BLEU was designed for translation but works for QA evaluation too.
    Use smoothing to avoid zero scores for short answers.
    """
    if not NLTK_AVAILABLE or not hypothesis or not reference:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

    try:
        smoothie = SmoothingFunction().method4   # Chen & Cherry smoothing

        hyp_tokens = word_tokenize(hypothesis.lower())
        ref_tokens = [word_tokenize(reference.lower())]   # list of references

        if not hyp_tokens or not ref_tokens[0]:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

        bleu1 = sentence_bleu(ref_tokens, hyp_tokens,
                              weights=(1, 0, 0, 0),
                              smoothing_function=smoothie)
        bleu2 = sentence_bleu(ref_tokens, hyp_tokens,
                              weights=(0.5, 0.5, 0, 0),
                              smoothing_function=smoothie)
        bleu4 = sentence_bleu(ref_tokens, hyp_tokens,
                              weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smoothie)
        return {
            "bleu1": round(bleu1, 4),
            "bleu2": round(bleu2, 4),
            "bleu4": round(bleu4, 4),
        }
    except Exception as e:
        print(f"    BLEU error: {e}")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}


def compute_meteor(hypothesis: str, reference: str) -> float:
    """
    Compute METEOR score.
    METEOR = Metric for Evaluation of Translation with Explicit ORdering
    Range: 0.0 to 1.0 (higher = better)

    METEOR is better than BLEU because it:
    - Considers synonyms (grow ≈ cultivate)
    - Weighs recall more than precision (important for agriculture advice)
    - Handles stemming (irrigating ≈ irrigated)
    """
    if not NLTK_AVAILABLE or not hypothesis or not reference:
        return 0.0
    try:
        hyp_tokens = word_tokenize(hypothesis.lower())
        ref_tokens = word_tokenize(reference.lower())
        if not hyp_tokens or not ref_tokens:
            return 0.0
        score = meteor_score([ref_tokens], hyp_tokens)
        return round(float(score), 4)
    except Exception as e:
        print(f"    METEOR error: {e}")
        return 0.0


def heuristic_score(answer: str, crop: str) -> dict:
    """
    Heuristic quality metrics (no reference answer needed).
    These were in the original evaluate.py — kept for comparison.
    """
    if not answer or answer == "ERROR":
        return {"relevance": 0.0, "completeness": 0.0, "practicality": 0.0, "heuristic_avg": 0.0}

    ans = answer.lower()

    # Relevance: does answer mention the crop?
    relevance = 1.0 if crop.lower() in ans else 0.5

    # Completeness: word count
    words = len(answer.split())
    if words >= 80:   completeness = 1.0
    elif words >= 40: completeness = 0.7
    else:             completeness = 0.3

    # Practicality: agricultural action keywords
    action_kws = [
        "kg", "litre", "ml", "week", "day", "month",
        "spray", "apply", "sow", "irrigat", "harvest",
        "dose", "treat", "june", "july", "august", "october",
        "%", "temperature", "hectare", "ha", "seed rate",
        "fertilizer", "pesticide", "fungicide", "insecticide",
    ]
    hits = sum(1 for kw in action_kws if kw in ans)
    practicality = min(hits / 4.0, 1.0)

    avg = round((relevance + completeness + practicality) / 3, 4)
    return {
        "relevance":     round(relevance, 4),
        "completeness":  round(completeness, 4),
        "practicality":  round(practicality, 4),
        "heuristic_avg": avg,
    }


# ─────────────────────────────────────────────────────────────
#  LOAD REFERENCE DATASET
# ─────────────────────────────────────────────────────────────
def load_reference_dataset():
    if not os.path.exists(REFERENCE_FILE):
        print(f"⚠  {REFERENCE_FILE} not found!")
        print("   Run: python dataset_generator.py  first.")
        print("   Falling back to heuristic-only evaluation.\n")
        return []

    with open(REFERENCE_FILE, encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} reference Q&A pairs from {REFERENCE_FILE}")
    return data


# ─────────────────────────────────────────────────────────────
#  MAIN EVALUATION
# ─────────────────────────────────────────────────────────────
def run_evaluation():

    print("=" * 70)
    print("  Agricultural RAG System — Full Evaluation")
    print(f"  Models  : {len(LLMS)}")
    print(f"  Metrics : ROUGE-1/2/L, BLEU-1/2/4, METEOR, Heuristic")
    print("=" * 70)

    reference_data = load_reference_dataset()

    # If no reference file, use built-in test questions (heuristic only)
    if not reference_data:
        reference_data = [
            {"id": i+1, "question": q, "crop": c,
             "question_type": "factual", "reference_answer": ""}
            for i, (q, c) in enumerate([
                ("When should I sow cotton in India?",                "cotton"),
                ("How to treat blast disease in rice?",               "rice"),
                ("What is the fertilizer dose for wheat?",            "wheat"),
                ("What pests affect brinjal crops?",                  "brinjal"),
                ("How to manage irrigation for groundnut?",           "groundnut"),
                ("What are symptoms of yellowing in maize?",          "maize"),
                ("How to control leaf curl virus in chilli?",         "chilli"),
                ("What is the sowing time for soyabean?",             "soyabean"),
                ("How much water does sugarcane need?",               "sugarcane"),
                ("What diseases affect tomato plants?",               "tomato"),
            ])
        ]

    all_results  = []
    model_scores = defaultdict(lambda: defaultdict(list))

    # ── Evaluation loop ───────────────────────────────────────
    for q_idx, item in enumerate(reference_data):
        question  = item["question"]
        crop      = item["crop"]
        reference = item.get("reference_answer", "")
        q_type    = item.get("question_type", "factual")

        print(f"\nQ{q_idx+1}/{len(reference_data)}: {question[:55]}... [{crop}]")

        # Retrieve same chunks for all models (fair comparison)
        chunks = retrieve(question, crop_filter=crop)
        if not chunks:
            chunks = retrieve(question, crop_filter=None)

        for llm_name, llm_mod in LLMS:

            result = llm_mod.generate_answer(question, chunks)

            if result and result.get("answer"):
                answer = result["answer"]

                # ── NLP Metrics ──────────────────────────────
                rouge  = compute_rouge(answer, reference)  if reference else {"rouge1": None, "rouge2": None, "rougeL": None}
                bleu   = compute_bleu(answer, reference)   if reference else {"bleu1": None, "bleu2": None, "bleu4": None}
                meteor = compute_meteor(answer, reference) if reference else None

                # ── Heuristic Metrics ────────────────────────
                heur   = heuristic_score(answer, crop)

                # ── Combined Score ───────────────────────────
                # If we have NLP metrics, use them; otherwise use heuristic
                if reference and rouge["rouge1"] is not None:
                    nlp_avg = round(
                        (rouge["rouge1"] + rouge["rougeL"] +
                         (bleu["bleu4"] or 0) + (meteor or 0)) / 4, 4
                    )
                    combined = round((nlp_avg + heur["heuristic_avg"]) / 2, 4)
                else:
                    nlp_avg  = None
                    combined = heur["heuristic_avg"]

                # ── Store result ─────────────────────────────
                row = {
                    "q_id":          item["id"],
                    "question":      question,
                    "crop":          crop,
                    "question_type": q_type,
                    "llm":           llm_name,
                    "answer":        answer,
                    "time_s":        result["time_s"],
                    "tokens":        result["tokens"],

                    # NLP Metrics
                    "rouge1":  rouge["rouge1"],
                    "rouge2":  rouge["rouge2"],
                    "rougeL":  rouge["rougeL"],
                    "bleu1":   bleu["bleu1"],
                    "bleu2":   bleu["bleu2"],
                    "bleu4":   bleu["bleu4"],
                    "meteor":  meteor,

                    # Heuristic
                    "relevance":     heur["relevance"],
                    "completeness":  heur["completeness"],
                    "practicality":  heur["practicality"],
                    "heuristic_avg": heur["heuristic_avg"],

                    # Overall
                    "combined_score": combined,
                }
                all_results.append(row)

                # Accumulate for summary
                for metric in ["rouge1", "rouge2", "rougeL", "bleu1", "bleu2", "bleu4", "meteor",
                                "relevance", "completeness", "practicality", "heuristic_avg"]:
                    val = row[metric]
                    if val is not None:
                        model_scores[llm_name][metric].append(val)
                model_scores[llm_name]["time_s"].append(result["time_s"])
                model_scores[llm_name]["combined"].append(combined)

                # Print progress
                r1 = f"{rouge['rouge1']:.3f}" if rouge["rouge1"] is not None else " N/A "
                bl = f"{bleu['bleu4']:.3f}"   if bleu["bleu4"]   is not None else " N/A "
                mt = f"{meteor:.3f}"           if meteor          is not None else " N/A "
                print(f"    ✓ {llm_name:<35}  R1={r1}  BLEU4={bl}  METEOR={mt}  time={result['time_s']}s")

            else:
                # Model failed
                row = {
                    "q_id": item["id"], "question": question, "crop": crop,
                    "question_type": q_type, "llm": llm_name, "answer": "ERROR",
                    "time_s": 0, "tokens": 0,
                    "rouge1": 0, "rouge2": 0, "rougeL": 0,
                    "bleu1": 0, "bleu2": 0, "bleu4": 0, "meteor": 0,
                    "relevance": 0, "completeness": 0, "practicality": 0,
                    "heuristic_avg": 0, "combined_score": 0,
                }
                all_results.append(row)
                print(f"    ✗ {llm_name} — FAILED")

                        # Prevent API rate limits

            if "Sambanova" in llm_name:
                time.sleep(10)

            elif "Cohere" in llm_name:
                time.sleep(4)

            else:
                time.sleep(1)   # rate limit

    # ─────────────────────────────────────────────────────────
    #  SAVE FULL RESULTS CSV
    # ─────────────────────────────────────────────────────────
    csv_path = "evaluation_results_full.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✅ Saved: {csv_path}")

    # ─────────────────────────────────────────────────────────
    #  BUILD SUMMARY TABLE
    # ─────────────────────────────────────────────────────────
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary_rows = []
    for llm_name, _ in LLMS:
        s = model_scores[llm_name]
        if not s.get("combined"):
            continue
        row = {
            "model":        llm_name,
            "rouge1":       avg(s.get("rouge1", [])),
            "rouge2":       avg(s.get("rouge2", [])),
            "rougeL":       avg(s.get("rougeL", [])),
            "bleu1":        avg(s.get("bleu1", [])),
            "bleu2":        avg(s.get("bleu2", [])),
            "bleu4":        avg(s.get("bleu4", [])),
            "meteor":       avg(s.get("meteor", [])),
            "relevance":    avg(s.get("relevance", [])),
            "completeness": avg(s.get("completeness", [])),
            "practicality": avg(s.get("practicality", [])),
            "heuristic_avg":avg(s.get("heuristic_avg", [])),
            "combined":     avg(s.get("combined", [])),
            "avg_time_s":   avg(s.get("time_s", [])),
            "num_questions": len(s.get("combined", [])),
        }
        summary_rows.append(row)

    summary_rows.sort(key=lambda x: -x["combined"])

    # Save NLP summary CSV
    nlp_csv = "rouge_bleu_summary.csv"
    with open(nlp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys() if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"✅ Saved: {nlp_csv}")

    # ─────────────────────────────────────────────────────────
    #  GENERATE REPORT
    # ─────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 75)
    lines.append("  Agricultural RAG System — Complete Evaluation Report")
    lines.append(f"  Questions evaluated : {len(reference_data)}")
    lines.append(f"  Models evaluated    : {len(LLMS)}")
    lines.append(f"  Metrics             : ROUGE-1/2/L, BLEU-1/2/4, METEOR, Heuristic")
    lines.append("=" * 75)

    # ── NLP Metrics table ─────────────────────────────────────
    lines.append("\n── NLP METRICS (ROUGE / BLEU / METEOR) ──────────────────────────")
    lines.append("")
    lines.append(f"{'Model':<35}{'R-1':>7}{'R-2':>7}{'R-L':>7}{'B-1':>7}{'B-4':>7}{'MTR':>7}")
    lines.append("-" * 70)
    for r in summary_rows:
        lines.append(
            f"{r['model']:<35}"
            f"{r['rouge1']:>7.4f}"
            f"{r['rouge2']:>7.4f}"
            f"{r['rougeL']:>7.4f}"
            f"{r['bleu1']:>7.4f}"
            f"{r['bleu4']:>7.4f}"
            f"{r['meteor']:>7.4f}"
        )

    # ── Heuristic Metrics table ───────────────────────────────
    lines.append("\n── HEURISTIC METRICS ─────────────────────────────────────────────")
    lines.append("")
    lines.append(f"{'Model':<35}{'Relev':>8}{'Compl':>8}{'Pract':>8}{'Avg':>8}{'Time':>8}")
    lines.append("-" * 75)
    for r in summary_rows:
        lines.append(
            f"{r['model']:<35}"
            f"{r['relevance']:>8.4f}"
            f"{r['completeness']:>8.4f}"
            f"{r['practicality']:>8.4f}"
            f"{r['heuristic_avg']:>8.4f}"
            f"{r['avg_time_s']:>7.2f}s"
        )

    # ── Combined ranking ──────────────────────────────────────
    lines.append("\n── COMBINED RANKING (NLP + Heuristic average) ───────────────────")
    lines.append("")
    lines.append(f"{'Rank':<6}{'Model':<35}{'Combined Score':>16}{'Avg Time':>10}")
    lines.append("-" * 70)
    for rank, r in enumerate(summary_rows, 1):
        medal = ["🥇", "🥈", "🥉", "  ", "  "][min(rank-1, 4)]
        lines.append(
            f"{medal} #{rank:<3}"
            f"{r['model']:<35}"
            f"{r['combined']:>14.4f}"
            f"{r['avg_time_s']:>9.2f}s"
        )

    # ── Winner summary ────────────────────────────────────────
    if summary_rows:
        winner   = summary_rows[0]
        fastest  = min(summary_rows, key=lambda x: x["avg_time_s"])
        best_nlp = max(summary_rows, key=lambda x: x["rouge1"])

        lines.append("")
        lines.append("=" * 75)
        lines.append(f"  🏆 BEST OVERALL  : {winner['model']}")
        lines.append(f"     Combined Score: {winner['combined']:.4f}")
        lines.append(f"     ROUGE-1       : {winner['rouge1']:.4f}")
        lines.append(f"     BLEU-4        : {winner['bleu4']:.4f}")
        lines.append(f"     METEOR        : {winner['meteor']:.4f}")
        lines.append("")
        lines.append(f"  ⚡ FASTEST        : {fastest['model']} ({fastest['avg_time_s']:.2f}s avg)")
        lines.append(f"  📊 BEST ROUGE-1   : {best_nlp['model']} ({best_nlp['rouge1']:.4f})")
        lines.append("")
        lines.append("  RECOMMENDATION FOR PROJECT DEMO:")
        lines.append(f"  Use '{winner['model']}' as the primary model.")
        lines.append(f"  Fallback to '{fastest['model']}' if speed is critical.")

    # ── Metric explanations ───────────────────────────────────
    lines.append("")
    lines.append("=" * 75)
    lines.append("  METRIC EXPLANATIONS (for your project report)")
    lines.append("=" * 75)
    lines.append("")
    lines.append("  ROUGE-1  : Word overlap between generated and reference answer.")
    lines.append("             Higher = more words from reference are in the answer.")
    lines.append("")
    lines.append("  ROUGE-2  : 2-word phrase overlap. Captures local fluency.")
    lines.append("             Higher = answer uses similar phrasing as reference.")
    lines.append("")
    lines.append("  ROUGE-L  : Longest common word sequence match.")
    lines.append("             Higher = answer follows similar word order.")
    lines.append("")
    lines.append("  BLEU-1   : 1-gram precision — are individual words correct?")
    lines.append("  BLEU-2   : 2-gram precision — are word pairs correct?")
    lines.append("  BLEU-4   : 4-gram precision — standard BLEU (most strict).")
    lines.append("             Originally for translation; adapted here for QA.")
    lines.append("")
    lines.append("  METEOR   : Considers synonyms + recall. Best for advisory text.")
    lines.append("             E.g. 'cultivate' ≈ 'grow', 'irrigating' ≈ 'irrigated'")
    lines.append("")
    lines.append("  Note: Scores above 0.3 are considered good for open-domain QA.")
    lines.append("  Agricultural QA typically scores 0.25–0.45 on ROUGE-1.")
    lines.append("=" * 75)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = "evaluation_report_full.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n✅ Saved: {report_path}")

    return all_results, summary_rows


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()