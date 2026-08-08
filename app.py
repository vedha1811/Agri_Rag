import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
load_dotenv()
from config import (
    CROP_CHOICES,
    MODELS,
    DEFAULT_MODEL
)

from retriever import (
    retrieve,
    normalize_score,
    collection
)

# =========================================================
# LLM Imports
# =========================================================
import llm_groq
import llm_cohere
import llm_sarvam
import llm_cerebras
import llm_sambanova

# =========================================================
# Manual Model Mapping
# =========================================================
LLM_MAP = {

    "groq-llama":
        llm_groq,

    "cohere-command":
        llm_cohere,

    "sarvam-ai":
        llm_sarvam,

    "cerebras-llama":
        llm_cerebras,

    "sambanova-llama":
        llm_sambanova,
}

# =========================================================
# Auto Ranking
# =========================================================
BEST_MODELS = [

    ("cohere-command", llm_cohere),

    ("groq-llama", llm_groq),

    ("cerebras-llama", llm_cerebras),

    ("sarvam-ai", llm_sarvam),

    ("sambanova-llama", llm_sambanova),
]

# =========================================================
# Flask
# =========================================================
app = Flask(
    __name__,
    template_folder="."
)

# =========================================================
# Home
# =========================================================
@app.route("/")
def index():

    return render_template(
        "template.html",

        crops=CROP_CHOICES,

        models=MODELS,

        default_model=DEFAULT_MODEL
    )

# =========================================================
# Query
# =========================================================
@app.route("/query", methods=["POST"])
def query():

    data = request.json

    q = data.get(
        "query",
        ""
    ).strip()

    crop_filter = data.get(
        "crop",
        "any"
    )

    model_key = data.get(
        "model",
        DEFAULT_MODEL
    )

    if not q:

        return jsonify({
            "error": "Empty query"
        }), 400

    # =====================================================
    # Retrieve
    # =====================================================
    results = retrieve(
        q,
        crop_filter=crop_filter
    )

    if not results:

        return jsonify({
            "results": [],
            "answer": "",
            "model_info": {}
        })

    # =====================================================
    # AUTO MODE
    # =====================================================
    if model_key == "auto":

        llm_result = None

        selected_model = "none"

        for model_name, llm_mod in BEST_MODELS:

            try:

                result = llm_mod.generate_answer(
                    q,
                    results
                )

                if result:

                    llm_result = result

                    selected_model = model_name

                    print(
                        f"[AUTO] Selected: {model_name}"
                    )

                    break

            except Exception as e:

                print(
                    f"[AUTO] {model_name} failed: {e}"
                )

                continue

    # =====================================================
    # MANUAL MODE
    # =====================================================
    else:

        llm_mod = LLM_MAP.get(
            model_key,
            llm_cohere
        )

        llm_result = llm_mod.generate_answer(
            q,
            results
        )

        selected_model = model_key

    # =====================================================
    # Build Retrieval Output
    # =====================================================
    out = []

    for score, doc, meta in results:

        out.append({

            "score":
                normalize_score(score),

            "text":
                doc,

            "crop":
                meta.get(
                    "crop",
                    "general"
                ),

            "filename":
                meta.get(
                    "filename",
                    ""
                ),

            "page":
                meta.get(
                    "page",
                    ""
                ),

            "doc_type":
                meta.get(
                    "doc_type",
                    ""
                ),
        })

    # =====================================================
    # Final Response
    # =====================================================
    return jsonify({

        "results": out,

        "answer":

            llm_result["answer"]

            if llm_result else "",

        "model_info": {

            "selected_model":
                selected_model,

            "name":

                llm_result["model"]

                if llm_result else "",

            "time_s":

                llm_result["time_s"]

                if llm_result else 0,

            "tokens":

                llm_result["tokens"]

                if llm_result else 0,

        } if llm_result else {}
    })

# =========================================================
# Health
# =========================================================
@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "chunks":
            collection.count(),

        "models":
            list(MODELS.keys())
    })

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    print("=" * 60)

    print("  Agricultural Advisory RAG System")

    print(
        f"  Chunks Loaded : {collection.count()}"
    )

    print(
        f"  Models        : {', '.join(MODELS.keys())}"
    )

    print(
        "  URL           : http://127.0.0.1:5000"
    )

    print("=" * 60 + "\n")

    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )