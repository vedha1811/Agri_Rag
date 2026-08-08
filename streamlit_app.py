import os
import tarfile
import requests
import streamlit as st


# =========================================================
# Deployment: Load pre-built ChromaDB
# =========================================================

DB_PATH = "agri_db"

if not os.path.exists(DB_PATH):
    st.info("Loading agricultural knowledge base...")

    download_url = (
        "https://github.com/vedha1811/Agri_Rag/"
        "releases/download/v1.0.0/agri_db.tar.gz"
    )

    archive_path = "agri_db.tar.gz"

    response = requests.get(
        download_url,
        stream=True,
        timeout=300
    )
    response.raise_for_status()

    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                f.write(chunk)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(".")

    os.remove(archive_path)

    st.success("Knowledge base loaded.")


from config import CROP_CHOICES, MODELS, DEFAULT_MODEL
from retriever import retrieve, normalize_score, collection

import llm_groq
import llm_cohere
import llm_sarvam
import llm_cerebras
import llm_sambanova


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Kisan Saathi",
    page_icon="🌾",
    layout="wide"
)


# =========================================================
# LLM Mapping
# =========================================================

LLM_MAP = {
    "groq-llama": llm_groq,
    "cohere-command": llm_cohere,
    "sarvam-ai": llm_sarvam,
    "cerebras-llama": llm_cerebras,
    "sambanova-llama": llm_sambanova,
}


BEST_MODELS = [
    ("cohere-command", llm_cohere),
    ("groq-llama", llm_groq),
    ("cerebras-llama", llm_cerebras),
    ("sarvam-ai", llm_sarvam),
    ("sambanova-llama", llm_sambanova),
]


# =========================================================
# Styling
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    .source-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="main-title">🌾 Kisan Saathi</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered agricultural advisory using Retrieval-Augmented Generation'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("⚙️ Advisory Settings")

    crop_filter = st.selectbox(
        "Select Crop",
        CROP_CHOICES,
        index=(
            CROP_CHOICES.index("any")
            if "any" in CROP_CHOICES
            else 0
        )
    )

    model_key = st.selectbox(
        "AI Model",
        list(MODELS.keys()),
        index=(
            list(MODELS.keys()).index(DEFAULT_MODEL)
            if DEFAULT_MODEL in MODELS
            else 0
        ),
        format_func=lambda x: MODELS[x]
    )

    st.divider()

    st.metric(
        "Knowledge Chunks",
        collection.count()
    )

    st.metric(
        "Crop Categories",
        len(CROP_CHOICES)
    )


# =========================================================
# Query Input
# =========================================================

st.subheader("Ask an Agricultural Question")

if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_area(
    "Your question",
    value=st.session_state.query,
    placeholder=(
        "Example: What are the major diseases affecting rice "
        "and how can they be managed?"
    ),
    height=120
)

st.session_state.query = query


# =========================================================
# Quick Questions
# =========================================================

st.write("**Quick Questions**")

quick_questions = [
    "How can I control pests in rice?",
    "What are the major diseases of cotton?",
    "How should maize be managed during kharif?",
    "What are common tomato diseases?",
    "How can I improve groundnut yield?",
    "What are IPM practices for crops?",
]

cols = st.columns(3)

for i, question in enumerate(quick_questions):

    if cols[i % 3].button(
        question,
        key=f"quick_{i}",
        use_container_width=True
    ):
        st.session_state.query = question
        st.session_state.quick_ask = True
        st.rerun()

# =========================================================
# Ask Button
# =========================================================

ask_button = st.button(
    "🔍 Ask Kisan Saathi",
    type="primary",
    use_container_width=True
)

ask = ask_button or st.session_state.get("quick_ask", False)

if st.session_state.get("quick_ask", False):
    st.session_state.quick_ask = False


# =========================================================
# Query Processing
# =========================================================

if ask:

    q = query.strip()

    if not q:

        st.error("Please enter an agricultural question.")

    else:

        with st.spinner(
            "Searching agricultural knowledge base and generating advice..."
        ):

            try:

                # -------------------------------------------------
                # Retrieval
                # -------------------------------------------------

                results = retrieve(
                    q,
                    crop_filter=crop_filter
                )

                if not results:

                    st.warning(
                        "No relevant agricultural information was found."
                    )

                else:

                    # -------------------------------------------------
                    # LLM Selection
                    # -------------------------------------------------

                    llm_result = None
                    selected_model = "none"

                    if model_key == "auto":

                        for model_name, llm_mod in BEST_MODELS:

                            try:

                                result = llm_mod.generate_answer(
                                    q,
                                    results
                                )

                                if result:

                                    llm_result = result
                                    selected_model = model_name

                                    break

                            except Exception as e:

                                print(
                                    f"[AUTO] {model_name} failed: {e}"
                                )

                                continue

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


                    # -------------------------------------------------
                    # Answer
                    # -------------------------------------------------

                    st.subheader("🌱 Agricultural Advisory")

                    if llm_result:

                        st.markdown(
                            f"""
                            <div class="answer-box">
                            {llm_result.get("answer", "")}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.caption(
                            f"Model: "
                            f"{llm_result.get('model', selected_model)}"
                            f"  |  "
                            f"Response time: "
                            f"{llm_result.get('time_s', 0)} s"
                            f"  |  "
                            f"Tokens: "
                            f"{llm_result.get('tokens', 0)}"
                        )

                    else:

                        st.warning(
                            "Unable to generate an answer from the available models."
                        )


                    # -------------------------------------------------
                    # Retrieved Sources
                    # -------------------------------------------------

                    st.subheader("📚 Retrieved Sources")

                    for i, (score, doc, meta) in enumerate(results, 1):

                        filename = meta.get(
                            "filename",
                            ""
                        )

                        page = meta.get(
                            "page",
                            ""
                        )

                        crop = meta.get(
                            "crop",
                            "general"
                        )

                        doc_type = meta.get(
                            "doc_type",
                            ""
                        )

                        relevance = normalize_score(score)

                        with st.expander(
                            f"Source {i} — {filename}"
                        ):

                            st.write(doc)

                            st.caption(
                                f"Crop: {crop} | "
                                f"Document type: {doc_type} | "
                                f"Page: {page} | "
                                f"Relevance: {relevance}"
                            )


            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )

                print(
                    f"[ERROR] {e}"
                )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Kisan Saathi — Agricultural Advisory RAG System"
)
