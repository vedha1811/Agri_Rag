# 🌾 Kisan Saathi — Adaptive Multi-LLM RAG for Agricultural Advisory

**Kisan Saathi** is an AI-powered agricultural advisory system built using **Retrieval-Augmented Generation (RAG)** to provide context-aware and practical responses to agricultural queries.

The system combines **semantic retrieval, cross-encoder reranking, domain-specific agricultural knowledge, and multiple Large Language Models (LLMs)** to generate reliable agricultural recommendations grounded in retrieved source documents.

### 🚀 Live Demo

**[Kisan Saathi — Streamlit App](https://agrirag-4v7elgsv9mpjakbckgrtj4.streamlit.app/)**

---

## 📌 Overview

Kisan Saathi processes agricultural documents into a searchable domain-specific knowledge base and uses a RAG pipeline to answer farmer queries.

The system:

1. Processes agricultural documents and scanned content.
2. Performs semantic chunking of the documents.
3. Generates dense vector embeddings using Sentence Transformers.
4. Stores the embeddings in ChromaDB.
5. Retrieves relevant agricultural passages for a user query.
6. Reranks retrieved passages using a cross-encoder.
7. Provides the retrieved context to multiple LLM providers.
8. Supports adaptive model selection for response generation.
9. Displays the generated agricultural advisory along with retrieved sources.

The final knowledge base contains:

- **13,833 indexed knowledge chunks**
- **42 crop categories**
- Domain-specific agricultural documents

---

## ✨ Key Features

- 🔎 Retrieval-Augmented Generation (RAG)
- 🌱 Domain-specific agricultural knowledge base
- 📄 Agricultural document processing
- 🖨️ OCR-assisted processing for scanned documents
- 🧩 Semantic document chunking
- 🧠 Dense semantic embeddings
- 🗄️ ChromaDB vector database
- 🔄 Cross-encoder reranking
- 🤖 Multi-LLM response generation
- ⚡ Adaptive / Auto model selection
- 📚 Retrieved source attribution
- 📊 Automated response evaluation
- 🚀 Streamlit-based deployment

---

## 🏗️ System Architecture

```text
                Agricultural Documents
                         │
                         ▼
             Document Processing / OCR
                         │
                         ▼
                 Semantic Chunking
                         │
                         ▼
             Sentence Transformer
                   Embeddings
                         │
                         ▼
                   ChromaDB
              Vector Knowledge Base
                         │
                         │
                  User Question
                         │
                         ▼
                Query Embedding
                         │
                         ▼
               Semantic Retrieval
                         │
                         ▼
              Cross-Encoder Reranking
                         │
                         ▼
                Top-K Context
                         │
                         ▼
              Adaptive LLM Routing
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
        Groq          Cohere        Sarvam-105B
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                Cerebras GPT-OSS
                     120B
                         │
                         ▼
             Context-Aware Agricultural
                    Advisory
                         │
                         ▼
                Retrieved Sources

🤖 Supported LLMs

The deployed system currently integrates four LLM providers:

Provider	Model
Groq	LLaMA 3.1 8B
Cohere	Command-R
Sarvam AI	Sarvam-105B
Cerebras	GPT-OSS 120B

The application also provides an Auto / Best Available Model option for adaptive model selection.

🔍 RAG Pipeline
1. Document Processing

Agricultural documents are collected and processed using PDF extraction and OCR techniques where required.

2. Semantic Chunking

Documents are divided into meaningful passages while preserving agricultural context.

3. Embedding Generation

The system uses:

all-MiniLM-L6-v2

to generate dense semantic embeddings.

4. Vector Storage

Embeddings and document metadata are stored in:

ChromaDB
5. Semantic Retrieval

For each user query, relevant passages are retrieved from the agricultural knowledge base.

6. Cross-Encoder Reranking

Retrieved passages are reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

The highest-ranked passages are passed to the selected LLM.

7. Response Generation

The LLM receives the retrieved agricultural context and generates a practical response constrained by the available knowledge.

📊 Evaluation

The system was evaluated using 48 reference agricultural questions across the supported LLMs.

Evaluation metrics include:

ROUGE-1
ROUGE-2
ROUGE-L
BLEU-1
BLEU-2
BLEU-4
METEOR
Relevance
Completeness
Practicality
Combined Score
Average Response Time
Final Model Comparison
Model	Successful	R-1	R-L	BLEU-4	METEOR	Heuristic	Combined	Avg. Time
Cohere — Command-R	48/48	0.3848	0.2688	0.1152	0.4174	0.9538	0.6252	32.49s
Groq — LLaMA 3.1 8B	48/48	0.4415	0.3417	0.1524	0.4146	0.8615	0.5995	0.75s
Sarvam-105B	48/48	0.3389	0.2599	0.1068	0.2813	0.6736	0.4602	1.25s
Cerebras — GPT-OSS 120B	47/48	0.2889	0.1908	0.0310	0.1734	0.7000	0.4355	2.13s
Key Results

🏆 Best Overall: Cohere — Command-R
Combined Score: 0.6252

⚡ Fastest Model: Groq — LLaMA 3.1 8B
Average Response Time: 0.75 seconds

📊 Best ROUGE-1: Groq — LLaMA 3.1 8B
ROUGE-1: 0.4415

The evaluation demonstrates a trade-off between response quality and generation latency across the different LLM providers.

Detailed evaluation results are available in:

evaluation_results_full.csv
rouge_bleu_summary.csv
evaluation_report_full.txt
🛠️ Technologies Used
Core
Python
Retrieval-Augmented Generation (RAG)
Streamlit
ChromaDB
Retrieval
Sentence Transformers
Cross-Encoder Reranking
Dense Vector Embeddings
LLM APIs
Groq API
Cohere API
Sarvam AI API
Cerebras API
Document Processing
PyMuPDF
Tesseract OCR
pdf2image
Pillow
LangChain Text Splitters
Evaluation
ROUGE
BLEU
METEOR
Relevance
Completeness
Practicality
📁 Project Structure
Agri_Rag/
│
├── streamlit_app.py
├── retriever.py
├── config.py
│
├── llm_groq.py
├── llm_cohere.py
├── llm_sarvam.py
├── llm_cerebras.py
│
├── 01_ingest.py
├── 02_query.py
├── 03_rag_answer.py
├── ocr_ingest.py
├── dataset_generator.py
│
├── Evaluate.py
│
├── evaluation_results.csv
├── evaluation_results_full.csv
├── evaluation_report_full.txt
├── rouge_bleu_summary.csv
│
├── reference_dataset.csv
├── reference_dataset.json
│
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/vedha1811/Agri_Rag.git
cd Agri_Rag

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
🔐 Environment Variables

API credentials are stored using environment variables.

Create a .env file:

GROQ_API_KEY=your_key
CO_API_KEY=your_key
SARVAM_API_KEY=your_key
CEREBRAS_API_KEY=your_key

Never commit .env or API keys to GitHub.

The .gitignore configuration excludes sensitive credentials.

▶️ Running the Application

Start the Streamlit application:

streamlit run streamlit_app.py

The application provides:

Crop selection
AI model selection
Auto model selection
Agricultural question input
Retrieved source display
Context-aware agricultural advisory
🗄️ Knowledge Base

The project uses a domain-specific agricultural knowledge base.

The generated ChromaDB database is intentionally excluded from Git because of its size.

agri_db/

The deployed application loads the pre-built knowledge base separately.

📚 Evaluation Dataset

The evaluation dataset contains 48 agricultural reference questions covering different crops and question types.

The evaluation pipeline compares responses from the supported LLMs using both:

lexical/text-generation metrics
heuristic agricultural response-quality metrics
🔒 Security

API credentials are handled through environment variables.

The following are excluded from version control:

.env
agri_db/
agri_db.tar.gz
__pycache__/
.DS_Store

No API credentials are stored in the source code.

🚀 Deployment

The application is deployed using Streamlit Community Cloud.

Live Application

Kisan Saathi — Live Demo

The deployed application loads the pre-built ChromaDB knowledge base and provides real-time agricultural advisory through the integrated LLM providers.

🎓 Academic Project

Kisan Saathi was developed as an academic project demonstrating the application of:

Retrieval-Augmented Generation
Semantic information retrieval
Cross-encoder reranking
Multi-LLM integration
Adaptive model selection
Domain-specific knowledge grounding
Automated LLM evaluation

The project focuses on building a practical agricultural advisory system where generated responses are grounded in retrieved agricultural source documents.

👩‍💻 Author

Vedha Smitha Murari

GitHub:
https://github.com/vedha1811


### One important correction

Your old README says:

> `Support for Cohere, Groq, Sarvam, Cerebras, and SambaNova`

**Don't keep that statement.** Your final deployed system has four active models. Although `llm_sambanova.py` still exists in the repository, SambaNova is no longer part of the deployed model selection after your final changes.

Also change the old:

> `Start the Flask application: python3 app.py`

to the Streamlit command:

```bash
streamlit run streamlit_app.py
