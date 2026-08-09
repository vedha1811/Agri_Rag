# 🌾 Kisan Saathi — Adaptive Multi-LLM RAG for Agricultural Advisory

**Kisan Saathi** is an AI-powered agricultural advisory system built using **Retrieval-Augmented Generation (RAG)** to provide context-aware and practical responses to agricultural queries.

The system combines **semantic retrieval, cross-encoder reranking, domain-specific agricultural knowledge, and multiple Large Language Models (LLMs)** to generate agricultural recommendations grounded in retrieved source documents.

### 🚀 Live Demo

👉 **[Kisan Saathi — Live Streamlit App](https://agrirag-4v7elgsv9mpjakbkcgrtj4.streamlit.app/)**

📂 **[GitHub Repository](https://github.com/vedha1811/Agri_Rag)**

---

## 📌 Overview

Kisan Saathi processes agricultural documents into a searchable domain-specific knowledge base and uses a RAG pipeline to answer agricultural questions.

The system follows the complete pipeline:

1. Processes agricultural documents and scanned content.
2. Performs semantic document chunking.
3. Generates dense vector embeddings using Sentence Transformers.
4. Stores embeddings and metadata in ChromaDB.
5. Retrieves relevant agricultural passages for a user query.
6. Reranks retrieved passages using a cross-encoder.
7. Provides the highest-ranked context to multiple LLM providers.
8. Supports adaptive / automatic model selection.
9. Generates a context-aware agricultural advisory.
10. Displays the retrieved source documents used for the response.

### Knowledge Base

- **13,833 indexed knowledge chunks**
- **42 crop categories**
- Domain-specific agricultural documents
- Source-aware retrieval and response generation

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
- 📊 Automated multi-model evaluation
- 🚀 Streamlit Community Cloud deployment

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
```

---

## 🤖 Supported LLMs

The deployed system currently integrates four LLM providers:

| Provider | Model |
|---|---|
| Groq | LLaMA 3.1 8B |
| Cohere | Command-R |
| Sarvam AI | Sarvam-105B |
| Cerebras | GPT-OSS 120B |

The application also provides a **Best Available Model (Auto)** option for adaptive model selection.

> **Note:** SambaNova was part of an earlier implementation but is not part of the final deployed model selection.

---

## 🔍 RAG Pipeline

### 1. Document Processing

Agricultural documents are collected and processed using PDF extraction and OCR techniques where required.

### 2. Semantic Chunking

Documents are divided into meaningful passages while preserving relevant agricultural context.

### 3. Embedding Generation

The system uses the following Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

The model converts agricultural passages into dense semantic vector representations.

### 4. Vector Storage

Embeddings and document metadata are stored in:

```text
ChromaDB
```

This provides the searchable vector knowledge base used during retrieval.

### 5. Semantic Retrieval

For each agricultural query, semantically relevant passages are retrieved from the knowledge base.

### 6. Cross-Encoder Reranking

Retrieved passages are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The highest-ranked passages are selected as the final context provided to the LLM.

### 7. Context-Aware Response Generation

The selected LLM receives the retrieved agricultural context and generates a practical response grounded in the available source material.

---

## 📊 Evaluation

The system was evaluated using **48 reference agricultural questions** across the supported LLMs.

### Evaluation Metrics

The evaluation considers:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU-1
- BLEU-2
- BLEU-4
- METEOR
- Relevance
- Completeness
- Practicality
- Combined Score
- Average Response Time

### Final Model Comparison

| Model | Successful | ROUGE-1 | ROUGE-L | BLEU-4 | METEOR | Heuristic | Combined | Avg. Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Cohere — Command-R** | 48/48 | 0.3848 | 0.2688 | 0.1152 | 0.4174 | **0.9538** | **0.6252** | 32.49s |
| **Groq — LLaMA 3.1 8B** | 48/48 | **0.4415** | **0.3417** | **0.1524** | 0.4146 | 0.8615 | 0.5995 | **0.75s** |
| **Sarvam-105B** | 48/48 | 0.3389 | 0.2599 | 0.1068 | 0.2813 | 0.6736 | 0.4602 | 1.25s |
| **Cerebras — GPT-OSS 120B** | 47/48 | 0.2889 | 0.1908 | 0.0310 | 0.1734 | 0.7000 | 0.4355 | 2.13s |

### 🏆 Key Results

**Best Overall Model:** Cohere — Command-R  
Combined Score: **0.6252**

**Fastest Model:** Groq — LLaMA 3.1 8B  
Average Response Time: **0.75 seconds**

**Best ROUGE-1:** Groq — LLaMA 3.1 8B  
ROUGE-1: **0.4415**

The evaluation demonstrates a clear trade-off between **response quality and generation latency** across the integrated LLM providers.

Detailed evaluation results are available in:

- `evaluation_results_full.csv`
- `rouge_bleu_summary.csv`
- `evaluation_report_full.txt`

---

## 🛠️ Technologies Used

### Core

- Python
- Retrieval-Augmented Generation (RAG)
- Streamlit
- ChromaDB

### Retrieval

- Sentence Transformers
- Dense Vector Embeddings
- Cross-Encoder Reranking

### LLM APIs

- Groq API
- Cohere API
- Sarvam AI API
- Cerebras API

### Document Processing

- PyMuPDF
- Tesseract OCR
- pdf2image
- Pillow
- LangChain Text Splitters

### Evaluation

- ROUGE
- BLEU
- METEOR
- Relevance
- Completeness
- Practicality

---

## 📁 Project Structure

```text
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
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vedha1811/Agri_Rag.git
cd Agri_Rag
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

The application uses environment variables to securely store API credentials.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key
CO_API_KEY=your_key
SARVAM_API_KEY=your_key
CEREBRAS_API_KEY=your_key
```

Replace the placeholder values with your own API keys.

### Security

**Never commit `.env` or API keys to GitHub.**

The `.gitignore` configuration excludes sensitive credentials from version control.

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application provides:

- Crop selection
- AI model selection
- Best Available Model / Auto selection
- Agricultural question input
- Semantic retrieval
- Cross-encoder reranking
- Context-aware agricultural advisory
- Retrieved source display

---

## 🗄️ Knowledge Base

The project uses a domain-specific agricultural knowledge base containing processed agricultural documents.

The generated ChromaDB database is intentionally excluded from Git because of its size:

```text
agri_db/
```

The source agricultural documents are also excluded from the public repository where applicable.

The deployed application loads the pre-built knowledge base separately.

---

## 📚 Evaluation Dataset

The evaluation dataset contains **48 agricultural reference questions** covering different crops and agricultural question types.

The evaluation pipeline compares responses generated by the supported LLMs using both:

- Lexical and text-generation metrics
- Heuristic agricultural response-quality metrics

This enables comparison of both **linguistic similarity** and **practical response quality**.

---

## 🔒 Security

API credentials are handled through environment variables and are not stored in the source code.

The following are excluded from version control:

```text
.env
agri_db/
agri_db.tar.gz
__pycache__/
.DS_Store
```

No API credentials are intentionally stored in the repository.

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

👉 **[Launch Kisan Saathi](https://agrirag-4v7elgsv9mpjakbkcgrtj4.streamlit.app/)**

The deployed application provides real-time agricultural advisory using the integrated LLM providers and the domain-specific agricultural knowledge base.

---

## 🎓 Academic Project

Kisan Saathi was developed as an academic project demonstrating the practical application of:

- Retrieval-Augmented Generation
- Semantic Information Retrieval
- Cross-Encoder Reranking
- Multi-LLM Integration
- Adaptive Model Selection
- Domain-Specific Knowledge Grounding
- Automated LLM Evaluation

The project focuses on building a practical agricultural advisory system where generated responses are grounded in retrieved agricultural source documents rather than relying solely on the underlying language model.

---

## ⭐ Project Highlights

- **13,833** indexed agricultural knowledge chunks
- **42** crop categories
- **48** reference evaluation questions
- **4** integrated LLM providers
- Semantic retrieval + cross-encoder reranking
- Adaptive / Auto LLM selection
- Automated multi-model evaluation
- Source-grounded agricultural recommendations
- Deployed using Streamlit Community Cloud

---

## 👩‍💻 Author

### Vedha Smitha Murari

GitHub:  
https://github.com/vedha1811

---

If you find this project useful, consider giving the repository a ⭐.
