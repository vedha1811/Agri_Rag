# Adaptive Context-Aware Multi-LLM RAG Framework for Agricultural Advisory

An intelligent agricultural advisory system based on Retrieval-Augmented Generation (RAG), semantic retrieval, cross-encoder reranking, and adaptive multi-LLM response generation.

## Overview

This project implements a Retrieval-Augmented Generation framework for answering agricultural queries using information retrieved from a domain-specific agricultural knowledge base.

The system processes agricultural documents, divides them into meaningful chunks, generates semantic embeddings, stores them in ChromaDB, retrieves relevant passages for a user query, and reranks the retrieved results using a cross-encoder.

The retrieved context is then provided to multiple Large Language Model (LLM) providers. The framework supports adaptive routing across different LLMs to generate context-aware and practical agricultural responses.

## Key Features

- Retrieval-Augmented Generation (RAG) for agricultural advisory
- Agricultural document processing and semantic chunking
- OCR-based processing for scanned documents
- Dense semantic embeddings using Sentence Transformers
- ChromaDB-based vector retrieval
- Cross-encoder reranking
- Multi-LLM integration
- Adaptive LLM routing
- Support for Cohere, Groq, Sarvam, Cerebras, and SambaNova
- Flask-based web application
- Automated response evaluation
- Evaluation using ROUGE, BLEU, METEOR, relevance, completeness, and practicality metrics

## System Architecture

```text
Agricultural Documents
        |
        v
Document Processing / OCR
        |
        v
Semantic Chunking
        |
        v
Sentence Transformer Embeddings
        |
        v
ChromaDB Vector Database
        |
        v
User Query
        |
        v
Semantic Retrieval
        |
        v
Cross-Encoder Reranking
        |
        v
Adaptive Multi-LLM Routing
        |
        +---- Cohere
        |
        +---- Groq
        |
        +---- Sarvam
        |
        +---- Cerebras
        |
        +---- SambaNova
        |
        v
Context-Aware Agricultural Response

Technologies Used
Python
Flask
Retrieval-Augmented Generation (RAG)
ChromaDB
Sentence Transformers
Cross-Encoder Reranking
LangChain Text Splitters
Cohere API
Groq API
Sarvam API
Cerebras API
SambaNova API
PyMuPDF
Tesseract OCR
pdf2image
Pillow

Project Structure

Agri_Rag/
|
+-- app.py
+-- config.py
+-- retriever.py
+-- llm.py
|
+-- llm_cohere.py
+-- llm_groq.py
+-- llm_sarvam.py
+-- llm_cerebras.py
+-- llm_sambanova.py
|
+-- 01_ingest.py
+-- 02_query.py
+-- 03_rag_answer.py
+-- ocr_ingest.py
+-- dataset_generator.py
+-- Evaluate.py
|
+-- template.html
+-- images/
|
+-- evaluation_results.csv
+-- evaluation_results_full.csv
+-- evaluation_report.txt
+-- evaluation_report_full.txt
+-- rouge_bleu_summary.csv
|
+-- reference_dataset.csv
+-- reference_dataset.json
+-- methodology.docx
|
+-- requirements.txt
+-- .gitignore
+-- README.md

Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Agri_Rag

Create a virtual environment:

python3 -m venv .venv

Activate the environment:

source .venv/bin/activate

Install the required Python packages:

pip install -r requirements.txt
Environment Variables

The application uses environment variables to store API credentials.

Create a .env file in the project root:

GROQ_API_KEY=your_key
CO_API_KEY=your_key
SARVAM_API_KEY=your_key
CEREBRAS_API_KEY=your_key
SAMBANOVA_API_KEY=your_key

Replace the values with your own API keys.

Never upload the .env file to GitHub.

The .env file is excluded from version control through .gitignore.

Running the Application

Start the Flask application:

python3 app.py

The application will be available at:

http://127.0.0.1:5000

Open the address in your web browser.

Knowledge Base

The system uses a domain-specific agricultural document collection as its knowledge source.

The documents are processed through the ingestion pipeline, converted into semantic embeddings, and stored in a ChromaDB vector database.

The following directories are intentionally excluded from the GitHub repository because of their size:

Dataset/
agri_db/

The Dataset/ directory contains the source agricultural documents, while agri_db/ contains the generated vector database.

Evaluation

The project includes evaluation results for the generated agricultural responses.

The evaluation considers multiple response-quality and text-generation metrics, including:

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
Combined evaluation score

Detailed evaluation results are available in:

evaluation_results_full.csv
evaluation_report_full.txt
Security

API credentials are handled through environment variables and are not included in the repository.

The following files and directories are intentionally excluded from version control:

.env
Dataset/
agri_db/
__pycache__/
.DS_Store
Academic Project

This project was developed as an academic implementation of an intelligent agricultural advisory system using Retrieval-Augmented Generation, semantic retrieval, reranking, and multi-LLM integration.
