Retrieval-Augmented Generation (RAG) with SQuAD v2
This project implements a Retrieval-Augmented Generation (RAG) pipeline using the SQuAD v2 dataset and LangChain, integrating document preprocessing, embedding generation, vector storage with FAISS, and question-answering via a local LLaMA-based model.
It also includes evaluation metrics such as F1 score, Exact Match (EM), Recall@K, and semantic similarity.

🚀 Features
Dataset Loading: Automatically downloads and loads SQuAD v2 via datasets.

Document Preprocessing:

Text cleaning (removes URLs, emails, phone numbers, special characters).

Optional summarization using BART-Large-CNN if context exceeds a word threshold.

Vector Store:

Embedding generation with BAAI/bge-small-en-v1.5.

Storage and retrieval via FAISS.

Local LLaMA Model:

Runs Phi-3-mini-4k-instruct-q4.gguf with llama-cpp-python.

Custom prompt templates for concise answers.

Evaluation Metrics:

SQuAD F1 & Exact Match scores.

Recall@K retrieval evaluation.

Semantic similarity with SentenceTransformers.
