# 📚 LLaMA RAG Demo with Phi-3 and FAISS

This project is a simple demonstration of **Retrieval-Augmented Generation (RAG)** using the **Phi-3-mini** model and the **LangChain** framework.  
It processes a literary text (*The Gift of the Magi*) and answers questions based solely on the text using **FAISS** for semantic search and **LLaMA CPP** for local inference.

---

## 🚀 Features
- Fetch and preprocess text from **Project Gutenberg**
- Split the text into **chunks** for better retrieval
- Build **embeddings** with the `BAAI/bge-small-en-v1.5` model
- Use **FAISS** for vector-based search
- Run the **Phi-3-mini-4k-instruct** model locally with `llama-cpp-python`
- Answer questions strictly based on the provided text (no external knowledge)

---

## 📦 Installation
First, install the required dependencies:

```bash
pip install langchain==0.2.5 faiss-cpu==1.8.0 sentence-transformers==3.0.1 clean-text[emoji] nltk llama-cpp-python==0.2.78 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install -U langchain-community langchain-text-splitters
```

---

## 📥 Download the Model
Download the **Phi-3-mini-4k-instruct-q4.gguf** model from Hugging Face:

```bash
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

## 📊 Example Output
```
[Question 1] What did Della sell and why?
✅ Answer: She sold her hair to buy a gift for Jim.

[Question 4] How much money did Della have at the beginning?
✅ Answer: One dollar and eighty-seven cents.
```

---

## ⚠️ Notes
- For GPU acceleration, ensure CUDA 12.4 is installed.
- The model size is ~2.3GB, so make sure you have enough disk space.
- If performance is slow, you can use a smaller **Q4** or **Q2** version of the model.

