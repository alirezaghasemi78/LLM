# 🧠 Text Classification with Sentence-BERT Embeddings

This project demonstrates several approaches for **binary text classification** (positive/negative reviews) using **sentence embeddings** from [Sentence-BERT](https://www.sbert.net/). It compares traditional classifiers, similarity-based methods, and zero-shot inference using natural language prompts.

---

## 📌 Overview

We experiment with the following pipelines:

1. **Supervised Classification**
   - Embeddings via `all-mpnet-base-v2`
   - Classifiers: Logistic Regression, Random Forest, SVM, Naive Bayes

2. **Similarity-based Classification (Centroid-based)**
   - Average embeddings per class
   - Use cosine similarity to classify unseen documents

3. **Zero-shot Classification**
   - Encode prompts like `"A negative review"` vs `"A positive review"`
   - Match documents with label embeddings via cosine similarity

4. **Zero-shot with Multiple Prompts + Ensemble**
   - Try multiple natural language prompts per label
   - Aggregate predictions via mean similarity (soft voting)

---

## 🧪 Sample Results

| Method                      | Accuracy | F1 Score |
|----------------------------|----------|----------|
| Logistic Regression        | 0.85     | 0.84     |
| Random Forest              | 0.82     | 0.81     |
| SVM (Linear)               | 0.83     | 0.83     |
| Naive Bayes                | 0.77     | 0.75     |
| Centroid (no classifier)   | 0.80     | 0.79     |
| Zero-shot (1 prompt)       | 0.70     | 0.69     |
| Zero-shot (3 prompts avg.) | 0.76     | 0.75     |

> 📌 Note: The actual performance depends on the dataset used. The example above assumes a standard sentiment analysis dataset like SST2 or IMDB.

---

## 🛠 How It Works

- **Embedding Generation**  
  We use `sentence-transformers/all-mpnet-base-v2` to embed text documents.

- **Supervised Models**  
  Fit classifiers directly on embedding vectors.

- **Centroid Method**  
  Average embeddings for each class and use similarity.

- **Zero-shot**  
  Embed label prompts and use cosine similarity without training.

- **Prompt Engineering**  
  Try different prompt phrasings to see how language affects zero-shot classification.

---

## 📦 Dependencies

Install required libraries with:

```bash
pip install -r requirements.txt
