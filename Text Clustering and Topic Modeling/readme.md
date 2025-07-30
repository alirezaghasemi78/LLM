# 🗞️ AG News Topic Modeling and Text Clustering

This project explores **unsupervised topic modeling and clustering** on the AG News dataset using **SentenceTransformers**, **UMAP**, **HDBSCAN**, and **BERTopic**. We further enhance topic representations with models like **KeyBERT**, **Maximal Marginal Relevance**, **Flan-T5**, and **Gemini-Pro** (Google Generative AI).

---

## 📚 Dataset

We use the [AG News dataset] from HuggingFace 🤗. It contains:

- 120,000 training samples
- 7,600 test samples  
- Each sample includes a **title** and a **short description**
- Four categories: `World`, `Sports`, `Business`, `Sci/Tech`

---

## 🚀 Project Pipeline

### 1. Embedding the Text

- We use `thenlper/gte-small` from `SentenceTransformers` to embed news abstracts into 384-dimensional vectors.

### 2. Dimensionality Reduction

- We apply [UMAP](https://umap-learn.readthedocs.io/) to reduce embeddings first to 5D for clustering and later to 2D for visualization.

### 3. Clustering

- We use [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) to find natural groupings in the reduced embedding space.
- Outliers (noise) are labeled as `-1`.

### 4. Topic Modeling with BERTopic

- We use [BERTopic](https://maartengr.github.io/BERTopic/) with custom embedding, UMAP, and HDBSCAN models.
- Visualizations include:
  - 📊 **Barcharts**
  - 🔥 **Heatmaps**
  - 🌲 **Hierarchical Topic Trees**
  - 🗺️ **Interactive Document Maps**

---

## 🧠 Enhancing Topic Representations

We experiment with multiple `representation_model`s to improve topic interpretability:

### ✅ KeyBERTInspired
Re-ranks topic words using cosine similarity to topic centroid.

### ✅ Maximal Marginal Relevance (MMR)
Balances relevance and diversity using a tunable `diversity` parameter.

### ✅ Flan-T5
Uses a text-to-text generation model (`google/flan-t5-small`) with a custom prompt to produce human-readable topic descriptions.

### ✅ Gemini-Pro (Google Generative AI)
Generates a **single descriptive sentence** for each topic using a custom prompt and the `gemini-pro` model.

---

## 🧪 Visualizations

- Scatter plot of 2D clusters using Matplotlib
- Interactive topic-document map with BERTopic
- Keyword barcharts and heatmaps
- Hierarchical structure of topics

---

## 🌥 Bonus: Word Clouds

Generate word clouds for individual topics with top `n` keywords.

```python
create_wordcloud(topic_model, topic=17)

