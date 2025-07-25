# 🧠 Sentiment Analysis on IMDB Reviews with Transformers

This repository demonstrates how to perform sentiment analysis on IMDB movie reviews using various Hugging Face transformer models. It includes:

* Fine-tuned transformer-based classification
* A live web app for sentiment prediction
* Model comparison
* Text summarization for handling long sequences

---

## 📂 Project Structure

The code is organized into **four main parts**:

---

### **📌 Part 1: Sentiment Classification with Transformers**

In this part, we load the IMDB dataset and apply a pre-trained transformer model `cardiffnlp/twitter-roberta-base-sentiment-latest` to classify reviews as positive or negative.

#### ✅ Features:

* Uses Hugging Face `pipeline` for inference
* Computes classification metrics using `sklearn`
* Handles full dataset in batch mode using `KeyDataset`

---

### **📌 Part 2: Streamlit Sentiment Analysis Web App**

This part builds an interactive Streamlit app that lets users input text and get real-time sentiment predictions.

#### ✅ Features:

* Uses `cardiffnlp/twitter-roberta-base-sentiment-latest`
* Simple and interactive UI for user input
* Displays prediction scores and sentiment class
* Can be deployed via `ngrok` (for sharing the app in Colab)

---

### **📌 Part 3: Comparing Models**

We experiment with two other sentiment classification models:

1. `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
2. `nlptown/bert-base-multilingual-uncased-sentiment`

Each model is evaluated on the IMDB test set using the same metrics, and results are compared for accuracy and robustness.

#### ✅ Highlights:

* `nlptown` model outputs star ratings (1-5), which are mapped to binary classes
* Filtering of ambiguous/neutral labels for a cleaner evaluation
* Handles both English and multilingual capabilities

---

### **📌 Part 4: Handling Long Texts with Summarization**

Some reviews in the IMDB dataset exceed the model’s input limit of 512 tokens. This part:

* Analyzes token length distribution
* Applies summarization using models like `google/pegasus-xsum` and `t5-small`
* Adds a new `processed_text` column with summarized content
* Re-evaluates sentiment classification using summarized texts

---

## 🧪 Models Used

| Model Name                                                   | Purpose                     | Type       |
| ------------------------------------------------------------ | --------------------------- | ---------- |
| `cardiffnlp/twitter-roberta-base-sentiment-latest`           | Sentiment classification    | RoBERTa    |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Binary sentiment            | DistilBERT |
| `nlptown/bert-base-multilingual-uncased-sentiment`           | Star rating-based sentiment | BERT       |
| `google/pegasus-xsum`                                        | Summarization               | PEGASUS    |
| `t5-small`                                                   | Summarization               | T5         |

---

## 🧰 Installation

Install all required packages:

```bash
pip install transformers datasets streamlit torch pyngrok sentence-transformers
```

---

## 🚀 Running the App

To launch the sentiment analysis app via Streamlit (e.g., in Colab):

```python
!streamlit run sentiment_app.py &
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```
<img width="991" height="725" alt="image" src="https://github.com/user-attachments/assets/f2350606-0cd0-4dbd-8518-c895af38b8e3" />

---

## 📊 Evaluation Metrics

All models are evaluated using:

* **Accuracy**
* **Precision / Recall / F1-Score**
* **Support for binary classification**

The `classification_report` from `sklearn` provides a detailed breakdown of performance.

---

## 📎 Dataset

* **Name**: IMDB Movie Reviews
* **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/imdb)
* **Split**: `train` / `test` (binary labels: 0 = negative, 1 = positive)

---

## 📈 Example Outputs

```bash
              precision    recall  f1-score   support

Negative Review       0.88      0.90      0.89      12500
Positive Review       0.90      0.88      0.89      12500
```

---

## 📌 Future Improvements

* Add model selection UI in Streamlit app
* Support for multi-language input
* Visualization of sentiment trends over time

---

## 🤝 Contribution

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License

This project is under the **MIT License**.

