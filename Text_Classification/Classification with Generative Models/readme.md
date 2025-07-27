# 🧠 Classification with Generative Models

This project demonstrates how to use **encoder-decoder generative models** like **Flan-T5** for **sentiment classification** on the [IMDb movie review dataset](https://huggingface.co/datasets/imdb). Instead of traditional discriminative models, we repurpose sequence-to-sequence models for binary classification using prompt-based learning.

## 📌 Key Features

- Uses Hugging Face Transformers pipelines (`text2text-generation`)
- Prompt-based sentiment classification (positive/negative)
- Evaluation using `sklearn` metrics
- Supports GPU acceleration via `device` flag
- Minimal dataset subset for fast experimentation

---

## 🧪 Task Description

Given a movie review text, the model is prompted to classify it as **positive** or **negative**. For example:

```
Prompt: Is the following sentence positive or negative? [text]
```

Expected model output: `positive` or `negative` (or `1` / `0` in the second version)

---

## 🧰 Dependencies

- Python ≥ 3.8  
- [transformers](https://github.com/huggingface/transformers)  
- [datasets](https://github.com/huggingface/datasets)  
- [scikit-learn](https://scikit-learn.org/)  
- [tqdm](https://github.com/tqdm/tqdm)  
- PyTorch (for GPU acceleration)

Install them via:

```bash
pip install transformers datasets scikit-learn tqdm torch
```

---

## 🚀 How to Run

### Option 1: Run as a script (argparse-based)

```bash
python main.py --model google/flan-t5-small --device cuda:0 --prompt "Is the following sentence positive or negative? "
```

This version:
- Loads a generative model
- Prepares the IMDb dataset
- Prompts the model with each text
- Maps generated outputs to binary labels
- Evaluates predictions using `classification_report`

### Option 2: Run notebook-style version

For smaller demos with custom prompt templates:

```bash
python demo.py
```

This version:
- Uses a custom prompt with `[DOCUMENT]` placeholder
- Maps outputs like `"0"` or `"1"` based on keywords
- Designed for interpretability and debugging

---

## 📊 Evaluation Metric

Classification performance is reported using:
- Precision
- Recall
- F1-score

With support for ambiguous outputs being marked as `-1`.

---

## 🧠 Models Used

- [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small)
- [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)

You can replace these with any T5-like model compatible with `text2text-generation`.

---

## 📁 Dataset

- IMDb movie reviews via `datasets.load_dataset("imdb")`
- Subset of 200–1000 samples used for quick inference and testing

---

## ⚠️ Notes

- Generated outputs may be ambiguous; edge cases are handled with warnings
- Larger models (e.g., `flan-t5-xl`) will produce better results but require more memory
- This is a demonstration of **prompt-based classification** — not fine-tuning

---

## 📄 License

MIT License
