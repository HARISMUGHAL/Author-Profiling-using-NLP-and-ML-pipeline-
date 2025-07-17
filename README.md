
# 🧠 Author Profiling with BERT and Classical NLP Models

This repository contains two separate pipelines for Author Profiling tasks — one based on **traditional NLP techniques** using Logistic Regression, and the other using **deep learning with BERT**. The goal is to predict author **gender** and **age group** from text.

---

## 📂 Project Structure

```
├── new_cleaned_scope1.csv         # Preprocessed dataset (Scope 1)
├── new_cleaned_scope2.csv         # Preprocessed dataset (Scope 2)
├── pan_sample_author_profiling.csv # PAN-style labeled data (gender, age, text)
├── author_profiling_lr.py         # Logistic Regression pipeline
├── PAN_style_author_profiling_using_bert.py  # BERT training & inference
├── app.py                         # Gradio interface for live predictions
├── bert_author_profiling.pt       # Fine-tuned BERT model
└── README.md
```
## 🔗 Download Trained BERT Model

The pretrained BERT model used for author profiling (gender & age prediction) is available here:

👉 [Download Model](https://drive.google.com/uc?export=download&id=1_F2jnMIDOBdpXh2VREDxsbeRpITG6sP7)

> Make sure to place the downloaded file as `bert_author_profiling.pt` in your project directory before running the prediction script.

---

## ✅ Features

### 1. 🔢 Logistic Regression (BoW + LSA)

- Preprocessing: lowercasing, stopword removal, punctuation removal
- Feature Extraction:
  - Bag of Words (BoW)
  - Latent Semantic Analysis (LSA)
- Trained using `LogisticRegression`
- Evaluates performance with `classification_report`

### 2. 🤖 BERT-Based Author Profiling

- Uses `bert-base-uncased` from Hugging Face
- Fine-tuned on PAN-style dataset with:
  - `gender` (female, male)
  - `age` (18–24, 25–34, 35–49, 50-XX)
- Custom PyTorch model with two classification heads
- Trained on both tasks simultaneously
- Model saved as `bert_author_profiling.pt`

### 3. 🌐 Gradio Web App

- Accepts raw article or paragraph text
- Displays predicted **gender** and **age group**
- Can be launched locally via `gr.Interface(...).launch()`

---

## 🚀 How to Run

### Clone the repo
```bash
git clone https://github.com/yourusername/author-profiling-nlp.git
cd author-profiling-nlp
```

### Create a virtual environment and install dependencies
```bash
pip install -r requirements.txt
```

### Run Logistic Regression pipeline
```bash
python author_profiling_lr.py
```

### Train and Test BERT model
```bash
python PAN_style_author_profiling_using_bert.py
```

### Launch Gradio App
```bash
python app.py
```

---

## 📊 Sample Predictions

| Text | Predicted Gender | Predicted Age |
|------|------------------|----------------|
| "Kal ka paper literally mujhe rula diya..." | female | 18–24 |
| "We migrated our CI/CD pipeline to GitHub Actions..." | male | 35–49 |

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first.

---

## 🧠 License

MIT License
