import gradio as gr
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PAN_style_author_profiling_using_bert import BertForAuthorProfiling  # You must have this class copied from training file

# Load tokenizer and device
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoders
df = pd.read_csv("pan_sample_author_profiling.csv")
gender_encoder = LabelEncoder()
age_encoder = LabelEncoder()
df["gender_label"] = gender_encoder.fit_transform(df["gender"])
df["age_label"] = age_encoder.fit_transform(df["age"])

# Load model
model = BertForAuthorProfiling()
model.load_state_dict(torch.load("bert_author_profiling.pt", map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict_author_profile(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

    with torch.no_grad():
        gender_logits, age_logits = model(**inputs)

    gender_id = gender_logits.argmax(dim=1).item()
    age_id = age_logits.argmax(dim=1).item()

    gender = gender_encoder.inverse_transform([gender_id])[0]
    age = age_encoder.inverse_transform([age_id])[0]

    return gender, age

# Gradio Interface
gr.Interface(
    fn=predict_author_profile,
    inputs=gr.Textbox(lines=4, placeholder="Paste any article or paragraph..."),
    outputs=["text", "text"],
    title="Author Profiling (BERT)",
    description="Predicts gender and age group from writing style using a fine-tuned BERT model."
).launch()
