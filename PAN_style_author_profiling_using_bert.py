import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

# Step 1: Load dataset
df = pd.read_csv("pan_sample_author_profiling.csv")

# TEMP: reduce to 500 for fast training
df = df.sample(500, random_state=42).reset_index(drop=True)

# Encode labels
gender_encoder = LabelEncoder()
age_encoder = LabelEncoder()
df["gender_label"] = gender_encoder.fit_transform(df["gender"])
df["age_label"] = age_encoder.fit_transform(df["age"])

# Split data
train_texts, val_texts, train_genders, val_genders = train_test_split(
    df["text"].tolist(), df["gender_label"].tolist(), test_size=0.2, random_state=42
)
_, _, train_ages, val_ages = train_test_split(
    df["text"].tolist(), df["age_label"].tolist(), test_size=0.2, random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

# Dataset class
class AuthorDataset(Dataset):
    def __init__(self, encodings, gender_labels, age_labels):
        self.encodings = encodings
        self.gender_labels = gender_labels
        self.age_labels = age_labels

    def __len__(self):
        return len(self.gender_labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["gender_labels"] = torch.tensor(self.gender_labels[idx])
        item["age_labels"] = torch.tensor(self.age_labels[idx])
        return item

train_dataset = AuthorDataset(train_encodings, train_genders, train_ages)
val_dataset = AuthorDataset(val_encodings, val_genders, val_ages)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model class
class BertForAuthorProfiling(nn.Module):
    def __init__(self, num_gender_labels=2, num_age_labels=4):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.gender_classifier = nn.Linear(self.bert.config.hidden_size, num_gender_labels)
        self.age_classifier = nn.Linear(self.bert.config.hidden_size, num_age_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        gender_logits = self.gender_classifier(pooled_output)
        age_logits = self.age_classifier(pooled_output)
        return gender_logits, age_logits

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForAuthorProfiling().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

print("Starting training...")
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gender_labels = batch["gender_labels"].to(device)
        age_labels = batch["age_labels"].to(device)

        optimizer.zero_grad()
        gender_logits, age_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss_gender = loss_fn(gender_logits, gender_labels)
        loss_age = loss_fn(age_logits, age_labels)
        loss = loss_gender + loss_age
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
gender_preds, age_preds = [], []
gender_true, age_true = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gender_labels = batch["gender_labels"]
        age_labels = batch["age_labels"]

        gender_logits, age_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        gender_preds.extend(torch.argmax(gender_logits, dim=1).cpu().numpy())
        age_preds.extend(torch.argmax(age_logits, dim=1).cpu().numpy())
        gender_true.extend(gender_labels.numpy())
        age_true.extend(age_labels.numpy())

print("\nGender Classification Report:")
print(classification_report(gender_true, gender_preds, target_names=gender_encoder.classes_))

print("Age Classification Report:")
print(classification_report(age_true, age_preds, target_names=age_encoder.classes_))

# Save model
torch.save(model.state_dict(), "bert_author_profiling.pt")
print("Model saved to: bert_author_profiling.pt")

# Live Prediction Test
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        gender_logits, age_logits = model(**inputs)
    pred_gender = gender_logits.argmax(dim=1).item()
    pred_age = age_logits.argmax(dim=1).item()
    gender = gender_encoder.inverse_transform([pred_gender])[0]
    age = age_encoder.inverse_transform([pred_age])[0]
    return gender, age

# Example
text = "I enjoy programming and science fiction. I'm passionate about startups and technology."
gender, age = predict(text)
print("Predicted Gender:", gender)
print("Predicted Age Group:", age)
