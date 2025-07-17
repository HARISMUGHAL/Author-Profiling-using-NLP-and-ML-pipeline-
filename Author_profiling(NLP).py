import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Step 1: Setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Step 2: Clean Function ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)              # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Step 3: Load Cleaned CSVs ---
df1 = pd.read_csv("new_cleaned_scope1.csv")
df2 = pd.read_csv("new_cleaned_scope2.csv")

# If cleaned_content doesn't exist, create it
if "cleaned_content" not in df1.columns:
    df1["cleaned_content"] = df1["content"].apply(clean_text)
if "cleaned_content" not in df2.columns:
    df2["cleaned_content"] = df2["content"].apply(clean_text)

# --- Step 4: Feature Extraction ---
vectorizer = CountVectorizer(max_features=5000)

x_bow_1 = vectorizer.fit_transform(df1["cleaned_content"])
x_bow_2 = vectorizer.fit_transform(df2["cleaned_content"])

lsa_1 = TruncatedSVD(n_components=min(300, x_bow_1.shape[1]), random_state=42)
lsa_2 = TruncatedSVD(n_components=min(300, x_bow_2.shape[1]), random_state=42)
X_lsa_1 = lsa_1.fit_transform(x_bow_1)
X_lsa_2 = lsa_2.fit_transform(x_bow_2)


print("Scope 1 - BoW:", x_bow_1.shape, "| LSA:", X_lsa_1.shape)
print("Scope 2 - BoW:", x_bow_2.shape, "| LSA:", X_lsa_2.shape)

# --- Step 5: Train & Evaluate ---
def train_lr_on_scope(X, y, scope_name, feature_type):
    print(f"\n📘 Logistic Regression | {scope_name} | {feature_type}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# Train on all combinations
y1 = df1['author']
y2 = df2['author']

train_lr_on_scope(x_bow_1, y1, "Scope 1", "BoW")
train_lr_on_scope(X_lsa_1, y1, "Scope 1", "LSA")
train_lr_on_scope(x_bow_2, y2, "Scope 2", "BoW")
train_lr_on_scope(X_lsa_2, y2, "Scope 2", "LSA")
