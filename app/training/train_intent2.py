import json
import os
import re
import pickle

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# ---------------- LOAD DATA ---------------- #
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
data = ds["train"]

# ---------------- CLEAN ---------------- #
def clean_text(text):
    return re.sub(r"\{\{.*?\}\}", "", text).strip().lower()

texts = []
labels = []

for item in data:
    text = clean_text(item["instruction"])
    intent = item["intent"]

    if len(text.split()) < 3:
        continue

    texts.append(text)
    labels.append(intent)

# ---------------- SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ---------------- VECTORIZER ---------------- #
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- MODEL ---------------- #
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------- EVALUATE ---------------- #
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------- SAVE ---------------- #
os.makedirs("intent_model", exist_ok=True)

with open("intent_model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("intent_model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Ultra-fast model trained!")


import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------- EVALUATE ---------------- #
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1)
}

print("📊 Metrics:", metrics)

# ---------------- SAVE METRICS ---------------- #
with open("intent_model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("intent_model/confusion_matrix.png")    