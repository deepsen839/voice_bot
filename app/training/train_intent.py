import json
import os
import numpy as np
import pickle
import re
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import torch


# ---------------- CONFIG ---------------- #
MODEL_NAME = "distilbert-base-uncased"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "intent_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- LOAD CSV ---------------- #
df = pd.read_csv("Customer_support_data.csv")
print(df.head())
print("Columns:", df.columns)
df = df.dropna(subset=["Customer Remarks", "category"])

# ---------------- LABEL MAP ---------------- #
label_map = {
    "order_status": 0,
    "cancel_order": 1,
    "refund": 2,
    "subscription": 3,
    "technical_issue": 4,
    "payment_issue": 5,
    "greeting": 6,
    "goodbye": 7,
    "complaint": 8,
    "other": 9
}

id2label = {v: k for k, v in label_map.items()}


# ---------------- MAP LABEL ---------------- #
def map_label(label):
    label = str(label).lower()

    if "cancel" in label:
        return "cancel_order"
    elif "refund" in label:
        return "refund"
    elif "order" in label:
        return "order_status"
    elif "payment" in label:
        return "payment_issue"
    elif "technical" in label or "issue" in label:
        return "technical_issue"
    elif "subscription" in label:
        return "subscription"
    elif "complaint" in label:
        return "complaint"
    elif "greeting" in label:
        return "greeting"
    elif "bye" in label:
        return "goodbye"
    else:
        return "other"


# ---------------- CLEAN TEXT ---------------- #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- PREPARE DATA ---------------- #
texts = []
labels = []

for _, row in df.iterrows():
    text = clean_text(row["Customer Remarks"])
    intent = map_label(row["category"])

    if intent not in label_map:
        continue

    if len(text.split()) < 3:
        continue

    texts.append(text)
    labels.append(label_map[intent])


# 🔥 REMOVE DUPLICATES
unique_data = list(set(zip(texts, labels)))
texts, labels = zip(*unique_data)
texts, labels = list(texts), list(labels)

print("Total samples:", len(texts))


# ---------------- SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)


# ---------------- TOKENIZER ---------------- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=64)

train_encodings = tokenize(X_train)
test_encodings = tokenize(X_test)


# ---------------- DATASET CLASS ---------------- #
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IntentDataset(train_encodings, y_train)
test_dataset = IntentDataset(test_encodings, y_test)


# ---------------- MODEL ---------------- #
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_map),
    id2label=id2label,
    label2id=label_map
)


# ---------------- METRICS ---------------- #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


# ---------------- TRAINING ---------------- #
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    save_energy="no"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# ---------------- TRAIN ---------------- #
trainer.train()


# ---------------- EVALUATION ---------------- #
predictions = trainer.predict(test_dataset)

y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_recall_fscore_support(y_true, y_pred, average="weighted")[0]),
    "recall": float(precision_recall_fscore_support(y_true, y_pred, average="weighted")[1]),
    "f1": float(precision_recall_fscore_support(y_true, y_pred, average="weighted")[2]),
}

print("📊 Metrics:", metrics)


# ---------------- SAVE ---------------- #
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Done!")