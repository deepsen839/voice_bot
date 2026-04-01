import json
import os
import numpy as np
import pickle

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- CONFIG ---------------- #
MODEL_NAME = "distilbert-base-uncased"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "intent_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- LOAD DATA ---------------- #
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
data = ds["train"]


# # ---------------- INTENT MAPPING ---------------- #
# def map_to_intent(text):
#     text = text.lower()

#     if "refund" in text:
#         return "refund"
#     elif "cancel" in text:
#         return "cancel_order"
#     elif "order" in text or "track" in text:
#         return "order_status"
#     elif "subscription" in text:
#         return "subscription"
#     elif "payment" in text or "card" in text:
#         return "payment_issue"
#     elif "error" in text or "bug" in text:
#         return "technical_issue"
#     elif "hello" in text or "hi" in text:
#         return "greeting"
#     elif "bye" in text:
#         return "goodbye"
#     elif "bad" in text or "complaint" in text:
#         return "complaint"
#     else:
#         return "other"


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


# ---------------- PREPARE DATA ---------------- #
# texts = []
# labels = []

# for item in data:
#     text = item["instruction"]
#     intent = map_to_intent(text)

#     texts.append(text)
#     labels.append(label_map[intent])

import re

def clean_text(text):
    return re.sub(r"\{\{.*?\}\}", "", text).strip().lower()


texts = []
labels = []

for item in data:
    text = clean_text(item["instruction"])

    # ✅ USE REAL LABEL FROM DATASET
    intent = item["intent"]

    if intent not in label_map:
        continue

    texts.append(text)
    labels.append(label_map[intent])
dataset = Dataset.from_dict({
    "text": texts,
    "label": labels
})

dataset = dataset.train_test_split(test_size=0.2, seed=42)


# ---------------- TOKENIZATION ---------------- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

dataset = dataset.map(tokenize)

dataset = dataset.remove_columns(["text"])
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")


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

    y_pred = np.argmax(logits, axis=1)
    y_true = labels

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

    print("📊 Eval Metrics:", metrics)
    return metrics


# ---------------- TRAINING ---------------- #
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)


# ---------------- TRAIN ---------------- #
trainer.train()


# ---------------- FINAL EVALUATION ---------------- #
predictions = trainer.predict(dataset["test"])

y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)


# ---------------- SAVE METRICS ---------------- #
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)
accuracy = accuracy_score(y_true, y_pred)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1)
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("📊 Final Metrics:", metrics)


# ---------------- CONFUSION MATRIX ---------------- #
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=list(label_map.keys()),
    yticklabels=list(label_map.keys())
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


# ---------------- SAVE MODEL ---------------- #
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "labels.pkl"), "wb") as f:
    pickle.dump(label_map, f)


print("✅ Model trained and saved successfully!")

