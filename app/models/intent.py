import os
import torch
import pickle
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR,'app', "intent_model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

with open(os.path.join(MODEL_PATH, "labels.pkl"), "rb") as f:
    label_map = pickle.load(f)

# 🔥 FIX
id2label = {v: k for k, v in label_map.items()}

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    print(id2label[pred.item()],float(conf.item()))
    return {
        "intent": id2label[pred.item()],
        "confidence": float(conf.item())
    }