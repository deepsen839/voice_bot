# app/models/intent.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("intent_model")
model = AutoModelForSequenceClassification.from_pretrained("intent_model")

labels = [
    "order_status", "cancel_order", "refund", "subscription",
    "technical_issue", "payment_issue", "greeting",
    "goodbye", "complaint", "other"
]

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    return {
        "intent": labels[pred.item()],
        "confidence": conf.item()
    }