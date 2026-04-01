import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- CONFIG ---------------- #

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 10

# Absolute path (safe for any execution location)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, "intent_model")

# ---------------- CREATE MODEL ---------------- #

print("🔄 Loading base model and tokenizer...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------- SAVE MODEL ---------------- #

os.makedirs(SAVE_PATH, exist_ok=True)

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\n✅ Base model initialized and saved successfully!")
print(f"📁 Saved at: {SAVE_PATH}")