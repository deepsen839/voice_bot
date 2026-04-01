import evaluate
from datasets import load_dataset, Audio
from app.models.asr import transcribe_audio
import soundfile as sf
import re
import json
import os
import io

# ==============================
# ENV CONFIG (optional but safe)
# ==============================
os.environ["HF_HOME"] = "/home/deep/.cache/huggingface"

# ==============================
# Load WER metric
# ==============================
wer_metric = evaluate.load("wer")

# ==============================
# Load dataset (streaming)
# ==============================
dataset = load_dataset(
    "librispeech_asr",
    "clean",
    split="test",
    streaming=True
)

# 🔥 IMPORTANT: Disable torchcodec decoding
dataset = dataset.cast_column("audio", Audio(decode=False))

# ==============================
# Text cleaning
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ==============================
# Evaluation loop
# ==============================
refs = []
preds = []

MAX_SAMPLES = 10  # adjust if needed

for i, sample in enumerate(dataset):
    if i >= MAX_SAMPLES:
        break

    temp_path = f"temp_{i}.wav"

    try:
        # ✅ SAFE audio decoding (NO torchcodec)
        audio_bytes = sample["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        # Save temp wav (if your ASR requires file input)
        sf.write(temp_path, audio, sr)

        # Transcribe
        prediction = transcribe_audio(temp_path)

        # Clean text
        refs.append(clean_text(sample["text"]))
        preds.append(clean_text(prediction))

    except Exception as e:
        print(f"[ERROR] Sample {i}: {e}")

    finally:
        # Always delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==============================
# Compute WER
# ==============================
wer = wer_metric.compute(references=refs, predictions=preds)
print("WER:", wer)

# ==============================
# Save metrics
# ==============================
metrics_path = "app/intent_model/metrics.json"

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
else:
    metrics = {}

metrics["wer"] = float(wer)

with open(metrics_path, "w") as f:
    print("1111")
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved successfully!")