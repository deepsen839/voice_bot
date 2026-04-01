# training/evaluate_asr.py

from models.asr import transcribe_audio
from utils.metrics import compute_wer

test_data = [
    ("audio1.wav", "where is my order"),
    ("audio2.wav", "i want refund"),
    ("audio3.wav", "cancel my subscription")
]

wers = []

for path, ref in test_data:
    pred = transcribe_audio(path)
    score = compute_wer(ref, pred)

    print(f"\nAudio: {path}")
    print("Ref:", ref)
    print("Pred:", pred)
    print("WER:", score)

    wers.append(score)

print("\n✅ Average WER:", sum(wers) / len(wers))