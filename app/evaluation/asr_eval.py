import evaluate
from datasets import load_dataset
from models.asr import transcribe_audio

# Load WER metric
wer_metric = evaluate.load("wer")

# Load ASR dataset (audio + text)
dataset = load_dataset("librispeech_asr", "clean", split="test[:20]")

references = []
predictions = []

for sample in dataset:
    audio = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]

    # Save temp audio
    import soundfile as sf
    temp_path = "temp.wav"
    sf.write(temp_path, audio, sampling_rate)

    # ASR prediction
    pred = transcribe_audio(temp_path)

    references.append(sample["text"].lower())
    predictions.append(pred.lower())

    print("REF:", sample["text"])
    print("PRED:", pred)
    print("-" * 30)

# Compute WER
wer_score = wer_metric.compute(
    references=references,
    predictions=predictions
)

print("🔥 FINAL WER:", wer_score)