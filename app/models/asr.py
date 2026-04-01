from faster_whisper import WhisperModel
model = WhisperModel("base",compute_type="float32")

def transcribe_audio(path):
    segments, _ = model.transcribe(
        path,
        beam_size=1,          # ⚡ faster
        vad_filter=True       # 🔥 remove silence
    )
    return " ".join([s.text for s in segments])