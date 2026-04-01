# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import shutil, os, uuid

from models.asr import transcribe_audio
from models.intent import predict_intent
from models.response import generate_response
from models.tts import text_to_speech
from fastapi.responses import FileResponse
import json
from fastapi import Form
from evaluation.metrics import compute_wer
app = FastAPI()

# ---------------- CREATE AUDIO FOLDER ---------------- #
os.makedirs("audio", exist_ok=True)

# ---------------- CORS ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- STATIC AUDIO ---------------- #
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

# ---------------- TRANSCRIBE ---------------- #
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    path = f"temp_{uuid.uuid4()}.wav"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = transcribe_audio(path)
    os.remove(path)

    return {"text": text}

# ---------------- INTENT ---------------- #
@app.post("/predict-intent")
async def intent_api(text: str):
    return predict_intent(text)

# ---------------- RESPONSE ---------------- #
@app.post("/generate-response")
async def response_api(intent: str):
    return {"response": generate_response(intent)}

# ---------------- TTS ---------------- #
@app.post("/synthesize")
async def synthesize(text: str):
    audio = text_to_speech(text)
    return {"audio": audio}

# ---------------- FULL PIPELINE ---------------- #
from fastapi import Form

@app.post("/voicebot")
async def voicebot(
    file: UploadFile = File(...),
    reference_text: str = Form(None)   # ✅ add this
):
    path = f"temp_{uuid.uuid4()}.wav"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    from utils.audio import convert_to_wav
    wav_path = convert_to_wav(path)

    text = transcribe_audio(wav_path)

    if not text.strip():
        os.remove(path)
        os.remove(wav_path)
        return {"error": "No speech detected"}

    # ---------------- INTENT ---------------- #
    intent = predict_intent(text)

    # ---------------- RESPONSE ---------------- #
    response = generate_response(intent["intent"])

    # ---------------- TTS ---------------- #
    audio = await text_to_speech(response)

    # ---------------- WER ---------------- #
    wer_score = None
    if reference_text:
        wer_score = compute_wer([reference_text], [text])
    # ---------------- CLEANUP ---------------- #
    os.remove(path)
    # os.remove(wav_path)

    return {
        "text": text,
        "intent": intent,
        "response": response,
        "audio": audio,
        "wer": wer_score   # ✅ return WER
    }

# ---------------- METRICS ---------------- #
@app.get("/metrics")
def get_metrics():
    path = "intent_model/metrics.json"

    if not os.path.exists(path):
        return {"error": "Metrics not found"}

    with open(path) as f:
        # return eval(f.read())  # simple load
        return json.load(f)


# ---------------- CONFUSION MATRIX ---------------- #
@app.get("/confusion-matrix")
def get_confusion_matrix():
    path = "intent_model/confusion_matrix.png"

    if not os.path.exists(path):
        return {"error": "Not found"}

    return FileResponse(path)