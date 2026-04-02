# 🎙️ AI Voice Bot for Customer Support

## 📌 Overview

This project is a production-ready AI Voice Bot that handles customer
queries using speech.

------------------------------------------------------------------------

## 🧠 Architecture

User Voice → ASR (Whisper) → Text Normalization → Rule Engine → ML Model
→ Response → TTS

------------------------------------------------------------------------

## 🤖 Model Choices

-   ASR: Whisper (robust for noisy audio)
-   Intent Model: DistilBERT (fast + accurate)
-   Rule Engine: Improves precision for critical intents
-   TTS: Lightweight synthesis

------------------------------------------------------------------------

## ⚙️ Setup Instructions

``` bash
git clone <repo>
cd real_time_audio
pip install -r requirements.txt
uvicorn app.main:app --reload
```

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   WER (Word Error Rate)

------------------------------------------------------------------------

## 🔌 API Usage

### 1. Voice Bot

POST `/voicebot`

Form Data: - file: audio file - reference_text (optional)

Response:

``` json
{
  "text": "...",
  "intent": "...",
  "response": "...",
  "audio": "...",
  "wer": 0.25
}
```

------------------------------------------------------------------------

## 🎧 Sample Test Audio

Located in `/app/demo/`

------------------------------------------------------------------------

## 📸 Demo

Include screenshots or demo video (3--5 mins)

------------------------------------------------------------------------

## 🚀 Features

-   Real-time voice processing
-   Hybrid AI (Rule + ML)
-   Robust ASR handling
-   Scalable FastAPI backend
