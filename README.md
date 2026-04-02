# 🎙️ AI Voice Bot for Customer Support

## 📌 Overview

This project is a production-ready AI Voice Bot that handles customer
queries using speech. It uses a hybrid AI system combining rule-based
logic and machine learning for high accuracy and robustness.

------------------------------------------------------------------------

## 🧠 Architecture

User Voice → ASR (Whisper) → Text Normalization → Rule Engine → ML Model
→ Response → TTS

------------------------------------------------------------------------

## 🤖 Model Choices Justification

### 1. Automatic Speech Recognition (ASR)

We use Whisper for speech-to-text conversion.

**Why Whisper?** - Handles noisy audio and accents - Works well for
real-world conversational input - Robust for incomplete sentences
(common in voice input)

------------------------------------------------------------------------

### 2. Intent Classification Model

We use **DistilBERT (distilbert-base-uncased)**.

**Why DistilBERT?** - Lightweight and fast (good for real-time
systems) - Strong language understanding - Pretrained on large
datasets - Easy to fine-tune on domain-specific data

------------------------------------------------------------------------

### 3. Hybrid Rule + ML Approach

We combine: - Rule-based system (high precision) - ML model
(generalization)

**Why Hybrid?** - Rules ensure correctness for critical intents (refund,
cancel) - ML handles flexible/unseen queries - Improves real-world
reliability

------------------------------------------------------------------------

### 4. Text-to-Speech (TTS)

Used to convert responses into audio.

**Why?** - Enables full voice interaction - Low latency - Enhances user
experience

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1. Clone Repository

``` bash
git clone https://github.com/deepsen839/voice_bot.git
cd real_time_audio
```

### 2. Create Virtual Environment

``` bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run Backend (FastAPI)

``` bash
uvicorn app.main:app --reload
```

Backend runs at:

    http://localhost:8000

### 5. Run Frontend (React)

``` bash
cd frontend
npm install
npm start
```

Frontend runs at:

    http://localhost:3000

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Word Error Rate (WER)

WER is used to evaluate ASR performance.

------------------------------------------------------------------------

## 🔌 API Usage

### POST `/voicebot`

**Form Data:** - file: audio file - reference_text (optional)

**Response:**

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

Available in:

    /app/demo/

------------------------------------------------------------------------

## 📸 Demo

Include: - API screenshots - UI interaction - Voice input/output -
Confusion matrix

------------------------------------------------------------------------

## 🚀 Features

-   Real-time voice processing
-   Hybrid AI (Rule + ML)
-   Noise-robust ASR
-   Confidence scoring
-   WER evaluation
-   Scalable FastAPI backend

------------------------------------------------------------------------

## 💡 Key Insight

Pure ML models struggle with critical intents in low-data scenarios.

This system solves it using: 👉 Rule-based precision + ML generalization

------------------------------------------------------------------------

## 📦 Tech Stack

-   FastAPI (Backend)
-   React (Frontend)
-   Transformers (DistilBERT)
-   Whisper (ASR)
-   PyTorch
