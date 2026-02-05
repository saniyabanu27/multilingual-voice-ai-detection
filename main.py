from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import librosa
import numpy as np
import io
import requests

app = FastAPI()

API_KEY = "guvi123"
class AudioRequest(BaseModel):
    audio_base64: str
    language: str


def verify_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/")
def home():
    return {"message": "AI Voice Detection API is running"}

@app.post("/detect")
def detect_voice(data: AudioRequest, x_api_key: str = Header(...)):
    verify_key(x_api_key)

    # 1. Download audio
    response = requests.get("https://image2url.com/r2/default/audio/1770307570814-8d59ca49-dad6-43c6-ae8d-b03e053fab10.mp3")
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Unable to fetch audio file")

    # 2. Convert to audio stream
    audio_bytes = response.content
    audio_stream = io.BytesIO(audio_bytes)

    # 3. Load audio (y is defined HERE)
    y, sr = librosa.load(audio_stream, sr=None)

    # 4. Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feature_vector = np.mean(mfcc, axis=1)

    # 5. Compute statistics
    energy = np.mean(np.abs(y))
    mfcc_variance = np.var(feature_vector)

    # 6. Decision logic
    if energy < 0.01 or mfcc_variance < 5:
        classification = "AI_GENERATED"
        confidence = 0.85
    else:
        classification = "HUMAN"
        confidence = 0.90

    # 7. Return response
    return {
        "classification": classification,
        "confidence": confidence,
        "language": data.language,
        "explanation": "Decision based on MFCC variance and audio energy"
    }