"""
SatyaCheck – Indian Fake News Detection API
FastAPI backend with ML + real-time news verification
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
from datetime import datetime
import hashlib
from typing import List
from pathlib import Path
import joblib

# Optional language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except:
    LANGDETECT_AVAILABLE = False


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "multilingual_model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "vectorizer.pkl")

FIREBASE_PROJECT_ID = "satyacheck-3d9fa"
FIREBASE_COLLECTION = "checks"


# ============================================================
# ML SYSTEM
# ============================================================

class MLModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_loaded = False

    def load(self):
        try:
            if Path(MODEL_PATH).exists() and Path(VECTORIZER_PATH).exists():
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.is_loaded = True
                logger.info("✅ ML model loaded")
                logger.info("✅ Vectorizer loaded")
            else:
                logger.error("❌ Model files missing")

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")


ml_system = MLModel()


# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(
    title="SatyaCheck Fake News Detection API",
    description="Indian Fake News Verification System",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting SatyaCheck API")
    ml_system.load()
    logger.info("🌐 Source verification ready")


# ============================================================
# SCHEMAS
# ============================================================

class CheckRequest(BaseModel):
    text: str = Field(..., description="News headline or paragraph")


class CheckResponse(BaseModel):
    verdict: str
    confidence: float
    language: str
    sources_checked: List[str]
    google_news_verified: bool
    explanation: str


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text: str):

    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')

    if telugu_chars > len(text) * 0.1:
        return "te"

    if LANGDETECT_AVAILABLE:
        try:
            return detect(text)
        except:
            pass

    return "en"


# ============================================================
# TRUSTED NEWS SOURCE CHECK
# ============================================================

def check_trusted_sources(query):

    sources = {
        "NDTV": "https://www.ndtv.com",
        "The Hindu": "https://www.thehindu.com",
        "Reuters": "https://www.reuters.com",
        "BBC": "https://www.bbc.com/news",
        "CNN": "https://edition.cnn.com",
        "ABP News": "https://news.abplive.com",
        "TV9 Telugu": "https://tv9telugu.com",
        "PIB": "https://pib.gov.in"
    }

    import re
    words = re.sub(r'[^a-zA-Z0-9\s]', '', query).lower().split()

    keywords = [w for w in words if len(w) > 4]

    if not keywords:
        keywords = words[:3]

    results = []

    headers = {"User-Agent": "Mozilla/5.0"}

    for name, url in sources.items():

        try:

            resp = requests.get(url, headers=headers, timeout=5)

            if resp.status_code == 200:

                text = BeautifulSoup(resp.text, "html.parser").get_text().lower()

                matches = sum(1 for kw in keywords if kw in text)

                if matches >= 2:
                    results.append(name)

        except Exception as e:
            logger.warning(f"Source check failed for {name}: {e}")

    return results


# ============================================================
# GOOGLE NEWS CHECK
# ============================================================

def check_google_news(query):

    try:

        url = f"https://news.google.com/search?q={urllib.parse.quote(query)}"

        headers = {"User-Agent": "Mozilla/5.0"}

        resp = requests.get(url, headers=headers, timeout=5)

        if resp.status_code == 200:

            soup = BeautifulSoup(resp.text, "html.parser")

            links = soup.find_all("a")

            if len(links) > 15:
                return True

    except Exception as e:
        logger.warning(f"Google News check failed: {e}")

    return False


# ============================================================
# FIREBASE STORAGE
# ============================================================

def save_to_firebase(text_hash, text, verdict, confidence, language, sources_ok):

    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{FIREBASE_COLLECTION}"

    preview = text[:80] + ("..." if len(text) > 80 else "")

    data = {
        "fields": {
            "textHash": {"stringValue": text_hash},
            "preview": {"stringValue": preview},
            "label": {"stringValue": verdict},
            "confidence": {"doubleValue": confidence},
            "lang": {"stringValue": language},
            "sourcesOk": {"integerValue": str(sources_ok)},
            "timestamp": {"timestampValue": datetime.utcnow().isoformat() + "Z"}
        }
    }

    try:
        requests.post(url, json=data, timeout=5)

    except Exception as e:
        logger.error(f"Firebase save error: {e}")


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"message": "SatyaCheck API running"}


@app.get("/api/health")
def health():
    return {"status": "healthy", "model_loaded": ml_system.is_loaded}


@app.post("/api/check", response_model=CheckResponse)
def check_news(body: CheckRequest):

    if not ml_system.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not body.text or not isinstance(body.text, str):
        raise HTTPException(status_code=400, detail="Invalid text input")

    text = body.text.strip()

    if len(text) <= 5:
        raise HTTPException(status_code=400, detail="Text too short")

    logger.info("🔍 Running ML prediction")

    try:

        clean_text = text.lower()

        vec = ml_system.vectorizer.transform([clean_text])

        if hasattr(ml_system.model, "predict_proba"):
            proba = ml_system.model.predict_proba(vec)[0]
            ml_real_prob = float(max(proba))
        else:
            prediction = ml_system.model.predict(vec)[0]
            ml_real_prob = 0.9 if prediction == 0 else 0.1

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    language = detect_language(text)

    logger.info("🌐 Checking trusted sources")

    sources = check_trusted_sources(text)

    logger.info("📰 Checking Google News")

    google_verified = check_google_news(text)

    score = ml_real_prob

    if len(sources) >= 2:
        score += 0.2

    if google_verified:
        score += 0.2

    score = min(score, 0.99)

    verdict = "Real" if score > 0.65 else "Fake"

    if verdict == "Real":

        if sources:
            explanation = f"Verified by {', '.join(sources[:3])}"

        elif google_verified:
            explanation = "Found in Google News"

        else:
            explanation = "AI model predicts this is likely real"

    else:

        if not sources and not google_verified:
            explanation = "Claim not found in trusted news sources"

        else:
            explanation = "AI detected misinformation patterns"

    text_hash = hashlib.sha256(text.encode()).hexdigest()

    save_to_firebase(text_hash, text, verdict, score, language, len(sources))

    return CheckResponse(
        verdict=verdict,
        confidence=round(score, 4),
        language=language,
        sources_checked=sources,
        google_news_verified=google_verified,
        explanation=explanation
    )
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ml_system.is_loaded
    }