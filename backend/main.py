"""
SatyaCheck – Real-time News Verification API
Uses DuckDuckGo search (no API key needed) to verify claims against
trusted Indian and international news sources in real-time.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import os
import time
import urllib.parse
from datetime import datetime
import hashlib
from typing import List
import requests

from dotenv import load_dotenv
load_dotenv()

# Optional language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

# DuckDuckGo search
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except Exception:
    DDG_AVAILABLE = False

# ============================================================
# LOGGING & CONFIG
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "satyacheck-3d9fa")
FIREBASE_COLLECTION  = "checks"

# ============================================================
# TRUSTED SOURCE DOMAINS
# ============================================================
INDIAN_SOURCES = {
    "thehindu.com", "ndtv.com", "timesofindia.indiatimes.com",
    "aninews.in", "ptinews.com", "indiatoday.in", "hindustantimes.com",
    "theprint.in", "thewire.in", "scroll.in", "news18.com",
}
INTL_SOURCES = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "aljazeera.com", "theguardian.com", "nytimes.com",
}
FACTCHECK_SOURCES = {
    "altnews.in", "boomlive.in", "factcheck.org",
    "vishvasnews.com", "thequint.com",
}

ALL_TRUSTED = INDIAN_SOURCES | INTL_SOURCES | FACTCHECK_SOURCES


def domain_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def classify_domain(domain: str):
    for d in FACTCHECK_SOURCES:
        if d in domain:
            return "factcheck"
    for d in INDIAN_SOURCES | INTL_SOURCES:
        if d in domain:
            return "trusted"
    return None


# ============================================================
# REAL-TIME SEARCH ENGINE (DuckDuckGo — no API key needed)
# ============================================================
def search_and_verify(query: str):
    """
    Step 1: Search for claim using DuckDuckGo news search.
    Step 2: Analyse results against trusted/fact-check domains.
    Returns (verdict, confidence, sources_list)
    """
    found_trusted: List[str]   = []
    found_factcheck: List[str] = []

    if not DDG_AVAILABLE:
        logger.warning("duckduckgo-search not available – falling back to UNVERIFIED")
        return "UNVERIFIED", 0.5, []

    try:
        with DDGS() as ddgs:
            # Search news tab first (most relevant for recency)
            results = list(ddgs.news(query, max_results=15))
            if not results:
                # Fall back to regular web search
                results = list(ddgs.text(query, max_results=15))

        logger.info(f"DDG returned {len(results)} results for query")

        for r in results:
            url  = r.get("url") or r.get("href", "")
            body = (r.get("body") or r.get("title") or "").lower()
            dom  = domain_of(url)
            cat  = classify_domain(dom)

            if cat == "factcheck":
                found_factcheck.append(url)
                # Check if fact-checker is confirming or debunking
                # words that suggest debunking
                if any(w in body for w in ["false", "fake", "misleading", "misinformation",
                                            "fabricated", "debunk", "no evidence", "incorrect"]):
                    # Strongly FAKE signal from a fact-checker
                    logger.info(f"Fact-checker debunk signal: {dom}")
                else:
                    # Fact-checker covering as real news
                    found_trusted.append(url)

            elif cat == "trusted":
                found_trusted.append(url)

    except Exception as e:
        logger.error(f"DDG search failed: {e}")
        return "UNVERIFIED", 0.5, []

    # ── Step 2: Evidence Analysis ──────────────────────────────
    all_sources = list(dict.fromkeys(found_factcheck + found_trusted))  # dedup, factcheck first

    if found_factcheck and not found_trusted:
        # Fact-checkers covered it but no mainstream source confirms it → FAKE
        return "FAKE", 0.88, all_sources[:5]

    if len(found_trusted) >= 3:
        return "REAL", 0.92, found_trusted[:5]

    if len(found_trusted) == 2:
        return "REAL", 0.82, found_trusted[:5]

    if len(found_trusted) == 1:
        # Only one source → UNVERIFIED (per your 7-day rule)
        return "UNVERIFIED", 0.60, found_trusted

    if found_factcheck:
        # Fact-checkers flagged it and no trusted sources confirm it
        return "FAKE", 0.78, found_factcheck[:3]

    # Zero credible sources at all
    return "UNVERIFIED", 0.45, []


# ============================================================
# LANGUAGE DETECTION
# ============================================================
def detect_language_name(text: str) -> str:
    telugu = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    hindi  = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if telugu > len(text) * 0.1:
        return "Telugu"
    if hindi > len(text) * 0.1:
        return "Hindi"
    if LANGDETECT_AVAILABLE:
        try:
            code = detect(text)
            return {"te": "Telugu", "hi": "Hindi", "en": "English"}.get(code, "Other")
        except Exception:
            pass
    return "English"


# ============================================================
# SCHEMAS
# ============================================================
class CheckRequest(BaseModel):
    text: str = Field(..., description="News headline or paragraph to verify")

class CheckResponse(BaseModel):
    verdict: str          # REAL | FAKE | UNVERIFIED
    confidence: float     # 0.0 – 1.0
    reason: str           # One sentence in input language
    sources: List[str]    # Verified URLs
    language_detected: str


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="SatyaCheck Verification Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://satyacheck.netlify.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Process-time header middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = time.time() - t0
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    if elapsed > 5:
        logger.warning(f"Slow request ({elapsed:.1f}s): {request.url.path}")
    return response


@app.get("/")
def root():
    return {"message": "SatyaCheck Verification API v2.0 — real-time"}


@app.get("/health")
def health():
    return {"status": "ok", "ddg_available": DDG_AVAILABLE}


@app.get("/api/stats")
def get_stats():
    """Returns placeholder stats (extend with real Firebase data if needed)."""
    return {"total_checks": 1254, "fake_detected": 420, "real_detected": 812}


@app.get("/api/feed")
def get_feed():
    return []


@app.post("/api/check", response_model=CheckResponse)
async def check_news(body: CheckRequest):
    text = body.text.strip()

    if not text or len(text) < 10:
        raise HTTPException(400, "Text too short – please provide a fuller headline or claim.")
    if len(text) > 2000:
        raise HTTPException(400, "Text exceeds 2000 character limit.")

    lang = detect_language_name(text)
    t0   = time.time()

    verdict, conf, sources = search_and_verify(text)

    elapsed = time.time() - t0
    logger.info(f"Verdict: {verdict} | conf: {conf:.2f} | sources: {len(sources)} | {elapsed:.2f}s")

    # Build reason in detected language
    REASONS = {
        "REAL": {
            "English": f"Confirmed by {len(sources)} trusted news source(s) in real-time search.",
            "Telugu":  "నిజమైన వార్తల వనరులు ఈ సమాచారాన్ని నిర్ధారిస్తున్నాయి.",
            "Hindi":   "वास्तविक समाचार स्रोतों ने इस जानकारी की पुष्टि की है।",
        },
        "FAKE": {
            "English": "No credible news sources confirm this claim; fact-checkers have flagged it.",
            "Telugu":  "విశ్వసనీయ వార్తల వనరులు ఈ క్లెయిమ్‌ను ధృవీకరించడం లేదు; ఫాక్ట్-చెకర్స్ దీన్ని తప్పుగా గుర్తించారు.",
            "Hindi":   "कोई विश्वसनीय समाचार स्रोत इस दावे की पुष्टि नहीं करता; फ़ैक्ट-चेकर्स ने इसे गलत बताया है।",
        },
        "UNVERIFIED": {
            "English": "Insufficient trusted sources found to confirm or deny this claim at this time.",
            "Telugu":  "ప్రస్తుతానికి ఈ క్లెయిమ్‌ను ధృవీకరించడానికి తగిన విశ్వసనీయ వనరులు లేవు.",
            "Hindi":   "इस समय दावे की पुष्टि या खंडन के लिए पर्याप्त विश्वसनीय स्रोत नहीं हैं।",
        },
    }
    reason = REASONS[verdict].get(lang, REASONS[verdict]["English"])

    # Async fire-and-forget to Firebase (don't block response)
    try:
        save_url = (
            f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}"
            f"/databases/(default)/documents/{FIREBASE_COLLECTION}"
        )
        fb_data = {
            "fields": {
                "preview":    {"stringValue": text[:120]},
                "label":      {"stringValue": verdict},
                "confidence": {"doubleValue": conf},
                "lang":       {"stringValue": lang},
                "sources":    {"integerValue": str(len(sources))},
                "timestamp":  {"timestampValue": datetime.utcnow().isoformat() + "Z"},
            }
        }
        requests.post(save_url, json=fb_data, timeout=2)
    except Exception:
        pass

    return CheckResponse(
        verdict=verdict,
        confidence=round(conf, 4),
        reason=reason,
        sources=sources,
        language_detected=lang,
    )