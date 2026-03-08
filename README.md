# 🇮🇳 SatyaCheck - Indian Fake News Detector
### *सत्यमेव जयते | Truth Always Prevails*

AI-powered fake news detection for **English and Telugu** news, built with FastAPI, scikit-learn, HTMX, and PostgreSQL.

---

## 📸 Features

- 🔍 **Instant Detection** - ML-powered analysis in <200ms
- 🗣️ **Bilingual** - English + Telugu (Romanized + Unicode script)
- 📊 **Risk Scoring** - VERY HIGH / HIGH / MEDIUM / LOW / VERY LOW
- 🚩 **Red Flag Detection** - Identifies urgency language, caps abuse, scam patterns
- 📈 **Analytics Dashboard** - Track detection statistics
- 🌐 **Trusted Sources** - Database of verified Indian news outlets
- 💬 **User Feedback** - Improve model with community corrections
- 🔒 **Privacy First** - No text stored verbatim, only anonymized hashes

---

## 🗂️ Project Structure

```
satyacheck/
├── backend/
│   ├── main.py              # FastAPI app + API routes
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # HTMX + TailwindCSS UI
├── ml/
│   ├── train.py             # ML training pipeline
│   ├── multilingual_model.pkl   # Trained model (after running train.py)
│   └── vectorizer.pkl       # TF-IDF vectorizer
├── deployment/
│   └── nginx.conf           # Production nginx config
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚡ Quick Start

### Option 1: Local Development (SQLite)

```bash
# 1. Clone repo
git clone https://github.com/yourname/satyacheck
cd satyacheck

# 2. Train ML model (optional, demo model auto-generated)
pip install -r ml/requirements.txt  # or use Google Colab
python ml/train.py
# → Saves multilingual_model.pkl, vectorizer.pkl

# Copy models to backend directory
cp ml/multilingual_model.pkl backend/
cp ml/vectorizer.pkl backend/
cp ml/model_metrics.json backend/

# 3. Start backend
cd backend
pip install -r requirements.txt
DEV_MODE=true uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Open browser
open http://localhost:8000
```

### Option 2: Docker (Production)

```bash
# Build and start all services
docker-compose up --build

# Access at http://localhost:80
```

---

## 🤖 ML Training (Google Colab Recommended)

```python
# In Google Colab:
!pip install scikit-learn indic-nlp-library nltk pandas langdetect joblib

# Run the training script
!python ml/train.py

# Expected output:
# ✅ Training complete!
# Model saved as: multilingual_model.pkl
# F1 Score: 0.8700+
```

### Training Datasets

| Dataset | Language | Samples | Source |
|---------|----------|---------|--------|
| Indian Fake News | English | ~40k | Kaggle |
| Custom Telugu Corpus | Telugu | ~200 | Hand-crafted |
| The Hindu / Eenadu | EN + TE | ~1000 | Scraped |
| WhatsApp Forwards | English | ~500 | Collected |

---

## 🔌 API Reference

### POST `/api/check`
```json
{
  "text": "BREAKING! Free 5G phones for all! Apply now!!!"
}
```
Response:
```json
{
  "prediction_id": 42,
  "label": "FAKE",
  "fake_probability": 0.94,
  "real_probability": 0.06,
  "confidence": 0.94,
  "language": "en",
  "language_name": "English",
  "risk_level": "VERY HIGH",
  "red_flags": [
    "🚨 Urgency language detected",
    "💰 Financial scam pattern",
    "🔞 Excessive caps/exclamation marks"
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET `/api/stats`
```json
{
  "total_checks": 1250,
  "fake_detected": 780,
  "real_detected": 470,
  "fake_percentage": 62.4,
  "avg_confidence": 0.87,
  "checks_today": 45,
  "languages": {"english": 1100, "telugu": 150}
}
```

Full API docs at: `http://localhost:8000/api/docs`

---

## 🚀 Deployment Options

### Railway (Recommended for Students - Free)
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Deploy
railway init
railway add postgresql  # Add PostgreSQL plugin
railway up

# Set environment variables
railway variables set DEV_MODE=false
```

### Render
```yaml
# render.yaml
services:
  - type: web
    name: satyacheck
    env: python
    buildCommand: "pip install -r backend/requirements.txt"
    startCommand: "cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: satyacheck-db
          property: connectionString
```

### Antigravity / VPS
```bash
# SSH into your server
ssh user@your-server.com

# Clone and start with Docker
git clone https://github.com/you/satyacheck
cd satyacheck
docker-compose -f docker-compose.yml up -d

# Check status
docker-compose ps
docker-compose logs api
```

---

## 🌿 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | sqlite:///fakenews_dev.db | PostgreSQL URL for production |
| `DEV_MODE` | `true` | Use SQLite (dev) vs PostgreSQL (prod) |
| `MODEL_PATH` | `multilingual_model.pkl` | Path to trained model |
| `VECTORIZER_PATH` | `vectorizer.pkl` | Path to TF-IDF vectorizer |

---

## 📊 Model Architecture

```
Input Text (English/Telugu)
        │
        ▼
┌─────────────────────┐
│  Language Detection  │ ─── Telugu? ──→ Transliteration (indic-nlp)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Text Preprocessing  │
│  • Lowercase         │
│  • Remove URLs/HTML  │
│  • Remove stopwords  │
│  • NLTK Stemming     │
└─────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  TF-IDF Vectorizer           │
│  • 8000 features             │
│  • Unigrams + Bigrams        │
│  • Sublinear TF normalization│
└─────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Voting Ensemble (soft voting)    │
│  ┌─────────────┐                 │
│  │  LogisticReg│ ──╮             │
│  └─────────────┘  │             │
│  ┌─────────────┐  ├──→ Vote → P(FAKE) │
│  │RandomForest │ ──┤             │
│  └─────────────┘  │             │
│  ┌─────────────┐  │             │
│  │  NaiveBayes │ ──╯             │
│  └─────────────┘                 │
└──────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Risk Classification │
│  • VERY HIGH (>80%)  │
│  • HIGH (60-80%)     │
│  • MEDIUM (40-60%)   │
│  • LOW (20-40%)      │
│  • VERY LOW (<20%)   │
└─────────────────────┘
```

---

## 🧪 Testing

```bash
# Test the API
curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"text": "ISRO successfully launched GSAT-20 satellite"}'

# Expected: {"label": "REAL", ...}

curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"text": "BREAKING! Free phones for everyone! Forward before deleted!!!"}'

# Expected: {"label": "FAKE", "risk_level": "VERY HIGH", ...}
```

---

## 🙏 Credits

- **The Hindu** & **Eenadu** for trusted news reference
- **Kaggle Indian Fake News Dataset** community
- **indic-nlp-library** for Telugu NLP
- **FastAPI** & **HTMX** communities
- **Tailwind CSS** for styling

---

## 📜 License

MIT License — Free to use, modify, and deploy for educational purposes.

---

*Built with 🧡 for the Indian CS student community. Fight misinformation!*

**💡 Pro Tip**: Run `python ml/train.py` in Google Colab for faster training on GPU!
