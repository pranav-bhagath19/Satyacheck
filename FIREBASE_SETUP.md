# 🔥 Firebase Setup Guide for SatyaCheck

## Why Firebase? (Not SQL)

| | Firebase Firestore | PostgreSQL |
|---|---|---|
| **Setup time** | 5 minutes | 30+ minutes |
| **Server needed?** | ❌ None (serverless) | ✅ Yes |
| **Real-time feed** | ✅ Built-in onSnapshot | ❌ Need WebSockets |
| **Free tier** | 50k reads/day | Paid hosting |
| **Scale** | Auto | Manual |
| **Best for** | This app ✅ | Complex queries |

**Verdict**: Firebase wins for this use case — no backend server, real-time feed, free tier is enough.

---

## What We Store (and Why)

```
Firestore Collection: "checks"
└── Document (auto-ID)
    ├── textHash: "sha256..."    ← Dedup only, text NOT stored
    ├── preview: "BREAKING!! ..." ← First 80 chars for feed UI  
    ├── label: "FAKE"           ← For stats
    ├── confidence: 0.87        ← For analytics
    ├── sourcesOk: 0            ← Sources that confirmed
    ├── sourcesTotal: 8         ← Total sources checked
    ├── lang: "en"              ← Language analytics
    ├── feedback: "agree"       ← Optional user feedback
    └── timestamp: ServerTimestamp ← For ordering feed
```

## What We DON'T Store (Privacy)
- ❌ Full news text
- ❌ User IP address  
- ❌ Browser/device info
- ❌ Any personal data

---

## Setup Steps (5 minutes)

### 1. Create Firebase Project
1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Click **"Add Project"**
3. Name it: `satyacheck`
4. Disable Google Analytics (not needed)

### 2. Enable Firestore
1. In left sidebar → **Firestore Database**
2. Click **"Create Database"**
3. Choose **"Start in test mode"** (for development)
4. Select region: `asia-south1` (Mumbai — closest to India)

### 3. Get Your Config
1. Project Settings (⚙️) → **Your Apps**
2. Click **"</>"** (Web app)
3. Register app name: `satyacheck-web`
4. Copy the `firebaseConfig` object

### 4. Update index.html
Replace the placeholder config in `index.html`:
```javascript
const firebaseConfig = {
  apiKey: "AIzaSy...",           // ← Your actual key
  authDomain: "satyacheck-xxxxx.firebaseapp.com",
  projectId: "satyacheck-xxxxx",
  storageBucket: "satyacheck-xxxxx.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123"
};
```

### 5. Set Firestore Rules
In Firestore → **Rules** tab, paste:
```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /checks/{docId} {
      // Anyone can read (for the public feed)
      allow read: if true;
      // Anyone can write new checks (we don't store personal data)
      allow create: if request.resource.data.keys().hasOnly([
        'textHash', 'preview', 'label', 'confidence', 
        'sourcesOk', 'sourcesTotal', 'lang', 'timestamp', 'feedback'
      ]);
      // Only update feedback field
      allow update: if request.resource.data.diff(resource.data).affectedKeys().hasOnly(['feedback']);
    }
  }
}
```

### 6. Add Index (for ordered feed)
In Firestore → **Indexes** → **Composite**:
- Collection: `checks`
- Field 1: `timestamp` (Descending)
- Click **Create Index**

---

## Deployment (Free Options)

### Option A: Firebase Hosting (Recommended)
```bash
npm install -g firebase-tools
firebase login
firebase init hosting
# Set public directory to: frontend
# Single-page app: No
firebase deploy
# → https://satyacheck-xxxxx.web.app
```

### Option B: Netlify (Drag & Drop)
1. Go to [netlify.com](https://netlify.com)
2. Drag the `frontend/` folder onto the deploy area
3. Done! Free HTTPS URL instantly.

### Option C: GitHub Pages
```bash
git init && git add . && git commit -m "init"
git push origin main
# Enable GitHub Pages in repo Settings → Pages
```

---

## Production Improvements

For real RSS-based source checking, add a small backend:

```python
# news_checker.py — Call from a Cloud Function
import feedparser, re

RSS_FEEDS = {
    'ndtv': 'https://feeds.feedburner.com/ndtvnews-top-stories',
    'thehindu': 'https://www.thehindu.com/news/national/feeder/default.rss',
    'bbc': 'https://feeds.bbci.co.uk/news/world/asia/india/rss.xml',
    'reuters': 'https://feeds.reuters.com/reuters/INtopNews',
    'pib': 'https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3',
}

def check_rss(keywords, source_id):
    feed = feedparser.parse(RSS_FEEDS[source_id])
    for entry in feed.entries:
        title = entry.get('title', '').lower()
        summary = entry.get('summary', '').lower()
        matches = sum(1 for kw in keywords if kw in title or kw in summary)
        if matches >= 2:
            return {'found': True, 'headline': entry.title, 'url': entry.link}
    return {'found': False}
```

Deploy as a Firebase Cloud Function:
```bash
firebase init functions
# Add feedparser, deploy
firebase deploy --only functions
```

---

## Free Tier Limits (More Than Enough)

| Resource | Free Limit | Your Usage |
|---|---|---|
| Reads/day | 50,000 | ~200-500 |
| Writes/day | 20,000 | ~50-100 |
| Storage | 1 GB | ~10 MB |
| Bandwidth | 10 GB/month | ~1 GB |

You'd need **10,000+ daily active users** to hit these limits. 🎉
