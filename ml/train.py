"""
Indian Fake News Detection - ML Training Script
Supports: English (Indian news) + Telugu news
Run in Google Colab or local environment
"""

# ============================================================
# STEP 0: Install dependencies
# ============================================================
# !pip install fastapi uvicorn scikit-learn indic-nlp-library nltk pandas
# !pip install langdetect joblib kaggle requests beautifulsoup4

import os
import re
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

# Language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("langdetect not installed. Install: pip install langdetect")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ============================================================
# STEP 1: DATASET LOADING (Indian Focus)
# ============================================================

def load_fake_news_india_dataset():
    """
    Load Indian Fake News dataset.
    Primary: Kaggle 'Indian Fake News' dataset
    Fallback: Generate representative Indian news samples
    """
    datasets = []

    # --- Option 1: Try Kaggle dataset ---
    try:
        # If running in Colab with Kaggle credentials:
        # !kaggle datasets download -d saurabhshahane/fake-news-classification
        # import zipfile; zipfile.ZipFile('fake-news-classification.zip').extractall()
        
        kaggle_files = [
            'WELFake_Dataset.csv',
            'fake-news-classification.csv', 
            'indian_fake_news.csv',
            'Fake.csv',
            'True.csv'
        ]
        for f in kaggle_files:
            if Path(f).exists():
                df = pd.read_csv(f)
                logger.info(f"Loaded Kaggle dataset: {f} ({len(df)} rows)")
                datasets.append(('kaggle', df))
    except Exception as e:
        logger.warning(f"Kaggle dataset not available: {e}")

    # --- Option 2: Create Indian news samples ---
    logger.info("Creating Indian news training samples...")
    
    indian_real_news = [
        # The Hindu style
        ("India's GDP growth rate stands at 6.8% for Q3 2024, according to Ministry of Finance data released today. The economic survey indicates steady progress in manufacturing and services sectors.", "REAL"),
        ("Prime Minister Modi inaugurated the new AIIMS hospital in Rajkot, Gujarat, expanding healthcare access to the region's 3 million residents.", "REAL"),
        ("The Supreme Court of India issued directives to all states regarding implementation of Right to Education Act, emphasizing enrollment targets.", "REAL"),
        ("ISRO's Chandrayaan-3 mission successfully landed on the Moon's south pole on August 23, 2023, making India the first country to achieve this feat.", "REAL"),
        ("The Reserve Bank of India kept repo rate unchanged at 6.5% in its latest monetary policy committee meeting, focusing on inflation control.", "REAL"),
        ("Tata Motors reported quarterly revenue of Rs 1.1 lakh crore, driven by strong demand for Jaguar Land Rover vehicles in international markets.", "REAL"),
        ("The Indian cricket team won the T20 World Cup 2024, defeating South Africa in the final held in Barbados.", "REAL"),
        ("Bengaluru metro Phase 3 construction work commenced with the foundation stone laid by Karnataka Chief Minister Siddaramaiah.", "REAL"),
        ("IIT Bombay researchers developed a cost-effective water purification system that can remove microplastics, published in Nature journal.", "REAL"),
        ("The Union Budget 2024-25 allocated Rs 11.11 lakh crore for infrastructure development, the highest ever capital expenditure.", "REAL"),
        ("Andhra Pradesh government announced implementation of YSR Rythu Bharosa scheme providing Rs 13,500 to each farmer family annually.", "REAL"),
        ("Telangana's Hyderabad ranked as India's top IT hub with over 1,500 companies employing 6 lakh technology professionals.", "REAL"),
        ("The National Health Mission achieved 90% immunization coverage across rural India, WHO report confirmed the milestone.", "REAL"),
        ("Indian Railways completed the electrification of 6,000 km of tracks, reducing diesel consumption by 40% on these routes.", "REAL"),
        ("Mukesh Ambani's Reliance Industries achieved market capitalization of Rs 20 lakh crore, first Indian company to reach this milestone.", "REAL"),
        
        # More Indian real news
        ("The Aadhaar system enrolled its 140 crore users, becoming world's largest biometric database, UIDAI announced.", "REAL"),
        ("Digital payments in India crossed 100 billion transactions in 2023, UPI accounting for 85% of all transactions.", "REAL"),
        ("IIM Ahmedabad placements saw average salary of Rs 35 lakhs per annum with McKinsey hiring maximum students.", "REAL"),
        ("Bhopal gas tragedy survivors received compensation after Supreme Court directed Union Carbide to pay additional amount.", "REAL"),
        ("India's renewable energy capacity reached 200 GW with solar contributing 73 GW, government data shows.", "REAL"),
    ]

    indian_fake_news = [
        # Misinformation patterns common in India
        ("BREAKING: Government announces free 5G phones for all BPL families. Apply on PM Jan Dhan portal before December 31 to avail scheme.", "FAKE"),
        ("VIRAL: Scientists confirm that drinking cow urine cures COVID-19 completely. Ayush Ministry endorses the treatment. Share before deleted!", "FAKE"),
        ("SHOCKING: Rahul Gandhi secretly converted to Islam in London ceremony. Photographic proof obtained by intelligence sources!!!", "FAKE"),
        ("ALERT: WhatsApp is going to charge Rs 500 per month from January. Forward this message to 20 friends to keep your account free!", "FAKE"),
        ("EXPOSED: The Ambani family controls India's entire food supply chain and is deliberately causing inflation to profit billions.", "FAKE"),
        ("BREAKING NEWS: India to merge with Pakistan and Bangladesh by 2025 under UN directive. Modi government hiding this from public.", "FAKE"),
        ("URGENT: New RBI circular makes all 500 rupee notes invalid from next week. Exchange at nearest bank branch immediately!", "FAKE"),
        ("PROVEN: Mobile towers are actually mind control devices. Eat 2 garlic cloves daily to block the radiation effects on brain.", "FAKE"),
        ("EXPOSED: Vaccines contain microchips that track your location for Bill Gates. IIT Delhi professor warns against vaccination!", "FAKE"),
        ("VIRAL: Government planning to impose 90% tax on gold jewelry. Sell all gold within 48 hours or face confiscation!", "FAKE"),
        ("ALERT: Onions grown in Maharashtra contain toxic pesticide that causes cancer. AP and Telangana farmers use safe methods only.", "FAKE"),
        ("BREAKING: Pakistan army has entered Kashmir with 50,000 troops. Indian Army orders evacuation of border areas immediately!", "FAKE"),
        ("CONFIRMED: Eating laddu from certain temple in UP causes COVID. 47 pilgrims died after consuming prasad last week!", "FAKE"),
        ("SHOCKING VIDEO: Congress leader caught in human trafficking sting operation in Delhi. Police have suppressed the FIR!", "FAKE"),
        ("URGENT: New law passed - all vehicles older than 5 years must be scrapped by March. Penalty Rs 1 lakh for non-compliance!", "FAKE"),
        
        # More fake news
        ("WhatsApp Gold has been launched with exclusive features. Only by invitation. Forward to 50 contacts to receive your invitation!", "FAKE"),
        ("BREAKING: China invaded Arunachal Pradesh with 10,000 troops. Modi-Xi secret deal allows Chinese control for 50 years!", "FAKE"),
        ("VIRAL: Eating turmeric mixed with oil at midnight cures diabetes permanently. Ancient Vedic remedy suppressed by pharma lobby!", "FAKE"),
        ("URGENT WARNING: New virus spreading in Mumbai more dangerous than COVID kills in 24 hours wash hands in hot water every hour!", "FAKE"),
        ("CONFIRMED: Elon Musk invests Rs 50,000 crore in Gujarat factory. Every Indian citizen gets Rs 5000 share money. Apply now!", "FAKE"),
    ]

    # Convert to DataFrame
    texts = [t[0] for t in indian_real_news + indian_fake_news]
    labels = [t[1] for t in indian_real_news + indian_fake_news]
    df_indian = pd.DataFrame({'text': texts, 'label': labels, 'language': 'en', 'source': 'indian'})
    datasets.append(('indian_samples', df_indian))

    return datasets


def load_telugu_dataset():
    """
    Load Telugu fake news dataset.
    Sources: Telugu news portals + transliterated samples
    """
    logger.info("Loading Telugu fake news dataset...")
    
    # Telugu news samples (romanized/transliterated for sklearn processing)
    telugu_real = [
        # Real Telugu news (romanized)
        ("Andhra Pradesh lo prati graama panchayat ki broadband internet connectivity ivvadam kosam 'Digital AP' prajaktha praarambhimchaaru.", "REAL"),
        ("Telangana sarkaaru raytulaki vastunna sahaayam 'Rythu Bandhu' scheme kinda Rs 10,000 per acre ichindi. 70 lakh mandiki labhimchindi.", "REAL"),
        ("Hyderabad lo TSRTC buses ki Hyderabad Metro tho integration chesaaru. Single card tho rendu transport modes use cheyabochu.", "REAL"),
        ("Visakhapatnam steel plant modernization kosam Rs 20,000 crore invest chestaami ani Union Minister chepparu.", "REAL"),
        ("Amaravati capital construction work meeru resume chesaaru. 3 lakh workers ki employment dorakutundi ani officials chepparu.", "REAL"),
        ("Telugu cinema industry mee 'RRR' film Oscar award ki select ayyindi. S.S. Rajamouli direct chesina ee film World famous ayyindi.", "REAL"),
        ("Telangana IT exports Rs 2 lakh crore cross chesayi. Hyderabad mee global IT hub ga maarindi ani report cheppundi.", "REAL"),
        ("AP lo Anna canteens 25 cities lo open chesaaru. Rs 5 ki full meal provide chestaaru ani CM Jagan chepparu.", "REAL"),
        ("JNTU Hyderabad mee research center lo made in India AI chip develop chesaaru. Qualcomm tho tie-up chesaaru.", "REAL"),
        ("Krishna river water sharing issue Supreme Court lo final verdict announce chesaaru. AP Telangana rendu states ki jarigindi.", "REAL"),
        
        # In Telugu script (will be transliterated)
        ("ఆంధ్రప్రదేశ్ ప్రభుత్వం రైతులకు రూ.13500 కోట్లు విడుదల చేసింది. YSR రైతు భరోసా పథకం కింద ప్రతి రైతు కుటుంబానికి నేరుగా డబ్బులు జమ అవుతాయి.", "REAL"),
        ("తెలంగాణ ప్రభుత్వం ప్రజలకు 200 యూనిట్లు ఉచిత కరెంట్ ఇస్తున్నది. గృహ వినియోగదారులందరికీ ఈ సౌకర్యం వర్తిస్తుంది.", "REAL"),
        ("హైదరాబాద్ నగరంలో కొత్త మెట్రో రైలు మార్గం నిర్మాణం ప్రారంభమైంది. 2026 నాటికి పూర్తవుతుందని అధికారులు తెలిపారు.", "REAL"),
        ("ఈనాడు పత్రిక నివేదిక ప్రకారం తెలంగాణలో IT రంగం 6 లక్షల ఉద్యోగాలు కల్పించింది.", "REAL"),
        ("విశాఖపట్నం పోర్టు సరుకు రవాణాలో జాతీయ రికార్డు సాధించింది. 10 కోట్ల టన్నుల కార్గో నిర్వహించింది.", "REAL"),
    ]
    
    telugu_fake = [
        # Fake Telugu news (romanized)
        ("BREAKING: AP government anni bank accounts freeze chesindi. July 15 to mundu mee money withdraw cheyandi. Ee message forward cheyandi!", "FAKE"),
        ("SHOCKING: Hyderabad lo secret underground tunnel lo gold discover chesaaru. Nizam treasure Rs 50,000 crore worth. Government hide chestaundi!", "FAKE"),
        ("ALERT: Telangana lo COVID new variant vastundi. Ee mutation 100% fatal. Turmeric water taagate pochaadu. Share cheyandi!", "FAKE"),
        ("VIRAL: KCR resign chestaadu ani secret sources chepputhunnaru. October 1 nundi Telangana President's rule lo untundi!", "FAKE"),
        ("URGENT: Government free laptops distributing chestuundi. Mee Aadhaar card tho register cheyandi. Only 1000 laptops left!", "FAKE"),
        
        # Telugu script fake news
        ("అత్యవసరం! వాట్సాప్ అకౌంట్లు జనవరి నుండి పెయిడ్ అవుతాయి. ఉచితంగా ఉంచుకోవాలంటే 30 మందికి ఫార్వార్డ్ చేయండి!", "FAKE"),
        ("షాకింగ్ న్యూస్: హైదరాబాద్‌లో కొత్త వైరస్ వ్యాపిస్తున్నది. 48 గంటల్లో మరణం సాధ్యమే. ఇంట్లో ఉండండి వెంటనే!", "FAKE"),
        ("బ్రేకింగ్: మోడీ ప్రభుత్వం తెలుగు రాష్ట్రాలను వేరే దేశానికి అమ్మేస్తున్నారు. రహస్య ఒప్పందం బయటపడింది!", "FAKE"),
        ("వైరల్: ఏపీ సీఎం అవినీతికి పట్టుబడ్డారు. 500 కోట్ల సొమ్ము స్విస్ బ్యాంకులో దాచారని CBI గుర్తించింది!", "FAKE"),
        ("అలర్ట్: చికెన్ తింటే కొత్త వ్యాధి వస్తుందని వైద్య నిపుణులు హెచ్చరిస్తున్నారు. ముందే చెప్పాం!", "FAKE"),
    ]
    
    texts = [t[0] for t in telugu_real + telugu_fake]
    labels = [t[1] for t in telugu_real + telugu_fake]
    df = pd.DataFrame({'text': texts, 'label': labels, 'language': 'te', 'source': 'telugu'})
    
    logger.info(f"Telugu dataset: {len(df)} samples ({df['label'].value_counts().to_dict()})")
    return df



def load_extended_dataset():
    """Extra training samples to boost model to 250+ total"""
    extra_real = [
        ("India achieved 100 gigawatt solar energy capacity milestone government announced renewable target ahead schedule", "REAL"),
        ("Supreme Court upheld electoral bonds scheme challenged petitioners constitutional bench verdict split", "REAL"),
        ("RBI issued guidelines digital lending apps mandatory disclosure interest rates fair practice code", "REAL"),
        ("Infosys Wipro TCS reported quarterly earnings revenue growth IT sector slowdown demand", "REAL"),
        ("India signed free trade agreement UK bilateral trade boost exports tariff reduction", "REAL"),
        ("AIIMS Delhi launched telemedicine platform rural patients specialist consultation remote areas", "REAL"),
        ("National Highway Authority completed 10000 km road construction financial year record infrastructure", "REAL"),
        ("India won gold medal Commonwealth Games athletics track field event national record", "REAL"),
        ("Sebi introduced new regulations mutual fund small cap mid cap rebalancing portfolio", "REAL"),
        ("Bengaluru water crisis Cauvery river allocation Karnataka Tamil Nadu dispute tribunal", "REAL"),
        ("PM inaugurated bullet train corridor Mumbai Ahmedabad construction progress update", "REAL"),
        ("India exported record wheat rice quantity food security global supply procurement", "REAL"),
        ("DRDO successfully tested hypersonic missile technology indigenous defense program", "REAL"),
        ("UPI transactions crossed 20 billion monthly record digital payments growth NPCI data", "REAL"),
        ("Central government launched PM Vishwakarma scheme artisans craftsmen skill development loan", "REAL"),
        ("Adani Ambani renewable energy investment solar wind power green hydrogen transition", "REAL"),
        ("India space station project Bharatiya Antariksha Station ISRO 2035 deadline announced", "REAL"),
        ("Cyclone warning issued Bay of Bengal coastal districts Andhra Pradesh Odisha evacuated", "REAL"),
        ("Lok Sabha passed Digital Personal Data Protection Bill privacy rights citizens obligations", "REAL"),
        ("IIT JEE Advanced results toppers marks cutoff category counseling schedule announced", "REAL"),
        ("Monsoon arrived Kerala June IMD forecast normal rainfall agricultural kharif season", "REAL"),
        ("Hyderabad airport handled record passengers domestic international flights growth", "REAL"),
        ("Telangana formation day celebrations Hyderabad June 2nd state government programs", "REAL"),
        ("Andhra Pradesh Amaravati High Court Jaganmohan Reddy capital construction resumed", "REAL"),
        ("YSRCP TDP political alliance election manifesto promises farmers welfare scheme AP", "REAL"),
        ("Eenadu Sakshi newspaper circulation Telugu readership media industry report", "REAL"),
        ("Pawan Kalyan Jana Sena party elected deputy chief minister Andhra Pradesh cabinet", "REAL"),
        ("KTR BRS party Telangana opposition leader Revanth Reddy congress government", "REAL"),
        ("Vizag Pharma City industrial corridor investment companies manufacturing hub AP", "REAL"),
        ("Sri Venkateswara temple Tirumala Tirupati pilgrim crowd darshan token system online", "REAL"),
    ]
    extra_fake = [
        ("URGENT forward this message government shutdown WhatsApp India January all accounts deleted", "FAKE"),
        ("SHOCKING Bill Gates microchips vaccines injected track location 5G towers activate chips", "FAKE"),
        ("VIRAL eating onion garlic prevents cancer doctors hide truth pharma lobby suppresses cure", "FAKE"),
        ("BREAKING Modi arrested corruption charges ED CBI raid secret Switzerland account exposed", "FAKE"),
        ("ALERT new law compulsory aadhaar link gold jewellery sell within 7 days or confiscated", "FAKE"),
        ("CONFIRMED China India secret treaty surrender Kashmir Ladakh 2025 government hiding public", "FAKE"),
        ("FREE MONEY government deposits Rs 10000 every citizen account PM Kisan apply before deadline", "FAKE"),
        ("PROVEN drinking turmeric milk cures diabetes permanently ancient vedic ayurveda suppressed", "FAKE"),
        ("EXPOSED Ambani Adani control entire country politicians puppet secret illuminati meeting", "FAKE"),
        ("BREAKING new virus deadlier COVID spreading Delhi Mumbai kills 24 hours lockdown imminent", "FAKE"),
        ("WARNING petrol price Rs 200 per litre next week government hiding stock up now sell car", "FAKE"),
        ("VIRAL WhatsApp Gold premium features exclusive invitation forward 50 contacts activate now", "FAKE"),
        ("CONFIRMED Pakistan nuclear missile launched India 3 cities destroyed government suppressing", "FAKE"),
        ("URGENT income tax department freeze all savings accounts PAN Aadhaar link deadline passed", "FAKE"),
        ("SHOCKING eating eggs causes coronavirus new research suppressed poultry industry paid scientists", "FAKE"),
        ("FREE laptop scheme students apply online government portal limited 500 available hurry", "FAKE"),
        ("BREAKING Rahul Gandhi convicted court sentenced prison Congress party banned India", "FAKE"),
        ("ALERT new traffic rule penalty Rs 50000 driving without specific certificate from January", "FAKE"),
        ("VIRAL ancient temple secret room gold treasure crores discovered government seized hiding", "FAKE"),
        ("EXPOSED fluoride water supply causes brain damage government intentional population control", "FAKE"),
        ("URGENT forward this SIM card stays active otherwise deactivated new TRAI rule implemented", "FAKE"),
        ("CONFIRMED moon landing fake NASA studio filmed Stanley Kubrick India must reveal truth", "FAKE"),
        ("BREAKING RBI demonetization all 2000 rupee notes invalid from tomorrow exchange immediately", "FAKE"),
        ("SHOCKING hospital harvesting organs poor patients anesthesia government covering up scandal", "FAKE"),
        ("FREE recharge trick dial USSD code get unlimited data Airtel Jio Vodafone share friends", "FAKE"),
        ("ALERT electricity bill waived permanently government scheme apply online portal limited", "FAKE"),
        ("VIRAL politician caught accepting bribe video proof CBI arrested mainstream media silent", "FAKE"),
        ("BREAKING India Pakistan war started border shelling army deployed emergency declared", "FAKE"),
        ("SHOCKING pesticide vegetables causing cancer farmers use chemicals banned countries import", "FAKE"),
        ("URGENT prayer chain forward message good luck 100 people or family member tragedy happens", "FAKE"),
    ]
    texts = [t[0] for t in extra_real + extra_fake]
    labels = [t[1] for t in extra_real + extra_fake]
    return pd.DataFrame({"text": texts, "label": labels, "language": "en", "source": "extended"})


# ============================================================
# STEP 2: PREPROCESSING
# ============================================================

class IndianNewsPreprocessor:
    """Preprocesses English and Telugu news text"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common Telugu stopwords (romanized)
        self.telugu_stopwords = {
            'undi', 'chesaaru', 'chepparu', 'ivvadam', 'meeru', 'mee', 
            'anni', 'lo', 'ki', 'ni', 'tho', 'kosam', 'ani', 'ee', 'aa',
            'oka', 'rendu', 'moodu', 'cheyandi', 'kinda', 'laki', 'valla'
        }
        
        # Try loading indic-nlp
        try:
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
            self.transliterator = UnicodeIndicTransliterator()
            self.indic_available = True
            logger.info("indic-nlp loaded successfully")
        except ImportError:
            self.indic_available = False
            logger.warning("indic-nlp not available. Telugu script will be romanized with basic mapping.")
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Telugu or English"""
        # Check for Telugu Unicode characters
        telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
        if telugu_chars > len(text) * 0.1:
            return 'te'
        
        # Check for Telugu romanization patterns
        telugu_markers = ['chepparu', 'chesaaru', 'undi', 'annaru', 'telugu', 
                         'hyderabad', 'andhra', 'telangana']
        text_lower = text.lower()
        if any(m in text_lower for m in telugu_markers):
            return 'te'
        
        if LANGDETECT_AVAILABLE:
            try:
                lang = detect(text)
                return lang if lang in ['en', 'te', 'hi'] else 'en'
            except:
                pass
        
        return 'en'
    
    def transliterate_telugu(self, text: str) -> str:
        """Convert Telugu script to Roman characters"""
        if self.indic_available:
            try:
                return self.transliterator.transliterate(text, 'te', 'en')
            except Exception:
                pass
        
        # Basic Telugu Unicode to romanization mapping
        telugu_map = {
            'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ii', 'ఉ': 'u',
            'ఊ': 'uu', 'ఎ': 'e', 'ఏ': 'ee', 'ఒ': 'o', 'ఓ': 'oo',
            'క': 'ka', 'గ': 'ga', 'చ': 'cha', 'జ': 'ja', 'ట': 'ta',
            'డ': 'da', 'త': 'tha', 'ద': 'dha', 'న': 'na', 'ప': 'pa',
            'బ': 'ba', 'మ': 'ma', 'య': 'ya', 'ర': 'ra', 'ల': 'la',
            'వ': 'va', 'శ': 'sha', 'స': 'sa', 'హ': 'ha', 'ళ': 'la',
            'ణ': 'na', 'ఱ': 'ra', 'ఫ': 'pha', 'ఖ': 'kha', 'ఘ': 'gha',
            'ఛ': 'cha', 'ఝ': 'jha', 'ఠ': 'tha', 'ఢ': 'dha', 'ధ': 'dha',
            'థ': 'tha', 'ఫ': 'fa', 'భ': 'bha', 'ష': 'sha',
            '్': '', 'ా': 'a', 'ి': 'i', 'ీ': 'ii', 'ు': 'u',
            'ూ': 'uu', 'ె': 'e', 'ే': 'ee', 'ొ': 'o', 'ో': 'oo',
            'ం': 'am', 'ః': 'ah', 'ఁ': 'an',
            'ళ': 'la', 'క్ష': 'ksha'
        }
        result = text
        for tel_char, roman in telugu_map.items():
            result = result.replace(tel_char, roman)
        # Remove remaining non-ASCII
        result = re.sub(r'[^\x00-\x7F]+', ' ', result)
        return result
    
    def clean_text(self, text: str, language: str = 'en') -> str:
        """Full preprocessing pipeline"""
        if not isinstance(text, str):
            return ""
        
        # Transliterate Telugu if needed
        if language == 'te':
            has_telugu_script = any('\u0C00' <= c <= '\u0C7F' for c in text)
            if has_telugu_script:
                text = self.transliterate_telugu(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = text.split()
        
        # Remove stopwords (combined English + Telugu)
        all_stops = self.stop_words | self.telugu_stopwords
        tokens = [t for t in tokens if t not in all_stops and len(t) > 2]
        
        # Stem
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def extract_features(self, text: str) -> dict:
        """Extract handcrafted features for fake news detection"""
        features = {}
        
        # Urgency indicators (common in fake news)
        urgency_words = ['breaking', 'urgent', 'alert', 'viral', 'shocking', 
                        'exposed', 'confirmed', 'secret', 'forward', 'share',
                        'atyanutam', 'shocking', 'breking', 'vayiral']
        text_lower = text.lower()
        features['urgency_score'] = sum(1 for w in urgency_words if w in text_lower)
        
        # Exclamation marks
        features['exclamation_count'] = text.count('!')
        
        # ALL CAPS ratio
        words = text.split()
        if words:
            caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
            features['caps_ratio'] = caps_ratio
        else:
            features['caps_ratio'] = 0
        
        # Question marks
        features['question_count'] = text.count('?')
        
        # Numbers (fake news often uses specific big numbers)
        features['number_count'] = len(re.findall(r'\d+', text))
        
        # Text length
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        
        # Fake news keywords for India context
        indian_fake_indicators = [
            'forward', 'share', 'crores', 'scheme', 'free', 'apply', 'whatsapp',
            'deleted', 'hidden', 'suppressed', 'secret', 'conspiracy'
        ]
        features['fake_indicators'] = sum(1 for w in indian_fake_indicators if w in text_lower)
        
        return features
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataframe"""
        logger.info(f"Preprocessing {len(df)} samples...")
        
        if 'language' not in df.columns:
            df['language'] = df['text'].apply(self.detect_language)
        
        df['clean_text'] = df.apply(
            lambda row: self.clean_text(row['text'], row.get('language', 'en')), 
            axis=1
        )
        
        # Normalize labels
        df['label'] = df['label'].str.upper()
        df['label_binary'] = (df['label'] == 'FAKE').astype(int)
        
        # Remove empty rows
        df = df[df['clean_text'].str.len() > 5].reset_index(drop=True)
        
        logger.info(f"After preprocessing: {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df


# ============================================================
# STEP 3: MODEL TRAINING
# ============================================================

class FakeNewsDetector:
    """
    Multilingual Fake News Detector for Indian news
    Supports: English (Indian context) + Telugu
    """
    
    def __init__(self):
        self.preprocessor = IndianNewsPreprocessor()
        self.model = None
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.model_version = "2.0-indian"
        self.metrics = {}
    
    def build_pipeline(self):
        """Build sklearn pipeline with TF-IDF + Ensemble"""
        
        # TF-IDF with ngrams for better feature extraction
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),          # Unigrams, bigrams, trigrams
            sublinear_tf=True,            # Log normalization
            min_df=2,                     # Minimum document frequency
            max_df=0.95,                  # Maximum document frequency
            analyzer='word',
            strip_accents='unicode',
            token_pattern=r'\b[a-z]{2,}\b'
        )
        
        # Ensemble: LogReg + RandomForest + NaiveBayes
        lr = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        )
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        nb = MultinomialNB(alpha=0.1)
        
        # Soft voting ensemble
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('nb', nb)],
            voting='soft'
        )
        
        return self
    
    def train(self, df: pd.DataFrame):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Preprocess
        df = self.preprocessor.preprocess_dataframe(df)
        
        X = df['clean_text'].values
        y = df['label_binary'].values
        
        # Build pipeline
        self.build_pipeline()
        
        # Vectorize
        logger.info("Vectorizing text (TF-IDF)...")
        X_vec = self.vectorizer.fit_transform(X)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        logger.info("Training ensemble model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'roc_auc': float(roc_auc_score(y_test, y_prob)),
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'model_version': self.model_version
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"MODEL PERFORMANCE")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {self.metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:  {self.metrics['roc_auc']:.4f}")
        logger.info(f"\nDetailed Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
        
        # Cross-validation — use 3-fold for small datasets, 5 for large
        n_samples = X_vec.shape[0]
        cv_folds = 3 if n_samples < 100 else 5
        logger.info(f"Running {cv_folds}-fold cross-validation ({n_samples} samples)...")
        cv_scores = cross_val_score(self.model, X_vec, y, cv=cv_folds, scoring='f1')
        logger.info(f"CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        self.metrics['cv_f1_mean'] = float(cv_scores.mean())
        self.metrics['cv_f1_std'] = float(cv_scores.std())
        
        return self
    
    def predict(self, text: str) -> dict:
        """Predict if a news article is fake"""
        lang = self.preprocessor.detect_language(text)
        clean = self.preprocessor.clean_text(text, lang)
        
        vec = self.vectorizer.transform([clean])
        prob = self.model.predict_proba(vec)[0]
        pred = self.model.predict(vec)[0]
        
        return {
            'label': 'FAKE' if pred == 1 else 'REAL',
            'fake_probability': float(prob[1]),
            'real_probability': float(prob[0]),
            'language': lang,
            'confidence': float(max(prob))
        }
    
    def save(self, model_path: str = 'multilingual_model.pkl', 
             vectorizer_path: str = 'vectorizer.pkl',
             metrics_path: str = 'model_metrics.json'):
        """Save trained model"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Model saved: {model_path}, {vectorizer_path}")
    
    @classmethod
    def load(cls, model_path: str = 'multilingual_model.pkl',
             vectorizer_path: str = 'vectorizer.pkl'):
        """Load trained model"""
        detector = cls()
        detector.model = joblib.load(model_path)
        detector.vectorizer = joblib.load(vectorizer_path)
        try:
            with open('model_metrics.json') as f:
                detector.metrics = json.load(f)
        except:
            detector.metrics = {}
        logger.info("Model loaded successfully")
        return detector


# ============================================================
# STEP 4: MAIN TRAINING EXECUTION
# ============================================================

if __name__ == "__main__":
    print("🇮🇳 Indian Fake News Detector - Training Pipeline")
    print("=" * 60)
    
    # Load datasets
    print("\n📊 Loading datasets...")
    all_dfs = []
    
    # English Indian news datasets
    datasets = load_fake_news_india_dataset()
    for name, df in datasets:
        if isinstance(df, pd.DataFrame) and 'text' in df.columns:
            all_dfs.append(df[['text', 'label']].assign(
                language='en', source=name
            ))
            print(f"  ✓ {name}: {len(df)} samples")
    
    # Telugu dataset
    df_telugu = load_telugu_dataset()
    all_dfs.append(df_telugu[['text', 'label', 'language']].assign(source='telugu'))
    print(f"  ✓ Telugu: {len(df_telugu)} samples")

    # Extended dataset (more English samples for better accuracy)
    df_ext = load_extended_dataset()
    all_dfs.append(df_ext)
    print(f"  ✓ Extended: {len(df_ext)} samples")
    
    # Combine all datasets
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Handle different label formats
    df_all['label'] = df_all['label'].astype(str).str.upper()
    label_map = {'1': 'FAKE', '0': 'REAL', 'TRUE': 'REAL', 'FALSE': 'FAKE'}
    df_all['label'] = df_all['label'].replace(label_map)
    
    print(f"\n📈 Total dataset: {len(df_all)} samples")
    print(f"   Distribution: {df_all['label'].value_counts().to_dict()}")
    
    # Train model
    print("\n🤖 Training model...")
    detector = FakeNewsDetector()
    detector.train(df_all)
    
    # Save model
    print("\n💾 Saving model...")
    detector.save()
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_cases = [
        "ISRO successfully launches GSAT-20 satellite providing broadband to remote areas of India",
        "BREAKING: Government giving free 5G phone to all! Apply now before December 31!",
        "Andhra Pradesh budget allocates Rs 5000 crore for education sector development",
        "VIRAL: Eating cow urine cures cancer! Share before government deletes this!!!",
        "Hyderabad lo kotta metro rail start ayyindi. 2 lakh mandiki upyogapadutundi.",
    ]
    
    print("\nTest Predictions:")
    print("-" * 70)
    for text in test_cases:
        result = detector.predict(text)
        emoji = "🔴 FAKE" if result['label'] == 'FAKE' else "🟢 REAL"
        lang = "Telugu" if result['language'] == 'te' else "English"
        print(f"{emoji} [{lang}] ({result['fake_probability']:.0%} fake confidence)")
        print(f"  Text: {text[:70]}...")
        print()
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved as: multilingual_model.pkl")
    print(f"   F1 Score: {detector.metrics.get('f1_score', 'N/A'):.4f}")
    print(f"   Use in FastAPI: joblib.load('multilingual_model.pkl')")
