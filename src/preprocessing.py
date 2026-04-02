"""
src/preprocessing.py  — fully offline-safe NLP pipeline
"""

import re
import sys
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SENSATIONAL_WORDS

logger = logging.getLogger(__name__)

# ── Built-in English stopwords (no corpus download needed) ───────────────────
_STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn","said","says","also","would","could","may","might",
    "shall","us","via","per","get","got","use","used","using","one","two",
    "three","four","five","six","seven","eight","nine","ten","new","old",
    "like","even","well","back","still","way","take","come","since","around",
    "going","make","made","say","know","see","look","think","go","want","give",
    "need","find","tell","ask","seem","feel","leave","call","keep","let",
    "show","hear","play","run","move","live","happen","believe","hold","bring",
    "write","provide","set","put","however","although","though","while",
    "therefore","thus","hence","indeed","moreover","furthermore","nevertheless",
}

# ── Regex-based lemmatiser (no wordnet corpus needed) ────────────────────────
_LEMMA_RULES = [
    (re.compile(r"ies$"),   "y"),
    (re.compile(r"ied$"),   "y"),
    (re.compile(r"ness$"),  ""),
    (re.compile(r"ment$"),  ""),
    (re.compile(r"tions$"), "te"),
    (re.compile(r"tion$"),  "te"),
    (re.compile(r"ings$"),  ""),
    (re.compile(r"ing$"),   ""),
    (re.compile(r"edly$"),  ""),
    (re.compile(r"ed$"),    ""),
    (re.compile(r"ers$"),   ""),
    (re.compile(r"er$"),    ""),
    (re.compile(r"est$"),   ""),
    (re.compile(r"ly$"),    ""),
    (re.compile(r"s$"),     ""),
]

def _lemmatize(token: str) -> str:
    for pat, rep in _LEMMA_RULES:
        cand = pat.sub(rep, token)
        if cand != token and len(cand) >= 3:
            return cand
    return token

# ── Compile-once regexes ──────────────────────────────────────────────────────
_HTML_RE    = re.compile(r"<[^>]+>")
_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_SPECIAL_RE = re.compile(r"[^a-z\s]")
_MULTI_SPC  = re.compile(r"\s+")
_SENSATIONAL = set(SENSATIONAL_WORDS)

tqdm.pandas()


# ── Character-level cleaners ──────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _SPECIAL_RE.sub(" ", text)
    return _MULTI_SPC.sub(" ", text).strip()


# ── Token-level processing ────────────────────────────────────────────────────

def tokenise(text: str) -> List[str]:
    return text.split()

def remove_stopwords(tokens: List[str], min_len: int = 2) -> List[str]:
    return [t for t in tokens if t not in _STOP_WORDS and len(t) >= min_len]

def lemmatise(tokens: List[str]) -> List[str]:
    return [_lemmatize(t) for t in tokens]


def full_pipeline(raw_text: str, method: str = "lemmatise") -> str:
    """
    Full preprocessing: clean → tokenise → remove stopwords → lemmatise.
    Returns a single string ready for TF-IDF vectorisation.
    """
    cleaned = clean_text(raw_text)
    tokens  = tokenise(cleaned)
    tokens  = remove_stopwords(tokens)
    tokens  = lemmatise(tokens)
    return " ".join(tokens)


# ── Meta-features ─────────────────────────────────────────────────────────────

def compute_meta_features(raw_text: str) -> dict:
    words = raw_text.split()
    char_len     = len(raw_text)
    word_count   = len(words)
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    try:
        blob = TextBlob(raw_text)
        polarity     = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)
    except Exception:
        polarity = subjectivity = 0.0

    lower_set         = {w.lower().strip(".,!?") for w in words}
    sensational_count = len(_SENSATIONAL & lower_set)
    url_count         = len(_URL_RE.findall(raw_text))
    exclamation_count = raw_text.count("!")
    question_count    = raw_text.count("?")
    capital_ratio     = sum(1 for c in raw_text if c.isupper()) / max(char_len, 1)

    return {
        "char_len":               char_len,
        "word_count":             word_count,
        "avg_word_len":           avg_word_len,
        "sentiment_polarity":     polarity,
        "sentiment_subjectivity": subjectivity,
        "sensational_count":      sensational_count,
        "url_count":              url_count,
        "exclamation_count":      exclamation_count,
        "question_count":         question_count,
        "capital_ratio":          capital_ratio,
    }


# ── DataFrame pipeline ────────────────────────────────────────────────────────

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col:  str = "text",
    title_col: str = "title",
    label_col: str = "label",
    method:    str = "lemmatise",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    logger.info("Preprocessing %d rows …", len(df))
    df = df.copy()
    df[text_col]  = df[text_col].fillna("").astype(str)
    df[title_col] = df[title_col].fillna("").astype(str)

    # Title weighted 2× to boost headline signal
    df["raw_combined"] = df[title_col] + " " + df[title_col] + " " + df[text_col]

    print("[Preprocessor] Running NLP pipeline …")
    df["cleaned_text"] = df["raw_combined"].progress_apply(full_pipeline)

    print("[Preprocessor] Computing meta-features …")
    meta_df = df["raw_combined"].progress_apply(
        lambda t: pd.Series(compute_meta_features(t))
    )
    df = pd.concat([df, meta_df], axis=1)

    mask = df["cleaned_text"].str.strip() == ""
    if mask.sum():
        logger.warning("Dropping %d empty rows", mask.sum())
        df = df[~mask].reset_index(drop=True)

    X_text = df["cleaned_text"].values
    y      = df[label_col].astype(int).values
    logger.info("Done. X=%s  y=%s", X_text.shape, y.shape)
    return df, X_text, y


if __name__ == "__main__":
    sample = (
        "<b>SHOCKING!</b> Scientists BANNED by Big Pharma for revealing "
        "secret cancer cure at http://fakecure.com — share before DELETED!!!"
    )
    print("RAW  :", sample[:80])
    print("CLEAN:", full_pipeline(sample))
    print("META :", compute_meta_features(sample))
