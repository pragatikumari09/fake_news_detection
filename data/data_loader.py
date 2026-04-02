"""
data/data_loader.py
-------------------
Handles:
  1. Loading the Kaggle Fake/Real News CSVs (or any labelled CSV).
  2. Generating a synthetic demo dataset so the pipeline runs end-to-end
     even without the real Kaggle files.
  3. Merging, shuffling, and saving the consolidated CSV.

Kaggle dataset:  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  - Fake.csv  (columns: title, text, subject, date)  → label = 0
  - True.csv  (columns: title, text, subject, date)  → label = 1
"""

import os
import sys
import random
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAKE_CSV, REAL_CSV, MERGED_CSV, RANDOM_SEED

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Synthetic article templates ───────────────────────────────────────────────

_FAKE_TEMPLATES = [
    ("SHOCKING: {person} EXPOSED for secret {scandal}!",
     "In a bombshell revelation that mainstream media won't tell you, {person} has "
     "been secretly involved in a massive {scandal} that affects millions. "
     "Anonymous sources close to the investigation say the conspiracy goes all the "
     "way to the top. Share this before it gets deleted! The truth is finally out."),

    ("BREAKING: Scientists BANNED for revealing {topic} cure",
     "A group of rogue scientists claims to have discovered a miracle cure for "
     "{topic} but says pharmaceutical companies are suppressing the findings. "
     "No peer-reviewed evidence exists, yet social media is ablaze with the "
     "unverified claim. Experts urge caution but the posts keep going viral."),

    ("ALERT: Government hiding {event} from the public",
     "Insiders leaked documents allegedly showing that officials knew about {event} "
     "months in advance but chose to cover it up. The documents, whose authenticity "
     "cannot be verified, are spreading rapidly on fringe websites. Critics call "
     "them fabricated; believers say the denial is itself proof of the cover-up."),
]

_REAL_TEMPLATES = [
    ("{agency} reports {pct}% rise in {sector} employment last quarter",
     "Official statistics released on {day} show that {sector} employment grew by "
     "{pct} percent in the last quarter, exceeding analyst forecasts of {low} "
     "percent. The {agency} attributed the gains to easing supply-chain pressures "
     "and increased consumer spending. Economists warn, however, that inflationary "
     "pressures could dampen growth in coming months."),

    ("Study finds {drug} effective against {disease} in clinical trial",
     "A phase-3 clinical trial published in The Lancet found that {drug} reduced "
     "{disease} mortality by {pct} percent compared with placebo. The randomised, "
     "double-blind trial enrolled {n} participants across 12 countries. Researchers "
     "cautioned that long-term side-effect data are still being collected and that "
     "broader approval would require regulatory review."),

    ("{country} parliament passes landmark {policy} bill",
     "Lawmakers in {country} voted {yes} to {no} on {day} to adopt the {policy} "
     "bill, which had been debated for three years. Supporters say the legislation "
     "will strengthen protections for citizens, while opposition groups have "
     "pledged to challenge certain provisions in court. The bill now awaits "
     "presidential assent before it becomes law."),
]

_PEOPLE    = ["Senator Johnson", "CEO Mark Fields", "Dr. Elena Voss", "General Hayes"]
_SCANDALS  = ["money-laundering scheme", "secret bioweapon program", "election fraud ring"]
_TOPICS    = ["cancer", "diabetes", "Alzheimer's", "COVID-19"]
_EVENTS    = ["the economic collapse", "the pandemic outbreak", "the climate disaster"]
_AGENCIES  = ["The Labor Department", "The Bureau of Statistics", "The Federal Reserve"]
_SECTORS   = ["technology", "healthcare", "manufacturing", "retail"]
_DRUGS     = ["Vaxicline", "Remizumab", "Therapeutol"]
_DISEASES  = ["lung cancer", "diabetes", "heart failure"]
_COUNTRIES = ["Germany", "Brazil", "Canada", "South Korea"]
_POLICIES  = ["data-privacy", "carbon-tax", "universal healthcare", "electoral reform"]
_DAYS      = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _fake_article() -> dict:
    tmpl = random.choice(_FAKE_TEMPLATES)
    title_t, body_t = tmpl
    kw = dict(
        person=random.choice(_PEOPLE),
        scandal=random.choice(_SCANDALS),
        topic=random.choice(_TOPICS),
        event=random.choice(_EVENTS),
    )
    return {
        "title": title_t.format(**kw),
        "text":  body_t.format(**kw),
        "label": 0,          # 0 = FAKE
        "subject": "politics",
    }


def _real_article() -> dict:
    tmpl = random.choice(_REAL_TEMPLATES)
    title_t, body_t = tmpl
    pct  = round(random.uniform(1.5, 8.5), 1)
    low  = round(pct - random.uniform(0.5, 2.0), 1)
    kw = dict(
        agency   = random.choice(_AGENCIES),
        pct      = pct,
        low      = low,
        sector   = random.choice(_SECTORS),
        drug     = random.choice(_DRUGS),
        disease  = random.choice(_DISEASES),
        n        = random.randint(500, 5000),
        country  = random.choice(_COUNTRIES),
        policy   = random.choice(_POLICIES),
        day      = random.choice(_DAYS),
        yes      = random.randint(150, 300),
        no       = random.randint(50, 149),
    )
    return {
        "title": title_t.format(**kw),
        "text":  body_t.format(**kw),
        "label": 1,          # 1 = REAL
        "subject": "worldnews",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_fake: int = 5000, n_real: int = 5000) -> pd.DataFrame:
    """
    Generate a balanced synthetic fake/real news dataset.
    Useful for smoke-tests and demos when the Kaggle files are unavailable.
    """
    print(f"[DataLoader] Generating {n_fake} fake + {n_real} real synthetic articles …")
    records = [_fake_article() for _ in range(n_fake)] + \
              [_real_article() for _ in range(n_real)]
    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[DataLoader] Synthetic dataset shape: {df.shape}")
    return df


def load_kaggle_dataset() -> pd.DataFrame:
    """
    Load the Kaggle Fake-and-Real-News CSVs.
    Expected columns: title, text, subject, date.
    Returns a unified DataFrame with an integer `label` column (0=FAKE, 1=REAL).
    """
    if not FAKE_CSV.exists() or not REAL_CSV.exists():
        raise FileNotFoundError(
            f"Kaggle CSVs not found.\n"
            f"  Expected: {FAKE_CSV}\n"
            f"            {REAL_CSV}\n"
            "Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )
    print("[DataLoader] Loading Kaggle dataset …")
    fake = pd.read_csv(FAKE_CSV)
    real = pd.read_csv(REAL_CSV)
    fake["label"] = 0
    real["label"] = 1
    df = pd.concat([fake, real], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[DataLoader] Kaggle dataset shape: {df.shape}")
    return df


def load_dataset(synthetic_fallback: bool = True,
                 n_synthetic: int = 10_000) -> pd.DataFrame:
    """
    Master loader: tries Kaggle files first; falls back to synthetic data.

    Parameters
    ----------
    synthetic_fallback : bool
        If True and Kaggle files are missing, generate synthetic data.
    n_synthetic : int
        Total rows for synthetic dataset (split 50/50).

    Returns
    -------
    pd.DataFrame  with columns: title, text, label, subject
    """
    try:
        df = load_kaggle_dataset()
    except FileNotFoundError as e:
        if not synthetic_fallback:
            raise
        print(f"[DataLoader] WARNING: {e}")
        print("[DataLoader] Falling back to synthetic dataset …")
        half = n_synthetic // 2
        df = generate_synthetic_dataset(n_fake=half, n_real=half)

    # Standardise column names
    for col in ["title", "text"]:
        if col not in df.columns:
            df[col] = ""
    df["label"] = df["label"].astype(int)

    # Save merged copy
    df.to_csv(MERGED_CSV, index=False)
    print(f"[DataLoader] Merged dataset saved → {MERGED_CSV}")
    return df


if __name__ == "__main__":
    df = load_dataset()
    print(df["label"].value_counts())
    print(df.head(3))
