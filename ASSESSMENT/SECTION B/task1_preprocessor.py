
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def prepare_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

prepare_nltk()
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

def preprocess_review(text: str) -> str:
    """Lowercase, remove punctuation, normalise whitespace, tokenize,
    remove English stopwords, stem, and return a space-joined string."""
    if not isinstance(text, str):
        raise TypeError("review must be a string")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    stems = [STEMMER.stem(t) for t in tokens]
    return " ".join(stems)

if __name__ == "__main__":
    samples = [
        "The pizza was AMAZING!!! It arrived hot and fresh.",
        "My order was late, cold, and disappointing.",
        "The app is easy to use, but the coupon did not work."
    ]
    for i, review in enumerate(samples, 1):
        print(f"Review {i}: {review}")
        print(f"Processed: {preprocess_review(review)}\n")
