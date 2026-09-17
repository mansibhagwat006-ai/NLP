"""
AI CORRECTED VERSION – Section D Step 2
Correction made after testing/reviewing the original:
1. Added stratify=labels so both sentiment classes are represented in the
   train/test split when possible.
2. Added a minimum-data/class validation and empty-review handling.
3. Uses a fixed random_state for reproducible results.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(stemmer.stem(word) for word in tokens)

reviews = [
    "The food was delicious and fresh", "Delivery was fast and perfect",
    "Amazing meal and friendly service", "The order arrived hot",
    "I loved the tasty pizza", "Excellent food quality",
    "The driver was polite", "Everything was great",
    "The food was cold and terrible", "My order arrived very late",
    "The meal tasted stale", "The delivery was disappointing",
    "The app crashed during checkout", "The payment failed repeatedly",
    "The food was undercooked", "My pizza was soggy",
    "The driver delivered the wrong order", "The service was awful",
    "I hated the meal", "The app is slow and broken",
]

labels = [
    "Positive","Positive","Positive","Positive","Positive","Positive","Positive","Positive",
    "Negative","Negative","Negative","Negative","Negative","Negative","Negative","Negative",
    "Negative","Negative","Negative","Negative"
]

processed = [preprocess(r) for r in reviews]

if len(processed) < 10 or len(set(labels)) < 2:
    raise ValueError("At least 10 reviews and two sentiment classes are required.")

# Corrected version: stratified split keeps class proportions more stable.
X_train, X_test, y_train, y_test = train_test_split(
    processed, labels, test_size=0.20, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

print("\nEnter three new reviews:")
for i in range(3):
    while True:
        new_review = input(f"Review {i+1}: ").strip()
        processed_review = preprocess(new_review)
        if processed_review:
            break
        print("Please enter a review containing at least one usable word.")
    prediction = model.predict(vectorizer.transform([processed_review]))
    print("Prediction:", prediction[0])
