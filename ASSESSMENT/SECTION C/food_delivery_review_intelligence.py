"""
Section C – Mini Capstone
Food Delivery Review Intelligence System

Run:
    python food_delivery_review_intelligence.py

The program trains two text classifiers from the built-in labelled examples:
- Logistic Regression for Positive/Negative sentiment
- Multinomial Naive Bayes for Delivery/Food Quality/App/General issue category

Every user review is preprocessed before being stored/classified.
"""
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

def preprocess(text):
    if not isinstance(text, str):
        raise TypeError("Review must be text.")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(STEMMER.stem(t) for t in tokens)

# Training examples. More than the minimum required samples are supplied.
TRAINING = [
("food was delicious and fresh", "Positive", "Food Quality"),
("delivery was fast and perfect", "Positive", "Delivery"),
("amazing meal and friendly service", "Positive", "General"),
("order arrived hot", "Positive", "Delivery"),
("loved tasty pizza", "Positive", "Food Quality"),
("excellent food quality", "Positive", "Food Quality"),
("driver was polite", "Positive", "Delivery"),
("everything was great", "Positive", "General"),
("app worked smoothly", "Positive", "App"),
("very happy with order", "Positive", "General"),
("fresh food and quick delivery", "Positive", "Delivery"),
("meal was excellent", "Positive", "Food Quality"),
("food was cold and terrible", "Negative", "Food Quality"),
("order arrived very late", "Negative", "Delivery"),
("meal tasted stale", "Negative", "Food Quality"),
("delivery was disappointing", "Negative", "Delivery"),
("app crashed during checkout", "Negative", "App"),
("payment failed repeatedly", "Negative", "App"),
("food was undercooked", "Negative", "Food Quality"),
("pizza was soggy", "Negative", "Food Quality"),
("driver delivered wrong order", "Negative", "Delivery"),
("service was awful", "Negative", "General"),
("app is slow and broken", "Negative", "App"),
("food never arrived", "Negative", "Delivery"),
("order was completely wrong", "Negative", "Delivery"),
("very bad experience", "Negative", "General"),
("coupon is not applying", "Negative", "App"),
("cannot log into the app", "Negative", "App"),
("quick delivery and tasty meal", "Positive", "Delivery"),
("great app and easy checkout", "Positive", "App"),
("friendly driver and hot food", "Positive", "Delivery"),
("fresh burger was wonderful", "Positive", "Food Quality"),
]

MIN_TRAINING_SAMPLES = 10

sentiment_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

category_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

def train_models():
    if len(TRAINING) < MIN_TRAINING_SAMPLES:
        return False
    texts = [preprocess(x[0]) for x in TRAINING]
    sentiment = [x[1] for x in TRAINING]
    categories = [x[2] for x in TRAINING]
    sentiment_model.fit(texts, sentiment)
    category_model.fit(texts, categories)
    return True

reviews = []

def classify_review(raw_review):
    if len(TRAINING) < MIN_TRAINING_SAMPLES:
        print(f"Not enough training samples. Add at least {MIN_TRAINING_SAMPLES} training samples.")
        return None
    cleaned = preprocess(raw_review)
    if not cleaned:
        print("The review contains no usable words after preprocessing.")
        return None
    sentiment = sentiment_model.predict([cleaned])[0]
    category = category_model.predict([cleaned])[0]
    return sentiment, category, cleaned

def add_review():
    raw = input("Enter customer review: ").strip()
    if not raw:
        print("Review cannot be empty.")
        return
    result = classify_review(raw)
    if result is None:
        return
    sentiment, category, cleaned = result
    reviews.append({
        "raw": raw,
        "processed": cleaned,
        "sentiment": sentiment,
        "category": category
    })
    print("Review added successfully.")

def classify_only():
    raw = input("Enter review to classify: ").strip()
    if not raw:
        print("Review cannot be empty.")
        return
    result = classify_review(raw)
    if result:
        sentiment, category, cleaned = result
        print(f"Processed text: {cleaned}")
        print(f"Predicted sentiment: {sentiment}")
        print(f"Predicted issue category: {category}")

def summary_report():
    print("\n===== SUMMARY REPORT =====")
    print(f"Total reviews added: {len(reviews)}")
    sentiments = Counter(r["sentiment"] for r in reviews)
    categories = Counter(r["category"] for r in reviews)
    print("\nSentiment counts:")
    for label in ["Positive", "Negative"]:
        print(f"  {label}: {sentiments.get(label, 0)}")
    print("\nIssue category counts:")
    for label in ["Delivery", "Food Quality", "App", "General"]:
        print(f"  {label}: {categories.get(label, 0)}")

    words = []
    for r in reviews:
        words.extend(r["processed"].split())
    print("\n5 most frequent content words:")
    for word, count in Counter(words).most_common(5):
        print(f"  {word}: {count}")

def main():
    if not train_models():
        print("Training data is below the minimum required amount. Classification is disabled.")
        return

    while True:
        print("\n===== FOOD DELIVERY REVIEW INTELLIGENCE SYSTEM =====")
        print("1. Add a new review")
        print("2. Classify a review")
        print("3. View summary report")
        print("4. Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            add_review()
        elif choice == "2":
            classify_only()
        elif choice == "3":
            summary_report()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
