"""
Task 4: Sentiment Analysis Pipeline – Model Comparison
"""
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

def preprocess_review(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(STEMMER.stem(t) for t in tokens)

data = [
    ("The food was delicious and fresh","Positive"),("Delivery was fast and perfect","Positive"),
    ("Amazing meal and friendly service","Positive"),("The order arrived hot","Positive"),
    ("I loved the tasty pizza","Positive"),("Excellent food quality","Positive"),
    ("The driver was polite","Positive"),("Everything was great","Positive"),
    ("The app worked smoothly","Positive"),("Very happy with my order","Positive"),
    ("Fresh food and quick delivery","Positive"),("The meal was excellent","Positive"),
    ("I will order again","Positive"),("The service was wonderful","Positive"),
    ("My order arrived perfectly","Positive"),
    ("The food was cold and terrible","Negative"),("My order arrived very late","Negative"),
    ("The meal tasted stale","Negative"),("The delivery was disappointing","Negative"),
    ("The app crashed during checkout","Negative"),("The payment failed repeatedly","Negative"),
    ("The food was undercooked","Negative"),("My pizza was soggy","Negative"),
    ("The driver delivered the wrong order","Negative"),("The service was awful","Negative"),
    ("I hated the meal","Negative"),("The app is slow and broken","Negative"),
    ("My food never arrived","Negative"),("The order was completely wrong","Negative"),
    ("Very bad experience","Negative"),
]

X = [preprocess_review(x[0]) for x in data]
y = [x[1] for x in data]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

results = []
for name, pipeline in [("Naive Bayes", nb_pipeline), ("Logistic Regression", lr_pipeline)]:
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, pos_label="Positive", zero_division=0),
        "Recall": recall_score(y_test, pred, pos_label="Positive", zero_division=0),
        "F1": f1_score(y_test, pred, pos_label="Positive", zero_division=0),
    })

comparison = pd.DataFrame(results)
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Deployment choice: Logistic Regression is selected here because it provides a
# strong linear classifier for TF-IDF features and its decision scores are useful
# for comparing confidence; this choice should be validated on a larger dataset.
