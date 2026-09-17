"""
AI ORIGINAL VERSION – Section D Step 1
This is intentionally preserved as the "original AI-generated" version.
Known limitation/bug: it uses a non-stratified split. On a small or imbalanced
dataset, the test split can contain only one class, making evaluation misleading.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
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

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed)

# Original version: no stratify argument.
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.20, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

for i in range(3):
    new_review = input(f"Enter new review {i+1}: ")
    processed_review = preprocess(new_review)
    prediction = model.predict(vectorizer.transform([processed_review]))
    print("Prediction:", prediction[0])
