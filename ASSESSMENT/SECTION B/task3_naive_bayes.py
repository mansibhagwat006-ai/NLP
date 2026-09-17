"""
Task 3: Complaint Category Classifier – Naive Bayes
"""
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

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
    ("My order is very late","Delivery"),("The driver has not arrived","Delivery"),
    ("My delivery is delayed again","Delivery"),("Where is my food order","Delivery"),
    ("The courier delivered to the wrong address","Delivery"),("My order never arrived","Delivery"),
    ("The delivery driver is stuck","Delivery"),("It took two hours to deliver","Delivery"),
    ("My pizza arrived cold","Food Quality"),("The food tastes stale","Food Quality"),
    ("My meal was undercooked","Food Quality"),("The burger was cold and soggy","Food Quality"),
    ("The food portion was disappointing","Food Quality"),("My fries were too salty","Food Quality"),
    ("The chicken was raw","Food Quality"),("The meal quality was poor","Food Quality"),
    ("The app keeps crashing","App"),("I cannot log into the app","App"),
    ("The payment page is not working","App"),("The app shows an error","App"),
    ("I cannot place an order","App"),("The checkout button is broken","App"),
    ("The app is very slow","App"),("My coupon is not applying","App"),
]

texts = [preprocess_review(x[0]) for x in data]
labels = [x[1] for x in data]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.20, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer()
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(Xtr, y_train)
pred = model.predict(Xte)

print("CLASSIFICATION REPORT")
print(classification_report(y_test, pred, zero_division=0))

print("CONFUSION MATRIX")
cm = confusion_matrix(y_test, pred, labels=["Delivery", "Food Quality", "App"])
print(pd.DataFrame(cm, index=["Delivery", "Food Quality", "App"],
                   columns=["Delivery", "Food Quality", "App"]))
