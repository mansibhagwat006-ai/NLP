"""
Task 2: TF-IDF Vectoriser for Menu Reviews
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "delicious spicy chicken arrived hot",
    "cold chicken arrived late",
    "fresh pizza with tasty cheese",
    "pizza was cold and soggy",
    "delivery was quick and friendly",
    "the meal was okay and average",
    "excellent burger with fresh ingredients",
    "slow delivery and disappointing meal",
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names_out()

df = pd.DataFrame(X.toarray(), columns=features,
                  index=[f"Review {i}" for i in range(1, len(corpus)+1)])

print("FULL TF-IDF MATRIX")
print(df.round(4))

print("\nTOP 3 TERMS PER REVIEW")
for idx, row in df.iterrows():
    top = row.sort_values(ascending=False).head(3)
    print(f"\n{idx}:")
    for word, score in top.items():
        print(f"  {word}: {score:.4f}")
