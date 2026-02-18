import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading datasets...")

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

print("Fake news:", len(fake))
print("Real news:", len(real))

data = pd.concat([fake, real], ignore_index=True)

data["content"] = data["title"].astype(str) + " " + data["text"].astype(str)

X = data["content"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Training complete. Model saved.")
