import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

data = pd.concat([fake, true])
data = data[["title", "text", "label"]]

data.to_csv("news.csv", index=False)

print("news.csv created successfully")
