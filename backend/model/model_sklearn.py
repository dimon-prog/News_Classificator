from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import log_loss
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys

csv.field_size_limit(sys.maxsize)
dataset = pd.read_csv("WELFake_Dataset.csv", engine="python")
dataset = dataset.dropna(subset=["title"])

titles = dataset["title"].tolist()
y = dataset["label"].values.astype(float)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(titles).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

