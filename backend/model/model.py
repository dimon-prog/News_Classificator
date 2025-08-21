from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
import csv
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

csv.field_size_limit(10**7)
data1 = pd.read_csv("data/fake_or_real_news.csv", engine="python")
data2 = pd.read_csv("data/FakeNewsNet.csv", engine="python")
data3 = pd.read_csv("data/WELFake_Dataset.csv", engine="python")

data2 = data2.rename(columns={"real": "label"})
data1["label"] = data1["label"].replace({"REAL": 1, "FAKE": 0})
data3["label"] = data3["label"].replace({1: 0, 0: 1})
data1 = data1[["title", "label"]]
data2 = data2[["title", "label"]]
data3 = data3[["title", "label"]]

dataset = pd.concat([data1, data2, data3], ignore_index=True)

dataset = dataset.dropna(subset=["title"])

titles = dataset["title"].tolist()
y = dataset["label"].values.astype(float)
vectorizer = TfidfVectorizer(max_features=50000, stop_words='english', min_df=5, max_df=0.9 )
X = vectorizer.fit_transform(titles)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
joblib.dump(vectorizer, "vectorizer.pkl")
lr = LogisticRegression(max_iter=1000, penalty="l2", C=10, solver="lbfgs")
lr.fit(X_train, y_train)
nb = MultinomialNB(alpha=0.01)
svc = LinearSVC(C=1, loss="squared_hinge")
calibrated_svc = CalibratedClassifierCV(svc, method="sigmoid", cv=2)
ensemble = VotingClassifier(estimators=[("lr", lr), ("nb", nb), ("svc", calibrated_svc)], voting="soft")
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict_proba(X_test)
accuracy = ensemble.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
loss = log_loss(y_test, y_pred)
print(f"loss= {loss}")
#joblib.dump(ensemble, "model.pkl")

