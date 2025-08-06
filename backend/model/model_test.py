from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys

csv.field_size_limit(sys.maxsize)
dataset = pd.read_csv("WELFake_Dataset.csv", engine="python")
dataset = dataset.dropna(subset=["title"])

test = ["BREAKING: PUTIN'S SUITCASE FOUND IN IT...!!!"]
titles = dataset["title"].tolist()
labels = dataset["label"].values.astype(float)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(titles).toarray()
X_test = vectorizer.transform(test).toarray()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Model:
    def __init__(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def training(self, X, labels):
        learning_rate = 1
        epochs = 2000
        loss_history = []
        n_samples = X.shape[0]
        for epoch in range(epochs):
            y_pred = sigmoid(np.dot(X, self.w) + self.b)
            error = y_pred - labels
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db
            loss = -np.mean(labels * np.log(y_pred) + (1 - labels) * np.log(1 - y_pred))
            loss_history.append(loss)
        plt.plot(loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


    def prediction(self, X):
        prediction = X.dot(self.w) + self.b
        return sigmoid(prediction)[0]
        #return "Fake" if sigmoid(prediction)[0] <= 0.5 else "True"


model_test = Model(X.shape[1])
model_test.training(X, labels)
print(model_test.prediction(X_test))
