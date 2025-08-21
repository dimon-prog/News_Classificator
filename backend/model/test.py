import joblib

vectorizer = joblib.load("vectorizer.pkl")
ensemble = joblib.load("model.pkl")
test = ["The iPhone 17 is expected to be released in September 2025."]
X_test_2 = vectorizer.transform(test)
print(ensemble.predict_proba(X_test_2))
print("True" if ensemble.predict_proba(X_test_2)[0][0] < 0.5 else "False")
