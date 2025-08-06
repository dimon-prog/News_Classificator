import pandas as pd
import os
#print(os.getcwd())
dataset = pd.read_csv("data/WELFake_Dataset.csv")
#print(dataset["title"])
print(dataset.columns)
print(dataset["title"][0])
print(dataset["label"][0])
#X = dataset["title"].tolist()
#labels = dataset["label"].values.astype(float)
#labels = labels.astype(float)
#print(labels)
