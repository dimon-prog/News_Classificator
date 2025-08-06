import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

data = pd.read_csv("data/WELFake_Dataset.csv")

clf = LogisticRegression()

