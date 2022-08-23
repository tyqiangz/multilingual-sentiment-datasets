import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

LINKS = [
    "https://raw.githubusercontent.com/huseinzol05/malay-dataset/master/sentiment/news-sentiment/sentiment-data-v2.csv",
    "https://raw.githubusercontent.com/huseinzol05/malay-dataset/master/sentiment/supervised-twitter/data.csv",
    "https://raw.githubusercontent.com/huseinzol05/malay-dataset/master/sentiment/supervised-twitter-politics/data.csv"
]

dfs = [
    pd.read_csv(LINKS[0]),
    pd.read_csv(LINKS[1], sep="\t"),
    pd.read_csv(LINKS[2], sep="\t")
]

texts = []
labels= []

for i in range(len(dfs[0])):
    if not pd.isna(dfs[0].loc[i, "label"]):
        texts.append(dfs[0].loc[i, "text"])
        labels.append(dfs[0].loc[i, "label"])
    
for i in range(len(dfs[1])):
    if not pd.isna(dfs[1].loc[i, "sentiment"]):
        texts.append(dfs[1].loc[i, "text"])
        labels.append(dfs[1].loc[i, "sentiment"])
    
for i in range(len(dfs[2])):
    if not pd.isna(dfs[2].loc[i, "sentiment"]):
        texts.append(dfs[2].loc[i, "text"])
        labels.append(dfs[2].loc[i, "sentiment"])

df = pd.DataFrame({"text": texts, "label": labels})
df["label"] = df["label"].str.lower()

X = df["text"]
y = df["label"]

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)

for train_index, dev_index in sss1.split(X, y):
    X_train, X_dev = X[train_index].reset_index(drop=True), X[dev_index].reset_index(drop=True)
    y_train, y_dev = y[train_index].reset_index(drop=True), y[dev_index].reset_index(drop=True)

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

for val_index, test_index in sss2.split(X_dev, y_dev):
    X_val, X_test = X_dev[val_index].reset_index(drop=True), X_dev[test_index].reset_index(drop=True)
    y_val, y_test = y_dev[val_index].reset_index(drop=True), y_dev[test_index].reset_index(drop=True)
    
train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
valid_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

train_df.to_csv("../data/ms/train.tsv", sep="\t", index=False)
valid_df.to_csv("../data/ms/valid.tsv", sep="\t", index=False)
valid_df.to_csv("../data/ms/test.tsv", sep="\t", index=False)