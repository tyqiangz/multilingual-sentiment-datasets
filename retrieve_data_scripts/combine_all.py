import pandas as pd

LANGUAGES = [
    'malay', 'hindi', 'japanese', 'german',
    'italian', 'english', 'portuguese', 'french',
    'spanish', 'chinese', 'indonesian', 'arabic'
]

df_list = {"train": [], "valid": [], "test": []}

for lang in LANGUAGES:
    for split in ["train", "valid", "test"]:
        print(lang, split)
        filepath = f"../data/{lang}/{split}.tsv"
        df = pd.read_csv(filepath, sep='\t')
        df["language"] = lang
        df_list[split].append(df)

train_df = pd.concat(df_list["train"])
valid_df = pd.concat(df_list["valid"])
test_df = pd.concat(df_list["test"])

train_df.to_csv("../data/all/train.csv", sep="\t", index=False)
valid_df.to_csv("../data/all/valid.csv", sep="\t", index=False)
test_df.to_csv("../data/all/test.csv", sep="\t", index=False)
