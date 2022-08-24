import pandas as pd

SPLITS = {"train", "valid", "test"}

for split in SPLITS:
    df = pd.read_csv(f"https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/{split}_preprocess.tsv",
                     sep="\t", names=["text", "label"])
    df["source"] = "indonlue/smsa"
    df.to_csv(f"../data/indonesian/{split}.csv", index=False)
