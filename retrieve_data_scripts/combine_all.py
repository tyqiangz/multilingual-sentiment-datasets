import pandas as pd

LANGUAGES = [
    'malay', 'hindi', 'japanese', 'german',
    'italian', 'english', 'portuguese', 'french',
    'spanish', 'chinese', 'indonesian', 'arabic'
]

for split in ["train", "valid", "test"]:
    df_list = []
    
    for lang in LANGUAGES:
        print(lang, split)
        filepath = f"../data/{lang}/{split}.csv"
        df = pd.read_csv(filepath)
        df["language"] = lang
        df_list.append(df)

        all_df = pd.concat(df_list)

        all_df.to_csv(f"../data/all/{split}.csv", index=False)