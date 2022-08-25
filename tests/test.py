""" test tthat all the data files conform to some common format """
import pandas as pd
import os

LANGUAGES = os.listdir("../data")

assert "all" in LANGUAGES

for lang in LANGUAGES:
    filenames = os.listdir(f'../data/{lang}')

    assert 'train.csv' in filenames, f"./{lang} doesn't have 'train.csv'"
    assert 'valid.csv' in filenames, f"./{lang} doesn't have 'valid.csv'"
    assert 'test.csv' in filenames, f"./{lang} doesn't have 'test.csv'"

    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(f'../data/{lang}/{split}.csv')
        # df = pd.read_csv(f"https://raw.githubusercontent.com/tyqiangz/multilingual-sentiment-datasets/main/data/{lang}/{split}.csv")

        assert 'text' in df.columns, f"./{lang}/{split}.csv doesn't have 'text' column"
        assert 'label' in df.columns, f"./{lang}/{split}.csv doesn't have 'label' column"
        assert 'source' in df.columns, f"./{lang}/{split}.csv doesn't have 'source' column"
        
        label_classes = df["label"].unique()

        assert len(label_classes) == 3
        assert "positive" in label_classes
        assert "neutral" in label_classes
        assert "negative" in label_classes

        if lang == 'all':
            assert 'language' in df.columns, f"./{lang}/{split}.csv doesn't have 'language' column"