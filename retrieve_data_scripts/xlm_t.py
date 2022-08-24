import requests
import pandas as pd

LANGUAGES = ['english', 'arabic', 'french', 'german', 
             'hindi', 'italian', 'portuguese', 'spanish']
SPLITS = {"train": "train", "val": "valid", "test": "test"}
ID2LABEL = {"0": "negative", "1": "neutral", "2": "positive"}

ENCODING = "utf-8"

for language in LANGUAGES:
    for split in SPLITS:
        text_url = f"https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/{language}/{split}_text.txt"
        label_url = f"https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/{language}/{split}_labels.txt"

        text_response = requests.get(text_url)
        texts = text_response.content.decode(ENCODING).split("\n")
            
        label_response = requests.get(label_url)
        labels = label_response.content.decode('ascii').split("\n")
        labels_in_text = [ID2LABEL[label] for label in labels]
        
        assert len(texts) == len(labels), f"Number of texts and labels not equal for {language}/{split}_text.txt and {language}/{split}_labels.txt"
        
        df = pd.DataFrame({"text": texts, "label": labels_in_text})
        df.to_csv(f"../data/{language}/{SPLITS[split]}.csv", index=False)
