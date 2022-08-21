import requests

LANGUAGES = {"english": "en"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}
ID2LABEL = {"0": "negative", "1": "neutral", "2": "positive"}

for language in LANGUAGES:
    for split in SPLITS:
        text_url = f"https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/{language}/{split}_text.txt"
        label_url = f"https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/{language}/{split}_labels.txt"

        text_response = requests.get(text_url)
        texts = text_response.content.decode('unicode-escape').split("\n")

        label_response = requests.get(label_url)
        labels = label_response.content.decode('unicode-escape').split("\n")

        assert len(texts) == len(
            labels), f"Number of texts and labels not equal for {language}/{split}_text.txt and {language}/{split}_labels.txt"

        with open(f"../{LANGUAGES[language]}/{SPLITS[split]}.tsv", "w") as f:
            for text, label in zip(texts, labels):
                f.write(text + "\t" + ID2LABEL[label] + "\n")
