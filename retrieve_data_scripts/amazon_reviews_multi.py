import datasets
import pandas as pd

def aggregate(texts, stars, filename_to_save):
    STAR_SENTIMENT_MAP = {1: "negative", 3: "neutral", 5: "positive"}
    indices = [i for i in range(len(stars)) if stars[i] in (1, 3, 5)]
    
    subset_texts = [texts[i] for i in indices]
    subset_stars = [stars[i] for i in indices]
    labels = [STAR_SENTIMENT_MAP[star] for star in subset_stars]


    df = pd.DataFrame({"text": subset_texts, "label": labels})
    
    df.to_csv(filename_to_save, index=False)

    return df

LANGUAGES = {"zh": "chinese", "ja": "japanese"}

SPLITS = {"train": "train", "validation": "valid", "test": "test"}

for lang in LANGUAGES:
    dataset = datasets.load_dataset("amazon_reviews_multi", lang)

    for split in SPLITS:
        aggregate(texts=dataset[split]["review_body"], 
                stars=dataset[split]["stars"], 
                filename_to_save=f"../data/{LANGUAGES[lang]}/{SPLITS[split]}.csv")