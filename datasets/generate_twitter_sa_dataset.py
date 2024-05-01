import json
import pandas as pd

def generate_twitter_sa_dataset(input_file, output_file):
    df = pd.read_csv(input_file, names=['id', 'entity', 'sentiment', 'text'])
    data = []
    labels = []
    for idx, row in df.iterrows():
        data.append(row['text'])
        if row['sentiment'] == 'Positive':
            labels.append(0)
        elif row['sentiment'] == 'Negative':
            labels.append(2)
        else:
            # Neutral
            labels.append(1)

    with open(output_file, "w") as f:
        json.dump({"data": data, "labels": labels}, f)

if __name__ == "__main__":
    generate_twitter_sa_dataset("datasets/twitter_sa/twitter_training.csv", 
                                "datasets/twitter_sa/twitter_training.json")
    generate_twitter_sa_dataset("datasets/twitter_sa/twitter_validation.csv", 
                                "datasets/twitter_sa/twitter_validation.json")