import json
import pandas as pd
import jsonlines

def generate_twitter_sa_dataset(input_file, output_file):
    df = pd.read_csv(input_file, names=['id', 'entity', 'sentiment', 'text'])
    samples = []
    # data = []
    # labels = []
    for idx, row in df.iterrows():
        # # data.append(row['text'])
        # label = None
        # if row['sentiment'] == 'Positive':
        #     # labels.append(0)
        #     label = 0
        # elif row['sentiment'] == 'Negative':
        #     # labels.append(2)
        #     label = 2
        # else:
        #     # Neutral
        #     # labels.append(1)
        #     label = 1
        if not isinstance(row['sentiment'], str) or not isinstance(row['text'], str) or len(row['text']) <= 0 or len(row['sentiment']) <= 0:
            continue
        samples.append({"input": row['text'], "label": row['sentiment']})

    # with open(output_file, "w") as f:
    #     # json.dump({"data": data, "labels": labels}, f)
    #     json.dump(samples, f)
    with jsonlines.open(output_file, "w") as writer:
        for sample in samples:
            writer.write(sample)

if __name__ == "__main__":
    generate_twitter_sa_dataset("./twitter_sa/twitter_training.csv",
                                "../data/twitter_training.json")
    generate_twitter_sa_dataset("./twitter_sa/twitter_validation.csv",
                                "../data/twitter_validation.json")