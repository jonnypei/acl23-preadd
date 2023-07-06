from utils.constants import wino_datasets

bias_sentences = {}

for k, dataset in wino_datasets.items():
    curr_sentences = []
    for data in dataset:
        if 'test' in data:
            continue
        with open(f"./data/wino/processed_data/{data}", "r") as f:
            curr_sentences.extend(f.readlines())

    with open(f"./data/{k}_stereotyped_sentences_devonly.txt", "w") as f:
        for sentence in curr_sentences:
            f.write(sentence)
