import json
import numpy as np
import argparse
import random

from tqdm import tqdm

from utils.constants import *
from utils.metrics import *

np.random.seed(0)

def sort_lines_by_prompt_toxicity(lines):
    scores = []
    for line in tqdm(lines):
        line = json.loads(line)
        if line['prompt']['toxicity'] is None:
            scores.append(0.0001)
        else:
            scores.append(line['prompt']['toxicity'])
    scores = np.array(scores)
    indices = np.argsort(scores)
    return list(reversed([lines[i] for i in indices]))  # descending order


def sort_lines_by_overall_toxicity(lines):
    scores = []
    for line in tqdm(lines):
        line = json.loads(line)
        prompt_tox = 0.0001 if line['prompt']['toxicity'] is None else line['prompt']['toxicity']
        cont_tox = 0.0001 if line['continuation']['toxicity'] is None else line['continuation']['toxicity']
        scores.append(max(prompt_tox, cont_tox))
    scores = np.array(scores)
    indices = np.argsort(scores)
    return list(reversed([lines[i] for i in indices]))  # descending order


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_prompts', type=int, default=1000)
    args = parser.parse_args()

    random.seed(0)

    with open("./data/realtoxicity/realtoxicityprompts.jsonl", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    random.shuffle(lines)

    with open(prompt_datasets["toxicity_basic"], "w") as f:
        for line in lines[:8]:
            f.write(line + "\n")

    with open(prompt_datasets["toxicity_random_small"], "w") as f:
        for line in lines[:100]:
            f.write(line + "\n")

    with open(prompt_datasets["toxicity_random"], "w") as f:
        for line in lines[:1000]:
            f.write(line + "\n")

    with open(f"{toxicity_prompts_dir}/training_prompts.jsonl", "w") as f:
        for line in lines[1000:]:
            f.write(line + "\n")

    with open(f"{toxicity_prompts_dir}/dss_prefixes.jsonl", "w") as f:
        for line in sort_lines_by_overall_toxicity(lines[1000:])[:1000]:
            f.write(line + "\n")

    lines = sort_lines_by_prompt_toxicity(lines)

    with open(f"{toxicity_prompts_dir}/toxic_basic_prompts.jsonl", "w") as f:
        for line in lines[:8]:
            f.write(line + "\n")

    with open(prompt_datasets["toxicity_toxic_small"], "w") as f:
        for line in lines[:100]:
            f.write(line + "\n")

    with open(prompt_datasets["toxicity_toxic"], "w") as f:
        for line in lines[:1000]:
            f.write(line + "\n")

    with open(f"{toxicity_prompts_dir}/toxic_training_prompts.jsonl", "w") as f:
        for line in lines[1000:]:
            f.write(line + "\n")

    with open(f"{toxicity_prompts_dir}/toxic_dss_prefixes.jsonl", "w") as f:
        for line in sort_lines_by_overall_toxicity(lines[1000:])[:1000]:
            f.write(line + "\n")
