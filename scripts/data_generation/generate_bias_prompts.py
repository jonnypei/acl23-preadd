import re
from utils.constants import pronouns, male_pronouns, female_pronouns, wino_datasets, bias_prompts_dir

occupation_bias_prompts = []
female_prompts = []
male_prompts = []

for k, datasets in wino_datasets.items():
    for dataset in datasets:
        if 'dev' in dataset:
            continue
        with open(f"./data/wino/raw_data/{dataset}", "r") as f:
            for line in f.readlines():
                if "type2" in dataset:
                    occupation_bias_prompt = ("".join([re.sub("\]", "", part) for part in line.split("[")
                                                       if not any(pn in re.sub("\]", "", part).strip().split(" ") for pn in pronouns)]))
                    occupation_bias_prompt = " ".join(
                        occupation_bias_prompt.split(" ")[1:])
                    if occupation_bias_prompt not in occupation_bias_prompts:
                        occupation_bias_prompts.append(occupation_bias_prompt)
                else:  # type1
                    gender_prompt = (
                        "".join([re.sub("\[", "", part) for part in line.split("]")][:2]))
                    gender_prompt = " ".join(gender_prompt.split(" ")[1:])
                    if any(pn in gender_prompt.split(" ") for pn in male_pronouns) and gender_prompt not in male_prompts:
                        male_prompts.append(gender_prompt)
                    if any(pn in gender_prompt.split(" ") for pn in female_pronouns) and gender_prompt not in female_prompts:
                        female_prompts.append(gender_prompt)

with open(f"{bias_prompts_dir}/basic_prompts_testonly.txt", "w") as f:
    for prompt in occupation_bias_prompts:
        f.write(prompt + "\n")

with open(f"{bias_prompts_dir}/female_prompts_testonly.txt", "w") as f:
    for prompt in female_prompts:
        f.write(prompt + "\n")

with open(f"{bias_prompts_dir}/male_prompts_testonly.txt", "w") as f:
    for prompt in male_prompts:
        f.write(prompt + "\n")
