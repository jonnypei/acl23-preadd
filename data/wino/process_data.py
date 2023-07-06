import re

# All standard format datasets in raw_data dir
wino_datasets = ["anti_stereotyped_type1.txt.dev",
                 "anti_stereotyped_type1.txt.test",
                 "anti_stereotyped_type2.txt.dev",
                 "anti_stereotyped_type2.txt.test",
                 "pro_stereotyped_type1.txt.dev",
                 "pro_stereotyped_type1.txt.test",
                 "pro_stereotyped_type2.txt.dev",
                 "pro_stereotyped_type2.txt.test"]
                 
# Cleans prompt data
for dataset in wino_datasets:
    with open(f"./raw_data/{dataset}", "r") as f:
        prompts = [" ".join(re.sub("[\[\]]", "", line).strip().split()[1:])
                   for line in f.readlines()]

    with open(f"./processed_data/{dataset}", "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")
