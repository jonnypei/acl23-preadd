import sys

PAD_TOKEN = '[PAD]'
EOT_TOKEN = '<|endoftext|>'
SEP = 50256  # just use the weird eot token

fudge_models = ['facebook/opt-125m', 'EleutherAI/gpt-neo-125M']
# HIDDEN_DIM = 1024

SAVE_PATH = "./methods/fudge/ckpt"
TOXICITY_DATA_PATH = "./data/toxicity_prompts/toxic_training_prompts.jsonl"
BIAS_DATA_PATH = "./data/wino/processed_data"
