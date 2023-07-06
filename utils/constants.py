########################################
# Gender Bias Generation Parameters
########################################

wino_datasets = {}
wino_datasets["anti"] = ["anti_stereotyped_type1.txt.dev",
                         "anti_stereotyped_type2.txt.dev",
                         "anti_stereotyped_type1.txt.test",
                         "anti_stereotyped_type2.txt.test"]
wino_datasets["pro"] = ["pro_stereotyped_type1.txt.dev",
                        "pro_stereotyped_type2.txt.dev",
                        "pro_stereotyped_type1.txt.test",
                        "pro_stereotyped_type2.txt.test"]

occupations = {"carpenter",
               "mechanic",
               "construction worker",
               "laborer",
               "driver",
               "sheriff",
               "mover",
               "developer",
               "farmer",
               "guard",
               "chief",
               "janitor",
               "lawyer",
               "cook",
               "physician",
               "CEO",
               "analyst",
               "manager",
               "supervisor",
               "salesperson",
               "editor",
               "designer",
               "accountant",
               "auditor",
               "writer",
               "baker",
               "clerk",
               "cashier",
               "counselor",
               "attendant",
               "teacher",
               "tailor",
               "librarian",
               "assistant",
               "cleaner",
               "housekeeper",
               "nurse",
               "receptionist",
               "hairdresser",
               "secretary"}
pronouns = ["he", "him", "his", "she", "her", "hers",
            "him.", "his.", "her."]
male_pronouns = ["he", "him", "his"]
female_pronouns = ["she", "her", "hers"]

########################################
# General Generation Parameters
########################################

methods = ["preadd",
           "raw_prompting",
           "neg_prompting",
           "pos_prompting",
           "fudge"]

tasks = ["toxicity", "bias"]

small_models = ["gpt2", "facebook/opt-125m"]
big_models = ["facebook/opt-6.7b", "EleutherAI/gpt-j-6B"]
models = small_models + big_models

toxicity_prompts_dir = "./data/toxicity_prompts"
bias_prompts_dir = "./data/bias_prompts"

prompt_datasets = {"toxicity_basic": f"{toxicity_prompts_dir}/basic_prompts.jsonl",
                   "toxicity_random": f"{toxicity_prompts_dir}/random_prompts.jsonl",
                   "toxicity_random_small": f"{toxicity_prompts_dir}/random_prompts_small.jsonl",
                   "toxicity_toxic": f"{toxicity_prompts_dir}/toxic_prompts.jsonl",
                   "toxicity_toxic_small": f"{toxicity_prompts_dir}/toxic_prompts_small.jsonl",
                   "bias_basic": f"{bias_prompts_dir}/basic_prompts_testonly.txt",
                   "bias_female": f"{bias_prompts_dir}/female_prompts_testonly.txt",
                   "bias_male": f"{bias_prompts_dir}/male_prompts_testonly.txt"}
toxicity_prompts = [p for p in prompt_datasets.keys() if "toxicity" in p]
bias_prompts = [p for p in prompt_datasets.keys() if "bias" in p]

all_prompts = toxicity_prompts + bias_prompts

########################################
# Metrics Parameters
########################################

PAD_TOKEN = '[PAD]'
EOT_TOKEN = '<|endoftext|>'

########################################
# Perspective API

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
########################################

# Input your PerspectiveAPI key here:
PERSPECTIVE_API_KEY = 

PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    # 'FLIRTATION'
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(
    a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)

########################################
# OpenAI
########################################

# Input your OpenAI api key here
OPENAI_API_KEY = 

