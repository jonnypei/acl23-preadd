CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength 0 --top_p=0.9 --model_string facebook/opt-125m
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength -1.0 --top_p=0.9 --model_string facebook/opt-125m
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength -2.0 --top_p=0.9 --model_string facebook/opt-125m
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength -3.0 --top_p=0.9 --model_string facebook/opt-125m
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength -4.0 --top_p=0.9 --model_string facebook/opt-125m
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength -5.0 --top_p=0.9 --model_string facebook/opt-125m
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength -10 --top_p=0.9 --model_string facebook/opt-125m
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength -20 --top_p=0.9 --model_string facebook/opt-125m
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_toxic_small --method preadd --prefix_setting pos --strength -30 --top_p=0.9 --model_string facebook/opt-125m