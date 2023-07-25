# PreAdD: Prefix-Adaptive Decoding for Controlled Text Generation

This repository contains a code implementation of the paper ["PreAdD: Prefix-Adaptive Decoding for Controlled Text Generation"](https://arxiv.org/abs/2307.03214) by Jonathan Pei, Kevin Yang, and Dan Klein. 

## Setup/Installation

(1) Install Python 3.7.3, but slightly older/new versions should work just fine.

(2) Clone this repo and move into it by running
```
git clone https://github.com/jonnypei/acl23-preadd
cd acl23-preadd
```

(3) Install the required packages by running

```
pip install -r requirements.txt
```

(4) Install the repo by running

```
pip install -e .
```

(5) Follow the instructions at https://developers.perspectiveapi.com/s/docs-get-started?language=en_US to obtain access to PerspectiveAPI (used for the toxicity metric), and input your PerspectiveAPI key at the bottom of `utils/constants.py`.

(6) Sign-up for [OpenAI API](https://openai.com/product) and generate a new API key (if you don't already have one). Then, input your OpenAI API key at the bottom of `utils/constants.py`.

(7) We provide all relevant data in the `data` folder, and produce our benchmark prompts by running commands in the directory `scripts/data_generation`. Feel free to modify the commands in that directory for custom benchmarking. 

## Prompting Model Server

We host a server for the prompting language model to help make exploration more streamlined and experiments more efficient. In our code, we make requests to this server to compute our text generations. Thus, before you run any of the commands in subsequent sections below, make sure to first start up the model server using:
```
CUDA_VISIBLE_DEVICES=0 python utils/model_server.py --model_string=facebook/opt-6.7b
```
Feel free to change the model to a larger one (e.g. `facebook/opt-30b`) when using our method as a baseline.

## Evaluation

### Toxicity Mitigation

To generate outputs and evaluate metrics, run:
```
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_toxicity.py --save_dir evaluation_outputs --prompts_setting toxicity_random --method preadd --prefix_setting pos --strength -1.0 --model_string facebook/opt-6.7b
```
**Note:** make sure that the `model_string` parameter here is the same as the model on the server (or at least has the same tokenizer).

### Bias Reduction

To generate outputs and evaluate metrics, run:
```
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_bias.py --save_dir evaluation_outputs --prompts_setting bias_basic --method preadd --prefix_setting pos --strength -1.0 --model_string facebook/opt-6.7b
```
**Note:** make sure that the `model_string` parameter here is the same as the model on the server (or at least has the same tokenizer).

### Sentiment Control

To generate outputs and evaluate metrics, run:
```
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/evaluate_sentiment.py --save_dir evaluation_outputs --prompts_setting sentiment --method preadd --prefix_setting pos --strength 2.0 --model_string facebook/opt-6.7b
```
**Note:** make sure that the `model_string` parameter here is the same as the model on the server (or at least has the same tokenizer).

## Analysis

### Toxicity Mitigation

To obtain summary results of a single set of outputs, run e.g.:
```
python scripts/analysis/analyze_toxicity_results.py --outputs evaluation_outputs/toxicity_random/preadd_pos_-1.0__outputs.jsonl
```

Then, to perform a paired t-test on two sets of outputs, run e.g.:
```
python scripts/analysis/analyze_toxicity_results.py --outputs evaluation_outputs/toxicity_random/preadd_pos_-1.0__outputs.jsonl evaluation_outputs/toxicity_random/raw_prompting_outputs.jsonl
```

### Bias Reduction

To obtain summary results of a single set of outputs, run e.g.:
```
python scripts/analysis/analyze_bias_results.py --outputs evaluation_outputs/bias_basic/preadd_pos_-1.0__outputs.jsonl
```

Then, to perform a paired t-test on two sets of outputs, run e.g.:
```
python scripts/analysis/analyze_bias_results.py --outputs evaluation_outputs/bias_basic/preadd_pos_-1.0__outputs.jsonl evaluation_outputs/bias_basic/raw_prompting_outputs.jsonl
```

### Sentiment Control

Run similar commands as previous two tasks.

## Adapting your own prefix

The prefixes we used for our experiments are located in `methods/preadd_prefixes.py`, sectioned by task. Feel free to play with your own prefixes by running e.g.
```
CUDA_VISIBLE_DEVICES=0 python methods/method_preadd.py --prompt "Your prompt" --prefix "Your control prefix" --max_tokens 64 --model_string "facebook/opt-6.7b"
```
**Note:** make sure that the `model_string` parameter here is the same as the model on the server (or at least has the same tokenizer).

## FUDGE Baseline

We provide our tuned parameters for the FUDGE control model in `methods/fudge/ckpt` (for `facebook/opt-125m`) and `methods/fudge/ckpt_neo` (for `EleutherAI/gpt-neo-125m`). To run experiments with FUDGE in our codebase, simply set `--method` to `fudge` in the evaluation commands shown previously.

To train your own FUDGE model, run e.g.
```
CUDA_VISIBLE_DEVICES=0 python methods/fudge/main.py --task toxicity --save_dir methods/fudge/your_ckpt --lr 1e-3
``` 
