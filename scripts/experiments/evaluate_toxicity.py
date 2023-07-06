import os
import json
import torch
import argparse
import openai
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

from methods.preadd_prefixes import *
from methods.method_preadd import generate_preadd
from methods.method_standard import generate_standard
from methods.method_fudge import generate_fudge

from utils.constants import *
from utils.metrics import *
from utils.utils import *

from methods.fudge.constants import fudge_models

from generate_control_text import generate_control_text


@torch.no_grad()
def generate_toxicity_evals(save_dir,
                            prompts_setting,
                            method,
                            prefix_setting,
                            num_sentences,
                            strength,
                            max_tokens,
                            top_k,
                            top_p,
                            thresh,
                            model_string,
                            fudge_model_string,
                            task,
                            display):

    prompts = prompt_datasets[prompts_setting]

    # Sentence Selection Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2").to(device)

    # Prefix Sentences
    with open("./data/toxicity_prompts/toxic_dss_prefixes.jsonl", "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        sentences = [d['prompt']['text'] + d['continuation']['text']
                     for d in data]
    embeddings = model.encode(sentences)

    generations = []
    outputs = []

    assert prompts.endswith('.jsonl')
    with open(prompts, "r") as f:
        for prompt in tqdm(f.readlines()):
            prompt = json.loads(prompt)['prompt']["text"]
            # Control-Generated Text
            if len(prompt) < 3:
                continue

            # Generate output
            generated_text = generate_control_text(task=task,
                                                    method=method,
                                                    prompt=prompt,
                                                    prefix_setting=prefix_setting,
                                                    num_sentences=num_sentences,
                                                    sentences=sentences,
                                                    sentence_embeds=embeddings,
                                                    sentence_model=model,
                                                    strength=strength,
                                                    max_tokens=max_tokens,
                                                    top_k=top_k,
                                                    top_p=top_p,
                                                    thresh=thresh,
                                                    model_string=model_string,
                                                    fudge_model_string=fudge_model_string) 
            
            if generated_text.startswith('</s>'):
                generated_text = generated_text[4:].strip()
            if method == "neg_prompting":
                generated_text = generated_text[len(all_prefixes[task]["neg"]):].strip()

            # Perspective API Scores
            summary_scores, _ = get_perspective_api_scores(
                generated_text, display)
            continuation_scores, _ = get_perspective_api_scores(
                generated_text[len(prompt):].strip(), display)

            # Fluency
            davinci_continuation_perplexity = fluency(prompt, generated_text)

            if display:
                print("Prompt:", prompt, "\n")
                if method == "preadd":
                    print("Prefix:", all_prefixes[task][prefix_setting], "\n")
                    print("Strength:", strength, "\n")
                print(f"Generated Text by {method}:", generated_text, "\n")
                print(f"Summary Scores: {summary_scores}", "\n\n=====\n")

            generations.append(generated_text)
            outputs.append({"content": generated_text,
                            "api_scores": dict(sorted(summary_scores.items())),
                            "continuation_api_scores": dict(sorted(continuation_scores.items())),
                            "davinci_continuation_perplexity": davinci_continuation_perplexity
                            })

    # Perplexity & Grammaticality
    ppl, _, _ = perplexity(generations)
    gram, _, _ = grammaticality(generations)
    for i in range(len(outputs)):
        outputs[i]["perplexity"] = ppl[i]
        outputs[i]["grammaticality"] = gram[i]

    write_eval_output_file(outputs, save_dir, prompts_setting,
                           method, prefix_setting, strength, top_k, top_p, thresh)

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--prompts_setting', required=True,
                        type=str, choices=toxicity_prompts)
    parser.add_argument('--method', required=True, type=str, choices=methods)
    parser.add_argument('--prefix_setting', type=str,
                        default="pos", choices=list(toxic_prefixes.keys()))
    parser.add_argument('--num_sentences', type=str, default=10)
    parser.add_argument('--strength', type=float, default=-1.0)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=0) # default is no top_k
    parser.add_argument('--top_p', type=float, default=1) # default is no top_p
    parser.add_argument('--thresh', type=float, default=0)
    parser.add_argument('--model_string', type=str,
                        default='facebook/opt-125m', choices=models)
    parser.add_argument('--fudge_model_string', type=str, 
                        default='facebook/opt-125m', choices=fudge_models)
    parser.add_argument('--display', action='store_true', default=False)
    args = parser.parse_args()

    # Evaluate model text generations
    generate_toxicity_evals(save_dir=args.save_dir,
                            prompts_setting=args.prompts_setting,
                            method=args.method,
                            prefix_setting=args.prefix_setting,
                            num_sentences=args.num_sentences,
                            strength=args.strength,
                            max_tokens=args.max_tokens,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            thresh=args.thresh,
                            model_string=args.model_string,
                            fudge_model_string=args.fudge_model_string,
                            task="toxicity",
                            display=args.display)
