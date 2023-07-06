import argparse
import torch

from utils.utils import *
from utils.constants import *
from utils.engine_util import *
from methods.fudge.util import load_fudge_model

"""
Controlled Generation with FUDGE

https://arxiv.org/abs/2104.05218
"""

@torch.no_grad()
def generate_fudge(prompt,
                   strength=0,
                   max_tokens=32,
                   temperature=1,
                   model_string='facebook/opt-125m',
                   control_model_string='facebook/opt-125m',
                   task=None,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    # prompt should be a string
    tokenizer = get_tokenizer(model_string)

    if model_string == "facebook/opt-6.7b":
        pronoun_tokens = dict([(pn, tokenizer.encode(pn)[1])
                          for pn in pronouns[:6]])
    elif model_string == "EleutherAI/gpt-j-6B":
        pronoun_tokens = dict([(pn, tokenizer.encode(pn)[0])
                          for pn in pronouns[:6]])

    prompt = tokenizer.encode(prompt)
    prompt_cache_id = None

    # load in FUDGE control model
    if control_model_string == 'facebook/opt-125m':
        ckpt_folder = 'ckpt'
    elif control_model_string == 'EleutherAI/gpt-neo-125M':
        ckpt_folder = 'ckpt_neo'

    if task == 'toxicity':
        control_model = load_fudge_model(
            f'./methods/fudge/{ckpt_folder}/toxicity_model_best.pth', task, control_model_string)
    elif task == 'bias':
        control_model = load_fudge_model(
            f'./methods/fudge/{ckpt_folder}/bias_model_best.pth', task, control_model_string)
    else:
        raise NotImplementedError

    for i in range(max_tokens):
        prompt_output = get_next_logprobs(
            prompt, model_string, cache_id=prompt_cache_id)
        prompt_cache_id = prompt_output['cache_id']

        base_logits = (torch.Tensor(
            prompt_output['logits']) / temperature).to(device)
        if task == 'toxicity':
            topk = 100
        else:  # bias
            topk = 5
            pronoun_mask = torch.zeros_like(base_logits)
            for pn in pronouns[:6]:
                pronoun_mask[pronoun_tokens[pn]] = 1
            base_logits = base_logits * \
                pronoun_mask + (1 - pronoun_mask) * -1e8

        # get logprobs and indices corresponding to the topk tokens in prompt_output
        top_logits, top_indices = torch.topk(base_logits, topk)  # dim = topk

        # form input to control model by trying to append each topk candidate to current sequence
        prompt_topk_candidates = torch.cat([torch.LongTensor(prompt).to(device).unsqueeze(
            0).expand(topk, -1), top_indices.unsqueeze(1)], dim=1)  # dim = topk x (seq+1)

        # plug into control model
        if strength == 0:
            control_offset = torch.zeros_like(top_logits).float()
        else:
            control_logits = control_model(prompt_topk_candidates)[:, -1, 0]
            if task == 'toxicity':  # fudge trained to predict toxicity
                negative_logprobs = torch.log(
                    1 - torch.sigmoid(control_logits))
                control_offset = negative_logprobs
            else:  # fudge trained to predict anti stereotype for bias
                positive_logprobs = torch.log(torch.sigmoid(control_logits))
                control_offset = positive_logprobs

        # combine logprobs to get final ones corresponding to top indices
        final_logits = top_logits + strength * control_offset

        # Decoding Options:
        if task == "bias":  # enforced pronoun decoding
            scattered_logits = torch.zeros_like(base_logits) - 1e8
            scattered_logits.scatter_(0, top_indices, final_logits)
            next_token, female_ratio, male_ratio = get_pronoun_token(
                scattered_logits, pronoun_tokens)
        else:  # standard randomized decoding
            next_token = top_indices[torch.multinomial(
                torch.softmax(final_logits, dim=0), 1)].item()

        prompt.append(next_token)

    if task == "bias":
        return tokenizer.decode(prompt), female_ratio, male_ratio
    return tokenizer.decode(prompt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_prompt', type=str, required=True)
    parser.add_argument('--strength', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--model_string', type=str, default='facebook/opt-125m',
                        choices=[model for model in models if "facebook/opt" in model])
    parser.add_argument('--task', type=str, default=None,
                        choices=["toxicity", "bias", None])
    args = parser.parse_args()
    print(generate_fudge(args.control_prompt, strength=args.strength, max_tokens=args.max_tokens,
          temperature=args.temperature, model_string=args.model_string, task=args.task))
