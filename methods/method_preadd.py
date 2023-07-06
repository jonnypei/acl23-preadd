import argparse
from utils.utils import *
from utils.constants import *
from utils.engine_util import *

"""
Controlled Generation with Prefix-Adapted Decoding
"""

def generate_preadd(prompt,
                    prefix,
                    strength=0,
                    max_tokens=32,
                    temperature=1,
                    top_k=0,
                    top_p=1,
                    thresh=0,
                    model_string='facebook/opt-125m',
                    task=None):
    # prompt should be a string
    tokenizer = get_tokenizer(model_string)

    if model_string == "facebook/opt-6.7b":
        pronoun_tokens = dict([(pn, tokenizer.encode(pn)[1])
                          for pn in pronouns[:6]])
    elif model_string == "EleutherAI/gpt-j-6B":
        pronoun_tokens = dict([(pn, tokenizer.encode(pn)[0])
                          for pn in pronouns[:6]])

    prefix = tokenizer.encode(prefix + " " + prompt)
    prefix_cache_id = None

    prompt = tokenizer.encode(prompt)
    prompt_cache_id = None

    for i in range(max_tokens):
        prompt_output = get_next_logprobs(
            prompt, model_string, cache_id=prompt_cache_id)
        prompt_cache_id = prompt_output['cache_id']

        prefix_output = get_next_logprobs(
            prefix, model_string, cache_id=prefix_cache_id)
        prefix_cache_id = prefix_output['cache_id']

        # base_logprobs = torch.Tensor(prompt_output['logits']) # no filtering
        base_logprobs = top_k_top_p_filtering(torch.Tensor(prompt_output['logits'])[None, :], top_k=top_k, top_p=top_p)[0] # top-k/top-p filtering
        # base_logprobs = thresh_filtering(torch.Tensor(prompt_output['logits'], thresh=TODO)) # absolute threshold filtering

        diff = torch.Tensor(prefix_output['logits']) - base_logprobs
        final_logprobs = (base_logprobs + diff * strength) / temperature

        # Decoding Options:
        if task == "bias":  # enforced pronoun decoding
            next_token, female_ratio, male_ratio = get_pronoun_token(
                final_logprobs, pronoun_tokens)
        else:  # standard randomized decoding
            next_token = torch.multinomial(
                torch.softmax(final_logprobs, dim=0), 1).item()

        prompt.append(next_token)
        prefix.append(next_token)

    if task == "bias":
        return tokenizer.decode(prompt), female_ratio, male_ratio
    return tokenizer.decode(prompt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--strength', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--thresh', type=float, default=0)
    parser.add_argument('--model_string', type=str,
                        default='facebook/opt-125m', choices=models)
    args = parser.parse_args()
    print(generate_preadd(args.prompt, args.prefix, strength=args.strength,
          max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, thresh=args.thresh, model_string=args.model_string))
