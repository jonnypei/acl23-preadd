import argparse
from utils.utils import *
from utils.constants import *
from utils.engine_util import *

"""
Generation with No Control
"""

def generate_standard(prompt,
                      max_tokens=32,
                      temperature=1,
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

    prompt = tokenizer.encode(prompt)
    prompt_cache_id = None

    for i in range(max_tokens):
        prompt_output = get_next_logprobs(
            prompt, model_string, cache_id=prompt_cache_id)
        prompt_cache_id = prompt_output['cache_id']

        final_logprobs = torch.Tensor(prompt_output['logits']) / temperature

        # Decoding Options:
        if task == "bias":  # enforced pronoun decoding
            next_token, female_ratio, male_ratio = get_pronoun_token(
                final_logprobs, pronoun_tokens)
        else:  # standard randomized decoding
            # next_token = torch.argmax(final_logprobs).item()
            next_token = torch.multinomial(
                torch.softmax(final_logprobs, dim=0), 1).item()

        prompt.append(next_token)

    if task == "bias":
        return tokenizer.decode(prompt), female_ratio, male_ratio
    return tokenizer.decode(prompt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_prompt', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--model_string', type=str,
                        default='facebook/opt-125m', choices=models)
    args = parser.parse_args()
    print(generate_standard(args.control_prompt, max_tokens=args.max_tokens,
          temperature=args.temperature, model_string=args.model_string))
