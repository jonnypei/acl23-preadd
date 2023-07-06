from transformers import AutoTokenizer
from utils.utils import *

from methods.preadd_prefixes import *
from methods.method_preadd import generate_preadd
from methods.method_standard import generate_standard
from methods.method_fudge import generate_fudge

def generate_control_text(task, 
                          method,
                          prompt, 
                          prefix_setting, 
                          num_sentences, 
                          sentences, 
                          sentence_embeds, 
                          sentence_model, 
                          strength, 
                          max_tokens, 
                          top_k,
                          top_p,
                          thresh,
                          model_string,
                          fudge_model_string):
    # generate output
    if method == "preadd":
        if prefix_setting == "dynamic":
            prefix = construct_prefix(source=prompt,
                                        sentences=sentences,
                                        sentence_embeds=sentence_embeds,
                                        model=sentence_model,
                                        task=task,
                                        mode=prefix_setting,
                                        num_sentences=num_sentences)
        else:
            prefix = all_prefixes[task][prefix_setting]

        # output = generate_preadd(prompt=prompt,
        #                             prefix=prefix,
        #                             strength=strength,
        #                             max_tokens=max_tokens,
        #                             top_k=top_k,
        #                             top_p=top_p,
        #                             thresh=thresh,
        #                             model_string=model_string,
        #                             task=task)
        while True:
            try:
                output = generate_preadd(prompt=prompt,
                                    prefix=prefix,
                                    strength=strength,
                                    max_tokens=max_tokens,
                                    top_k=top_k,
                                    top_p=top_p,
                                    thresh=thresh,
                                    model_string=model_string,
                                    task=task)
                break
            except:
                print(prompt)
                print(
                    f"Prefix length of {len(prefix)} is too long. Try truncating.")
                tokenizer = AutoTokenizer.from_pretrained(model_string)
                prefix_tokens = tokenizer.encode(prefix)
                prefix_tokens = prefix_tokens[:len(prefix_tokens)//2]
                prefix = tokenizer.decode(prefix_tokens)
    elif method == "raw_prompting":
        output = generate_standard(prompt=prompt,
                                    max_tokens=max_tokens,
                                    model_string=model_string,
                                    task=task)
    elif method == "neg_prompting":
        prefix = all_prefixes[task]["neg"]
        output = generate_standard(prompt=prefix + " " + prompt,
                                    max_tokens=max_tokens,
                                    model_string=model_string,
                                    task=task)
    elif method == "fudge":
        output = generate_fudge(prompt=prompt,
                                strength=1,
                                max_tokens=max_tokens,
                                model_string=model_string,
                                control_model_string=fudge_model_string,
                                task=task)
    else:
        raise NotImplementedError
    
    return output