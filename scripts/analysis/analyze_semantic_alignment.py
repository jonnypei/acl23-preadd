import argparse
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from sentence_transformers import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, required=True)
    parser.add_argument('--outputs', type=str, required=True)
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    prompts_f = args.prompts
    outputs_f = args.outputs
    
    with open(prompts_f, 'r') as f:
        prompts = []
        for line in f:
            prompts.append(json.loads(line)['prompt']['text'])
        num_prompts = len(prompts)

    with open(outputs_f, 'r') as f:
        outputs = [output['content'] for output in json.load(f)]
        # outputs = []
        # for line in f:
        #     outputs.append(json.loads(line)['content'])
        num_outputs = len(outputs)
    
    assert num_prompts == num_outputs

    continuations = []
    for i in range(num_prompts):
        continuations.append(outputs[i][len(prompts[i]):].strip())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2").to(device)

    semantic_alignments = []
    for i in tqdm(range(num_prompts)):
        prompt = prompts[i]
        continuation = outputs[i][len(prompts[i]):].strip()

        if args.debug and i % 100 == 0:
            print(outputs[i])
            print(prompt)
            print(continuation)

        prompt_embedding = model.encode(prompt)
        continuation_embedding = model.encode(continuation)

        alignment = util.pytorch_cos_sim(prompt_embedding, continuation_embedding).item()
        semantic_alignments.append(alignment)
    
    print(f"Prompts:", prompts_f.split('/')[-1])
    print(f"Outputs:", outputs_f.split('/')[-1])
    print(f"Average Semantic Alignment:", sum(semantic_alignments) / len(semantic_alignments))



    