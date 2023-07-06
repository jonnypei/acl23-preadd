import argparse
import json
import numpy as np
from scipy.stats import ttest_rel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=str, required=True, nargs='+')
    args = parser.parse_args()

    fname_individual_results = {}
    for fname in args.outputs:
        with open(fname, 'r') as f:
            # lines = [json.loads(line) for line in f]
            lines = json.load(f)

        tox = []
        cont_tox = []
        ppl = []
        total = len(lines)
        for line in lines:
            tox.append(line['api_scores']['TOXICITY'])
            cont_tox.append(line['continuation_api_scores']['TOXICITY'])
            ppl.append(line['davinci_continuation_perplexity'])

        print("Statistics of", fname)
        print(f"    Sample size: {total}")
        print(f"    Average toxicity: {sum(tox) / total}")
        print(f"    Average toxicity of continuations: {sum(cont_tox) / total}")
        print(f"    Std toxicity of continuations: {np.std(cont_tox)}")
        print(f"    Average perplexity of continuations: {sum(ppl) / total}\n")

        fname_individual_results[fname] = (tox, cont_tox, ppl)

    if len(args.outputs) > 1:
        print('avg tox p value', "{:.6f}".format(ttest_rel(
            fname_individual_results[args.outputs[0]][0], fname_individual_results[args.outputs[1]][0]).pvalue))
        print('avg cont tox p value', "{:.3E}".format(ttest_rel(
            fname_individual_results[args.outputs[0]][1], fname_individual_results[args.outputs[1]][1]).pvalue))
        print('avg ppl p value', "{:.6f}".format(ttest_rel(
            fname_individual_results[args.outputs[0]][2], fname_individual_results[args.outputs[1]][2]).pvalue))
