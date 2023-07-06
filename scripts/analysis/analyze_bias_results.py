import argparse
import json
from scipy.stats import ttest_rel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=str, required=True, nargs='+')
    args = parser.parse_args()

    fname_ratio_diffs = {}
    for fname in args.outputs:
        print(fname)
        with open(fname, 'r') as f:
            data = json.load(f)

        print("Output Profession Statistics:")
        keys = sorted(list(data.keys()))
        total = 0
        ratio_diff = 0
        ratio_diffs = []
        for key in keys:
            ratio = data[key][0] / (data[key][0] + data[key][1])
            total += data[key][0] + data[key][1]
            ratio_diff += abs(ratio - 0.5)
            ratio_diffs.append(abs(ratio - 0.5))
            print(f"    {key}: {ratio}")

        print("\nStatistics of", fname)
        print('     Sample size', total)
        print('     Average ratio diff from 0.5:', ratio_diff / len(keys), "\n")

        fname_ratio_diffs[fname] = ratio_diffs

    if len(args.outputs) > 1:
        print(ttest_rel(fname_ratio_diffs[args.outputs[0]],
              fname_ratio_diffs[args.outputs[1]]).pvalue)
        print('p value', "{:.3E}".format(ttest_rel(
            fname_ratio_diffs[args.outputs[0]], fname_ratio_diffs[args.outputs[1]]).pvalue))
