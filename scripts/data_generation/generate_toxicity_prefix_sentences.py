import json
from glob import glob
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER


def top_toxicity_sentences(num_sentences=1000, display=False):
    """Extracts the top realtoxicity sentences based on each attribute score"""

    data = []
    with open("data/realtoxicity/realtoxicityprompts.jsonl", "r") as f:
        for line in f:
            line_info = json.loads(line)

            line_info["prompt"]["text"] += line_info["continuation"]["text"]
            # Will need to adjust later for more accurate API scores
            for attribute in PERSPECTIVE_API_ATTRIBUTES_LOWER:
                if line_info["prompt"][attribute] and line_info["continuation"][attribute]:
                    line_info["prompt"][attribute] = max(
                        line_info["prompt"][attribute], line_info["continuation"][attribute])

            data.append(line_info["prompt"])

    with open("data/realtoxicity/toprealtoxicitysentences.txt", "w") as f:
        top_sentences = set()
        for score in PERSPECTIVE_API_ATTRIBUTES_LOWER:
            curr_data = [sentence for sentence in data if sentence[score]]
            curr_data = sorted(curr_data, key=lambda x: -
                               x[score])[:num_sentences]
            sentences = [sentence["text"].strip() for sentence in curr_data]
            top_sentences.update(sentences)

            if display:
                print(f"Top 10 sentences according to '{score}' score:\n")
                print(sentences[:10])
                print("\n===\n")
        f.write("\n".join(top_sentences))

    with open("data/realtoxicity/allrealtoxicitysentences.jsonl", "w") as f:
        for sentence in data:
            json.dump({"text": sentence["text"],
                      "toxicity": sentence["toxicity"]}, f)
            f.write("\n")

    return top_sentences


def toxigen_sentences():
    """Extracts all the unique sentences in the TOXIGEN dataset"""

    unique_sentences = set()
    for name in glob("data/TOXIGEN/*"):
        if name[len("data/TOXIGEN/"):len("data/TOXIGEN/")+4] != "hate":
            continue
        with open(name, "r") as f:
            text = [s.strip() + "." for s in f.read().split("\\n-")]
            text = [s[2:] if s[:2] == "- " else s for s in text]
            text = [s[0].upper() + s[1:] for s in text]
            unique_sentences.update(text)
    with open("data/TOXIGEN/all_TOXIGEN_sentences.txt", "w") as f:
        for s in unique_sentences:
            f.write(s + "\n")
    return unique_sentences


if __name__ == "__main__":
    toxicity_data = top_toxicity_sentences(num_sentences=100)
    toxigen_data = toxigen_sentences()

    all_prefix_data = toxicity_data.union(toxigen_data)
    with open("data/toxicity_prefix_sentences.txt", "w") as f:
        f.write("\n".join(all_prefix_data))
