import json
from statistics import mean, stdev

import nltk
from tqdm import tqdm
from classopt import classopt, config

@classopt(default_long=True)
class Args:
    input_file: str = config(long="--input", short='-i', default="data/BilaS/bilas_train_mecab.jsonl")
    top_num: int = config(long="--top", short='-t', default=5)


def main():
    args: Args = Args.from_args()

    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    
    lengths = []
    for line in tqdm(lines, desc="Calc Max Length..."):
        line = json.loads(line)
        text = nltk.tokenize.word_tokenize(line['parse_sentence'])
        lengths.append(len(text))
    
    lengths = sorted(lengths, reverse=True)
    
    print(f"Result Top{args.top_num} Length >> {lengths[:args.top_num]}")
    print(f"Show Less Top{args.top_num} Length >> {lengths[-args.top_num:]}")
    print(f"Show Mean Length >> {mean(lengths)}")
    print(f"Show Std  Length  >> {stdev(lengths)}")


if __name__ == "__main__":
    main()