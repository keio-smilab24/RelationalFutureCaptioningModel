import json

from classopt import classopt, config


@classopt(default_long=True)
class Args:
    root: str = config(long='--root', short='-r', default='data/BilaS')
    input_file: str = config(long="--input", short='-i', default='bilas_valid_mecab.jsonl')
    output_file: str = config(long='--output', short='-o', default="caption_valid.json")

def main():
    args: Args = Args.from_args()

    input_path = args.root + args.input_file
    output_path = args.root + args.output_file

    with open(input_path) as f:
        lines = f.read()
        lines = lines.split('\n')
    
    captions = dict()
    for line in lines:
        data = json.loads(line)
        captions[data['scene']] = data['parse_sentence']
    

    with open(output_path, 'a') as f:
        f.write(json.dumps(captions, ensure_ascii=False))

if __name__ == "__main__":
    main()