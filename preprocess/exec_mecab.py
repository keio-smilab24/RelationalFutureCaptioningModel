from email.policy import default
import json
import MeCab

# pip install mecab-python3==1.0.3
# pip install unidic-lite==1.0.8

from classopt import classopt, config

@classopt(default_long=True, default_short=True)
class Args:
    input_path: str = config(long="--input", short="-i", default="/home/taku/RFCM/RelationalFutureCaptioningModel/datasets/BillaS/billas.jsonl")
    output_path: str = config(long="--output", short="-o", default="/home/taku/RFCM/RelationalFutureCaptioningModel/datasets/BillaS/billas_mecab.jsonl")
    delete_list: str = [".", "?", "!", "。", "？", "！"]


def main():
    args = Args.from_args()

    m = MeCab.Tagger('-Owakati')

    with open(args.input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            sentence = line['sentence']

            m_sentence = m.parse(sentence)
            m_sentence = m_sentence.replace(' \n', '')

            line['parse_setntence'] = m_sentence

            with open(args.output_path, "a") as output:
                json.dump(line, output, ensure_ascii=False)
                output.write('\n')


if __name__ == "__main__":
    main()