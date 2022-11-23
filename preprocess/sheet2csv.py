import argparse
import csv
import re

from classopt import classopt, config
import pandas as pd

@classopt(default_long=True)
class Args:
    data_dir: str = config(long="--dir", short='-d', default='data/BilaS/S-set4/sentences/')
    csv_path: str = config(long="--csv", short="-c", default="S-set4_all.csv")
    output_path: str = config(long="--output", short="-o", default="sentences_S-set4.csv")
    delete_two_text: bool = config(long='--delete', short='-del', default=False)
    verNum: int = config(long='--ver', short='-v', default=1, choices=[1,2])

def delete_two_sentence(text):
    """
    'また'を含む文がある場合、
    'また'以降の文を排除して1文にする処理
    """
    text = str(text)
    if re.search(r'。また、', text):
        text = re.sub(r"。また、.*", r"", text)

    return text

def main():
    args = Args.from_args()

    df = pd.read_csv(args.data_dir+args.csv_path)
    df = df[2:]

    df_is_col = df[df["isColl"] == 1]
    print(type(df_is_col))
    
    Ver = f'SentenceVer{args.verNum}'

    if args.delete_two_text:
        df_is_col[Ver] = df_is_col[Ver].map(delete_two_sentence)
    
    df_is_col = df_is_col[["Scene", Ver]]

    columns = ['Scene', 'Sentence']
    df_is_col.columns = columns

    df_is_col.to_csv(args.data_dir+args.output_path, index=False)

if __name__ == "__main__":
    main()
