import argparse
from cmath import nan
import csv
from turtle import pd

from classopt import classopt, config
import pandas as pd

@classopt(default_long=True)
class Args:
    data_dir: str = config(long="--dir", short='-d', default='data/BilaS/S-set4/sentences/')
    csv_path: str = config(long="--csv", short="-c", default="S-set4_all.csv")
    output_path: str = config(long="--output", short="-o", default="sentences_S-set4.csv")

def main():
    args = Args.from_args()

    df = pd.read_csv(args.data_dir+args.csv_path)
    df = df[2:]

    df_is_col = df[df["isColl"] == 1]
    df_is_col = df_is_col[["Scene", "SentenceVer1"]]

    df_is_col.to_csv(args.data_dir+args.output_path, index=False)

if __name__ == "__main__":
    main()
