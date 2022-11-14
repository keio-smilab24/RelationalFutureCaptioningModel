import argparse
from cmath import nan
import csv
from turtle import pd

from classopt import classopt
import pandas as pd

@classopt(default_long=True)
class Args:
    data_dir: str = "/home/initial/RFCM/RelationalFutureCaptioningModel/data/S-set4/sentences/"
    csv_path: str = "S-set4_1001to1500.csv"
    save_path: str = "sentences_1001to1500.csv"

def main():
    args = Args.from_args()

    df = pd.read_csv(args.data_dir+args.csv_path)
    df = df[2:]

    df_is_col = df[df["collision"] == 1]
    df_is_col = df_is_col[["SCENE", "Ver1"]]

    df_is_col.to_csv(args.data_dir+args.save_path, header=None)

if __name__ == "__main__":
    main()
