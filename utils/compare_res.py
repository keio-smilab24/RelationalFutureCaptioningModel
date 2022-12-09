"""
エラー分析用スクリプト
結果のファイルからタイムスタンプ,生成文,正解文を取り出して
出力用ファイルにクリップごとに並べて出力
結果ディレクトリ:./results/XXX/caption

例: python utils/compare_res.py -i results/linear_to_cnn/caption/translations_29_test.json -o results/linear_to_cnn/compare.csv
"""

import sys
import csv
import json

from classopt import classopt, config

@classopt(default_long=True)
class Args:
    input_path: str = config(long='--input', short='-i', type=str)
    output_path: str = config(long="--output", short="-o", type=str)

def compare_res(input_path, output_path):
    """
    Args:
        res_path(string): 結果のファイルパス(json)
        out_path(string): エラー分析のための出力ファイルパス(csv)
    """
    with open(input_path, "r") as f:
        results = json.load(f)
    
    results = results["results"]
    outputs = []
    header = ["id", "gen_sent", "gt_sent"]
    outputs.append(header)

    for id in results:
        texts = results[id]
        for clip in texts:
            output = []
            output.append(clip["clip_id"])
            output.append(clip["sentence"])
            output.append(clip["gt_sentence"][0])
            outputs.append(output)

    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(outputs)


def main():
    args: Args = Args.from_args()
    compare_res(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
