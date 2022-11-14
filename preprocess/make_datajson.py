import json
import os

from classopt import classopt
import pandas as pd

@classopt(default_long=True)
class Args:
    root: str = "/home/initial/RFCM/RelationalFutureCaptioningModel/datasets/BillaS/"
    sentence_path: str = "/home/initial/RFCM/RelationalFutureCaptioningModel/datasets/BillaS/S-set4/sentences/sentences_1001to1500.csv"
    save_path: str = "/home/initial/RFCM/RelationalFutureCaptioningModel/datasets/BillaS/billas.jsonl"

def main():
    """
    Billa-Sのdata関連をまとめたjsonファイルを作成する
    
    Contents:
        image_rgb: カメラ視点のRGB画像へのpath
        image_depth: カメラ視点のdepth画像へのpath
        target_rgb: 対象物体のrgb画像へのpath
        target_depth: 対象物体のdepth画像へのpath
        attention_map: Ponnetの出力attention map画像へのpath
        sentence: 説明文
    """

    args: Args = Args.from_args()

    billas = dict()

    # カメラ視点の画像
    images_rgb = os.path.join(args.root, "S-set4/images/1002_rgb.jpg")
    images_depth = os.path.join(args.root, "S-set4/images/1002_depth.jpg")
    
    # 対象物体の画像
    df = pd.read_csv(os.path.join(args.root, 'S-set4/sim_x_test.csv'), header=None, index_col=None)
    target_rgb = os.path.join(args.root, df.iloc[3, 2])
    target_depth = os.path.join(args.root, df.iloc[3, 3])
    
    # attention map
    attention_map_rgb = os.path.join(args.root, "S-set4/attention_maps/over_1002_rgb.jpg")
    attention_map_depth = os.path.join(args.root, "S-set4/attention_maps/over_1002_depth.jpg")

    # 説明文
    sentence_df = pd.read_csv('/home/initial/RFCM/RelationalFutureCaptioningModel/datasets/BillaS/S-set4/sentences/sentences_1001to1500.csv', header=None, index_col=None)
    columns = ["ID", "SCENE", "SENTENCE"]
    sentence_df.columns = columns
    sentence = sentence_df[sentence_df["SCENE"] == 1002]["SENTENCE"][1]
    
    billas['image_rgb'] = images_rgb
    billas['image_depth'] = images_depth
    billas['target_rgb'] = target_rgb
    billas['target_depth'] = target_depth
    billas['attention_map_rgb'] = attention_map_rgb
    billas['attention_map_depth'] = attention_map_depth
    billas['sentence'] = sentence
    
    with open(args.save_path, 'a') as f:
        f.write(json.dumps(billas, ensure_ascii=False))

if __name__ == "__main__":
    main()