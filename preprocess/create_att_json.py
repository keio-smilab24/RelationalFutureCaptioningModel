from glob import glob
import os
from pathlib import Path
import json
import csv

from tqdm import tqdm
from classopt import classopt, config

@classopt(default_long=True)
class Args:
    root: str = config(long='--root', short='-r', default='/home/initial/RFCM/RelationalFutureCaptioningModel/data/Ponnet/S-set4/')
    json_path: str = config(long='--json_path', short='-jp', default="/home/initial/RFCM/RelationalFutureCaptioningModel/data/Ponnet/S-set4/att_path.jsonl")
    csv_path: str = config(long='--csv_path', short='-cp', default="/home/initial/RFCM/RelationalFutureCaptioningModel/data/Ponnet/S-set4/att_path.csv")
    imageNum: int = 2000
    make_json: bool = config(short='--json', default=False)
    make_csv: bool = config(short='--csv', default=False)


def main():
    '''
    att_gray_SCENE_rgb.jpg
    att_gray_SCENE_depth.jpg
    att_map_SCENE_rgb.jpg
    att_map_SCENE_depth
    over_SCENE_rgb.jpg
    over_SCENE_depth.jpg
    '''
    args: Args = Args.from_args()

    for id in tqdm(range(1, args.imageNum+1)):

        attentions = sorted(glob(args.root+'attentions/'+f"*_{id}_*.jpg"))
        assert len(attentions)==6

        att_gray_rgb = str(Path('/'.join(attentions[1].split('/')[-3:])))
        att_gray_depth = str(Path('/'.join(attentions[0].split('/')[-3:])))

        att_map_rgb = str(Path('/'.join(attentions[3].split('/')[-3:])))
        att_map_depth = str(Path('/'.join(attentions[2].split('/')[-3:])))

        over_rgb = str(Path('/'.join(attentions[5].split('/')[-3:])))
        over_depth = str(Path('/'.join(attentions[4].split('/')[-3:])))
        
        att_ave = str(Path("/".join(glob(args.root+'attentions/'+f'*_{id}.jpg')[0].split('/')[-3:])))
        
        if args.make_json:
            att_dict = {}
            att_dict["id"] = id
            att_dict["att_gray_rgb"] = att_gray_rgb
            att_dict["att_gray_depth"] = att_gray_depth
            att_dict["att_map_rgb"] = att_map_rgb
            att_dict["att_map_depth"] = att_map_depth
            att_dict["over_rgb"] = over_rgb
            att_dict["over_depth"] = over_depth
            att_dict["att_ave"] = att_ave
            
            with open(args.json_path, "a") as wf:
                wf.write(json.dumps(att_dict, ensure_ascii=False))
        
        if args.make_csv:
            att_csv = [id, att_gray_rgb, att_gray_depth, att_map_rgb, att_map_depth, over_rgb, over_depth, att_ave]
            
            with open(args.csv_path, 'a') as wf:
                writer = csv.writer(wf)
                writer.writerow(att_csv)


if __name__ == "__main__":
    main()

