from typing import Any, Optional, Callable, Tuple, Dict
import csv
import json
import os
from pathlib import Path

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BilaS(Dataset):
    """
    """
    def __init__(
        self,
        root: str,
        params: Dict[str, Any],
        data_file: str = "bilas_mecab.jsonl",
        transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = root
        self.data_file = data_file
        self.transforms = transforms

        self.bildata_path = Path(self.root, "BilaS")
        self.pondata_path = Path(self.root, "Ponnet")

        self.data_path = Path(self.bildata_path, self.data_file)

        self.images_rgb = []
        self.images_depth = []
        self.targets_rgb = []
        self.targets_depth = []
        self.attention_maps_rgb = []
        self.attention_maps_depth = []
        self.sentences = []

        with open(self.data_path, 'r') as f:
            lines = f.read()
            lines = lines.split('\n')[:-1]
            for line in lines:
                df = json.loads(line)

                self.images_rgb.append(Path(self.pondata_path, df['image_rgb']))
                self.images_depth.append(Path(self.pondata_path, df['image_depth']))
                self.targets_rgb.append(Path(self.pondata_path, df['target_rgb']))
                self.targets_depth.append(Path(self.pondata_path, df['target_depth']))
                self.attention_maps_rgb.append(Path(self.pondata_path, df['attention_map_rgb']))
                self.attention_maps_depth.append(Path(self.pondata_path, df['attention_map_depth']))
                self.sentences.append(df['parse_sentence'])

    def __getitem__(self, index) -> Tuple[Any, Any]:
        # カメラ視点画像
        image_rgb = Image.open(self.images_rgb[index]).convert("RGB")
        image_depth = Image.open(self.images_depth[index]).convert("RGB")

        # 対象物体
        target_rgb = Image.open(self.targets_rgb[index]).convert("RGB")
        target_depth = Image.open(self.targets_depth[index]).convert("RGB")

        # attention map
        attention_map_rgb = Image.open(self.attention_maps_rgb[index]).convert('RGB')
        attention_map_depth = Image.open(self.attention_maps_depth[index]).convert("RGB")

        # 説明文
        sentence = self.sentences[index]

        return (image_rgb, image_depth), (target_rgb, target_depth), (attention_map_rgb, attention_map_depth), sentence

    def __len__(self):
        return len(self.images_rgb)

def main():
    '''
    Datasetの確認
    '''
    params = {
        "image_rgb": [0.5, 0.5],
        "image_depth": [0.5, 0.5],
        "target_rgb": [0.5, 0.5],
        "target_depth": [0.5, 0.5],
        "attention_rgb": [0.5, 0.5],
        "attention_depth": [0.5, 0.5],
    }
    datasets = BilaS(
        root = "/home/initial/RFCM/RelationalFutureCaptioningModel/data/", params=params, 
    )
    dataloader = DataLoader(
        dataset=datasets,
        batch_size = 2,
        shuffle=True,
    )

    print('------ len dataloader ------')
    print(len(dataloader))

    for images, targets, attention_maps, sentence in tqdm(dataloader):
        print(images)
        print(targets)
        print(attention_maps)
        print(sentence)

if __name__ == '__main__':
    main()