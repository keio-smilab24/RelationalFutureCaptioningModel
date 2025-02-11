"""
Captioning dataset.
"""
import copy
import json
import csv
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import nltk
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import cv2

from utils.configs import Config


def make_dict(train_caption_file, word2idx_filepath, datatype: str="bila"):
    """
    Summary:
        学習集合に含まれる単語から辞書(word2id)を作成する
    """
    if datatype == "bila":
        max_words = 0
        sentence_list = []
        words = []
        with open(train_caption_file) as f:
            sentence_dict = json.load(f)
        for sample in sentence_dict:
            sentence_list.append(sample["sentence"])
        for sent in sentence_list:
            word_list = nltk.tokenize.word_tokenize(sent)
            max_words = max(max_words, len(word_list))
            words.extend(word_list)

    elif datatype == "bilas":
        train_caption_file = "data/BilaS/bilas_train_mecab.jsonl"
        labels_file = "data/BilaS/labels.txt"
        word2idx_filepath = "data/BilaS/ponnet_word2idx.json"
        max_words = 0
        sentence_list = []
        words = []
        with open(train_caption_file) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            sentence_list.append(line['parse_sentence'])
        for sent in sentence_list:
            word_list = nltk.tokenize.word_tokenize(sent)
            max_words = max(max_words, len(word_list))
            words.extend(word_list)
        with open(labels_file) as f:
            lines = f.readlines()
        for line in lines:
            words.append(line.replace("\n",""))


    # default dict
    word2idx_dict =\
        {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[VID]": 3, "[BOS]": 4, "[EOS]": 5, "[UNK]": 6}
    word_idx = 7

    # 辞書の作成
    for word in words:
        if word not in word2idx_dict:
            word2idx_dict[word] = word_idx
            word_idx += 1

    # 辞書ファイルの作成
    with open(word2idx_filepath, "w") as f:
        if datatype == 'bila':
            json.dump(word2idx_dict, f, indent=0)
        elif datatype == 'bilas':
            json.dump(word2idx_dict, f, indent=0, ensure_ascii=False)


class BilaDataset(data.Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    def __init__(
        self,
        dataset_name: str,
        max_t_len: int,
        max_v_len: int,
        max_n_sen: int,
        mode: str="train",
        annotations_dir: str = "data",
        datatype: str = "bila",
        clip_dim: int = 768,
    ):
        self.datatype = datatype
        self.clip_dim = clip_dim

        # clip normalize para
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        # metadata settings
        self.dataset_name = dataset_name # BILA
        if datatype == "bila":
            self.annotations_dir = Path(annotations_dir) # annotations
        elif datatype == 'bilas':
            self.annotations_dir = Path('data/BilaS/')

        # Video feature settings
        self.num_images = max_v_len - 2

        # Parameters for sequence lengths
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len
        self.max_n_sen = max_n_sen

        # Train or val mode
        self.mode = mode

        # determine metadata file
        tmp_path = "BILA"
        self.mode_bilas = "val"
        if mode == "train":
            if datatype == "bila":
                data_path = self.annotations_dir / tmp_path / "captioning_train.json" # annotations/BILA/captioning_train.json
            elif datatype == 'bilas':
                data_path = self.annotations_dir / 'bilas_train_mecab.jsonl'
        elif mode == "val":
            if datatype == "bila":
                data_path = self.annotations_dir / tmp_path / "captioning_val.json" # annotations/BILA/captioning_train.json
            elif datatype == 'bilas':
                data_path = self.annotations_dir / 'bilas_valid_mecab.jsonl'
        elif mode == "test":
            if datatype == "bila":
                data_path = self.annotations_dir / tmp_path / "captioning_test.json" # annotations/BILA/captioning_train.json
            elif datatype == 'bilas':
                data_path = self.annotations_dir / 'bilas_test_mecab.jsonl'
            mode = "val"
            self.mode = "val"
            self.mode_bilas = "test"
        else:
            raise ValueError(
                f"Mode must be [train, val] for {self.dataset_name}, got {mode}"
            )

        if datatype == 'bila':
            self.word2idx_file = (
                self.annotations_dir / self.dataset_name / "ponnet_word2idx.json"
            )
        elif datatype == "bilas":
            self.word2idx_file = Path(self.annotations_dir, "ponnet_word2idx.json")

        if not os.path.exists(self.word2idx_file):
            make_dict(data_path, self.word2idx_file, datatype)

        self.word2idx = json.load(self.word2idx_file.open("rt", encoding="utf8")) # 辞書 {word : ID(int)}
        self.idx2word = {int(v): k for k, v in list(self.word2idx.items())} # 逆
        print(f"WORD2IDX: {self.word2idx_file} len {len(self.word2idx)}")

        # load and process captions and video data
        # clip_id, sentence
        coll_data = []
        if datatype == "bila":
            with open(data_path) as f:
                raw_data = json.load(f)
            for line in tqdm(raw_data):
                coll_data.append(line)

        elif datatype == "bilas":
            with open(data_path) as f:
                lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                raw_data = {
                    'clip_id': line['setNum']+"_"+line["scene"],
                    'sentence': line['parse_sentence'],
                }
                coll_data.append(raw_data)

        self.data = coll_data

        # map video id and clip id to clip number
        self.clip_nums = []
        for clip in tqdm(range(len(self.data))):
            self.clip_nums.append(str(clip)) # ['0', '1', ..., '999']

        print(f"Dataset {self.dataset_name} #{len(self)} {self.mode}")

    def __len__(self):
        return len(self.data)
        # return int(len(self.data)/40)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def _load_bbox_feats(self, raw_name: str):
        if self.datatype == 'bilas':
            # N_XXX -> N / XXX
            setNum = raw_name.split('_')[0]
            scene  = int(raw_name.split('_')[-1])

            data_dir = os.path.join("data", "Ponnet", f"S-set{setNum}")

            bbox_file_path = os.path.join(data_dir, "csv_bbox", f'{scene:04}_rgb_bbox.csv')
            bbox_feat_path = os.path.join(data_dir, "csv_feature", f'{scene:04}_rgb_features.csv')
            label_path = os.path.join(data_dir, "csv_label", f'{scene:04}_rgb_label.csv')

            bbox_list = self._load_from_csv(bbox_file_path)
            bbox_feats_list = self._load_from_csv(bbox_feat_path)
            label_list = self._load_label_from_csv(label_path)

            # 検出したbboxの数を計算
            num_bb = len(bbox_feats_list)

            # bboxが0個の場合空の値で補間
            if num_bb == 0:
                # frcnnの次元: 1024
                bbox_feats_list = [[0]*1024 for _ in range(1)]
                bbox_list = [[0]*6 for _ in range(1)]
                label_list = [""]

            bboxes = np.asarray(bbox_list)
            bbox_feats = np.asarray(bbox_feats_list)

        return bboxes, bbox_feats, label_list

    def _load_label_from_csv(self, file_path: str):
        """
        Summary:
            csvファイルからデータを読み込む
        Args:
            file_path: ファイルパス
        Return:
            データが格納されたリスト
        """
        outputs = []
        with open(file_path) as f:
            csv_reader = csv.reader(f)
            for row_str in csv_reader:
                outputs.append(row_str[0])
        return outputs

    def _load_from_csv(self, file_path: str):
        """
        Summary:
            csvファイルからデータを読み込む
        Args:
            file_path: ファイルパス
        Return:
            データが格納されたリスト
        """
        outputs = []
        with open(file_path) as f:
            csv_reader = csv.reader(f)
            for row_str in csv_reader:
                row_float = [float(s) for s in row_str]
                outputs.append(row_float)
        return outputs

    def _load_ponnet_img_feature(
        self,
        raw_name: str,
        base_frame: float=4.2,
        interval: float=0.2,
    ) -> Tuple[np.array, np.array, List[np.array]]:
        """
        Summary:
            bila:
                画像を格納したリスト・t+1の画像・最も前の画像を返す
            bilas:
                カメラ視点のrgbd・targetのrgbd・attentionのrgbdの6枚を格納したリストを返す
                t+1, 最も前の画像はカメラ視点のrgbとしている
        Args:
            raw_name   : Video ID | Scene
            num_images : 画像の枚数
            base_frame : bilaデータセットのtとなる秒数
            interval   : bilaデータセットにおいて各画像の差分時間
        """
        if self.datatype == "bila":
            data_dir = os.path.join("data", "BILA", "ponnet_data")
            frame = base_frame

            img_list = []
            for _ in range(min(self.num_images,6)):
                frame -= interval
                img_path = os.path.join(data_dir, f"{frame:.1f}s_center_frames", raw_name+".png")
                img = torch.from_numpy(cv2.imread(img_path).astype(np.float32)).clone()
                img_list.insert(0, img)

            rec_img = cv2.imread(os.path.join(data_dir, f"{frame:.1f}s_center_frames", raw_name+".png"))
            rec_img = cv2.resize(rec_img, dsize=(16, 16))

        elif self.datatype == "bilas":
            setNum = raw_name.split('_')[0]
            scene  = raw_name.split('_')[-1]
            img_list = []

            if self.mode == "train":
                datapath = 'data/BilaS/bilas_train_mecab.jsonl'
            elif self.mode == "val":
                datapath = 'data/BilaS/bilas_valid_mecab.jsonl'

            if self.mode_bilas == 'test':
                datapath = 'data/BilaS/bilas_test_mecab.jsonl'

            df = pd.read_json(datapath, orient='records', lines=True)
            df_scene = df[df["setNum"] == int(setNum)]
            df_scene = df_scene[df_scene['scene']==int(scene)]

            img_list_path = [
                'image_rgb', 'image_depth', 'target_rgb', 'target_depth',
                'attention_map_rgb', 'attention_map_depth',
            ]

            img_list = []
            ponnet_path = Path('data/Ponnet')
            sam_rgb = cv2.resize(cv2.imread(str(Path(ponnet_path, df_scene['sam_rgb'].iloc[-1]))).astype(np.float32), dsize=(224, 224))
            sam_depth = cv2.resize(cv2.imread(str(Path(ponnet_path, df_scene['sam_depth'].iloc[-1]))).astype(np.float32), dsize=(224, 224))
            for idx in range(self.num_images):
                img_path = str(Path(ponnet_path, df_scene[img_list_path[idx]].iloc[-1]))
                if img_list_path[idx] == "image_rgb":
                    rec_img = cv2.imread(img_path).astype(np.float32)
                    rec_img = cv2.resize(cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB), dsize=(16, 16))
                img = cv2.resize(cv2.imread(img_path).astype(np.float32), dsize=(224, 224))
                if img_list_path[idx] == "attention_map_rgb":
                    img = cv2.addWeighted(img, 0.5, sam_rgb, 0.5, 0, dtype=cv2.CV_32F)
                if img_list_path[idx] != "attnetion_map_rgb":
                    img = img / 255.0
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img_list_path[idx] == "attention_map_depth":
                    img = cv2.addWeighted(img, 0.5, sam_depth, 0.5, 0, dtype=cv2.CV_32F)
                if img_list_path == "image_rgb":
                    img = sam_rgb
                if img_list_path == "image_depth":
                    img = sam_depth
                img = self.normalize_img(img)
                img = torch.from_numpy(img)
                img_list.append(img)

        return img_list, rec_img

    def normalize_img(self, img):
        """
        画像をnormalizeする
        """
        H,W,C = img.shape
        for ch in range(C):
            img[ch] = (img[ch] - self.mean[ch]) / self.std[ch]
        return img


    def convert_example_to_features(self, example):
        """
        example single snetence
        {
            "clip_id": str,
            "duration": float,
            "timestamp": [st(float), ed(float)],
            "sentence": str
        } or
        {
            "clip_id": str,
            "duration": float,
            "timestamps": list([st(float), ed(float)]),
            "sentences": list(str)
        }
        """
        raw_name = example["clip_id"] # clip_id : VideoID/SceneID
        img_list, rec_img = self._load_ponnet_img_feature(raw_name)

        bboxes, bbox_feats, labels = self._load_bbox_feats(raw_name)

        single_features = []
        single_meta = []
        # imageとtextを合わせた特徴やmask等の作成
        data, meta = self.clip_sentence_to_feature(
            example["clip_id"],
            example["sentence"],
            rec_img,
            img_list,
            bboxes,
            bbox_feats,
            labels,
        )

        single_features.append(data)
        single_meta.append(meta)

        return single_features, single_meta

    def clip_sentence_to_feature(
        self,
        name,
        sentence,
        gt_rec,
        img_list,
        bboxes,
        bbox_feats,
        labels,
    ):
        """
        make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            video_feats: Either np.array of rgb+flow features or Dict[str, np.array] of COOT embeddings
            clip_idx: clip number in the video (needed to loat COOT features)
        """

        # 画像の特徴量 + textの特徴量を合わせた形状のfeatと画像用のmaskを作成
        img_feats, txt_feats, video_tokens, video_mask = self._load_indexed_image_feature(img_list)

        # text用のtokenとmaskを作成
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)
        label_tokens = []
        for label in labels:
            label_tokens.extend(nltk.tokenize.word_tokenize(label))

        # img_tokenとtext_tokenを連結
        input_tokens = video_tokens + text_tokens

        # token(単語含む) -> id列
        input_ids = [
            self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens
        ]
        label_ids = [
            self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in label_tokens
        ]

        # text部分のみID, その他は-1のリストを作成
        # -1 はCElossの計算に含まれない
        input_labels = (
            [self.IGNORE] * len(video_tokens)
            + [ self.IGNORE if m == 0 else tid
                for tid, m in zip(input_ids[-len(text_mask):], text_mask)][1:]
            + [self.IGNORE]
        )

        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len

        labels = np.array(label_ids).astype(np.int64)
        labels = labels[..., np.newaxis]

        data = dict(
            name=name,
            input_tokens=input_tokens,
            input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            img_feats=img_feats.astype(np.float32),
            txt_feats=txt_feats.astype(np.float32),
            gt_rec = gt_rec,
            bboxes=bboxes.astype(np.float32),
            bbox_feats=bbox_feats.astype(np.float32),
            labels = labels
        )
        meta = dict(name=name, sentence=sentence)

        return data, meta


    def _load_indexed_image_feature(
        self,
        img_list: list,
    ):
        """
        Summary:
            imageのtokenとmask
            image + textの形状の特徴量を作成し、image部分に特徴量を入れる
        Args:
            img_list: 画像の特徴量を格納したリスト
        """

        # VIDEO_TOKEN : CLS, VID, ..., VID, SEP
        video_tokens = (
            [self.CLS_TOKEN]
            + [self.VID_TOKEN] * (self.max_v_len - 2)
            + [self.SEP_TOKEN]
        )

        # VIDEO関連のトークン部分を1とするmaskを作成
        # TODO : 質問ver3.4
        mask = [1] * self.max_v_len

        # img / text の特徴量を作成
        img_feats = np.zeros((self.max_v_len-2, *img_list[0].shape))
        txt_feats = np.zeros((self.max_t_len, self.clip_dim))


        for idx in range(len(img_list)):
            img_feats[idx] = img_list[idx]


        return img_feats, txt_feats, video_tokens, mask

    def _tokenize_pad_sentence(self, sentence):
        """
        Summary:
            text用のtokenとmaskを作成して返す
            token : BOS, Word1, ..., WordN, EOS, SEP, ..., SEP
            mask  :  1     1    ...    1     1    0   ...   0
        """
        # token作成
        text_tokens = nltk.tokenize.word_tokenize(sentence.lower())[: self.max_t_len - 2]
        text_tokens = [self.BOS_TOKEN] + text_tokens + [self.EOS_TOKEN]

        # 不足分はPADを追加
        num_words = len(text_tokens)
        mask = [1] * num_words + [0] * (self.max_t_len - num_words)
        text_tokens += [self.PAD_TOKEN] * (self.max_t_len - num_words)

        return text_tokens, mask

    def convert_ids_to_sentence(
        self, ids, rm_padding=True, return_sentence_only=True
    ) -> str:
        """
        A list of token ids
        """
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [
                self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]
            ]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)

    def collate_fn(self, batch):
        """
        Summary:
            DataLoaderにおけるcollate_fnの自作版
            batch内にbbox等で可変長の入力を持つときに自作する必要あり
        """
        # meta側
        raw_batch_meta = [e[1] for e in batch]
        batch_meta = []
        for e in raw_batch_meta:
            cur_meta = dict(name=None, timestamp=[], gt_sentence=[])
            for d in e:
                cur_meta["clip_id"] = d["name"]
                cur_meta["gt_sentence"].append(d["sentence"])
            batch_meta.append(cur_meta)

        # items側
        batch = [e[0] for e in batch]
        # Step1: pad each example to max_n_sen
        max_n_sen = max([len(e) for e in batch]) # 1

        raw_step_sizes = []
        padded_batch = []
        padding_clip_sen_data = copy.deepcopy(batch[0][0]) # doesn"t matter which one is used
        padding_clip_sen_data["input_labels"][:] = BilaDataset.IGNORE
        for ele in batch: # [data{}] (length 1)
            cur_n_sen = len(ele)
            if cur_n_sen < max_n_sen:
                # noinspection PyAugmentAssignment
                ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
            raw_step_sizes.append(cur_n_sen)
            padded_batch.append(ele)

        # Step2: batching each steps individually in the batches
        collated_step_batch = []
        for step_idx in range(max_n_sen):
            collated_step = step_collate([e[step_idx] for e in padded_batch])
            collated_step_batch.append(collated_step)
        return collated_step_batch, raw_step_sizes, batch_meta


def pad_tensors(key, inputs, lens=None, pad=0):
    """
    Summary:
        batch内の可変長のデータ(bbox等)の形状を0埋めして揃える関数
    Args:
        keys   : 入力データ(辞書)のkey (bboxed or bbox_feats)
        inputs : 実際のbatch中の入力データ
        lens   : 最大形状列
        pad    : 穴埋めする値
    """
    # 最大数を計算
    if lens is None:
        lens = [e.shape[0] for e in inputs]
    max_len = max(lens)

    # batch_size
    bs = len(inputs)

    # 配列の作成
    dim = inputs[0].shape[-1]
    dtype = inputs[0].dtype
    output = np.zeros((bs, max_len, dim), dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(inputs, lens)):
        output[i, :l, :] = t.data
    if key == "bboxes":
        output = torch.from_numpy(output)
        # 面積情報の追加
        output = torch.cat([output, output[:,:,4:5]*output[:,:,5:]], dim=-1)
        return output

    else:
        return torch.from_numpy(output)



def prepare_batch_inputs(batch, use_cuda: bool, non_blocking=False):
    """
    各入力データをcudaに乗せる
    """
    inputs = dict()
    batch_size = len(batch["name"])
    for k, v in list(batch.items()):
        assert batch_size == len(v), (batch_size, k, v)
        if use_cuda:
            if isinstance(v, torch.Tensor):
                v = v.cuda(non_blocking=non_blocking)
        inputs[k] = v
    return inputs


def step_collate(padded_batch_step):
    """
    The same step (clip-sentence pair) from each example
    """
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        elif key in ["bboxes", "bbox_feats", "labels"]:
            c_batch[key] = pad_tensors(key, [d[key] for d in padded_batch_step])
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def create_datasets_and_loaders(
    cfg: Config,
    annotations_dir: str,
    datatype: str = 'bila',
) -> Tuple[BilaDataset, BilaDataset, DataLoader, DataLoader]:
    # create the dataset
    dset_name_train = cfg.dataset_train.name
    train_dataset = BilaDataset(
        dset_name_train,
        cfg.max_t_len,
        cfg.max_v_len,
        cfg.max_n_sen,
        mode="train",
        annotations_dir=annotations_dir,
        datatype=datatype,
        clip_dim=cfg.clip_dim,
    )
    # add 10 at max_n_sen to make the inference stage use all the segments
    # max_n_sen_val = cfg.max_n_sen + 10
    max_n_sen_val = cfg.max_n_sen
    val_dataset = BilaDataset(
        cfg.dataset_val.name,
        cfg.max_t_len,
        cfg.max_v_len,
        max_n_sen_val,
        mode="val",
        annotations_dir=annotations_dir,
        datatype=datatype,
        clip_dim=cfg.clip_dim,
    )

    train_loader = data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.dataset_train.shuffle,
        num_workers=cfg.dataset_train.num_workers,
        pin_memory=cfg.dataset_train.pin_memory,
    )
    val_loader = data.DataLoader(
        val_dataset,
        collate_fn=val_dataset.collate_fn,
        batch_size=cfg.val.batch_size,
        shuffle=cfg.dataset_val.shuffle,
        num_workers=cfg.dataset_val.num_workers,
        pin_memory=cfg.dataset_val.pin_memory,
    )
    test_dataset = BilaDataset(
        cfg.dataset_val.name,
        cfg.max_t_len,
        cfg.max_v_len,
        max_n_sen_val,
        mode="test",
        annotations_dir=annotations_dir,
        datatype=datatype,
        clip_dim=cfg.clip_dim,
    )
    test_loader = data.DataLoader(
        test_dataset,
        collate_fn=val_dataset.collate_fn,
        batch_size=cfg.val.batch_size,
        shuffle=cfg.dataset_val.shuffle,
        num_workers=cfg.dataset_val.num_workers,
        pin_memory=cfg.dataset_val.pin_memory,
    )

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader
