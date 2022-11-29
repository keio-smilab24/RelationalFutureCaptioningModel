"""
Captioning dataset.
"""
import copy
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import nltk
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import cv2

from mart.configs_mart import MartConfig, MartPathConst
from nntrainer.typext import ConstantHolder


class DataTypesConstCaption(ConstantHolder):
    """
    Possible video data types for the dataset:
    Video features or COOT embeddings.
    """

    VIDEO_FEAT = "video_feat"
    COOT_EMB = "coot_emb"


def make_dict(train_caption_file, word2idx_filepath, datatype: str="bila"):
    '''
    Args:
        Before:
            train_caption_file: annotations/BILA/captioning_train.json
            word2idx_filepath: annotations/ponnet_word2idx.json
        After:
            train_caption_file: data/BillaS/bilas_mecab.jsonl
            word2idx_filepath: data/BillaS/ponnet_word2idx.json
    '''

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
        recurrent: bool=True,
        untied: bool=False,
        video_feature_dir: Optional[str] = None,
        annotations_dir: str = "data",
        preload: bool = False,
        datatype: str = "bilas",
    ):
        self.datatype = datatype

        # metadata settings
        self.dataset_name = dataset_name # BILA
        if datatype == "bila":
            self.annotations_dir = Path(annotations_dir) # annotations
        elif datatype == 'bilas':
            self.annotations_dir = Path('data/BilaS/')

        # Video feature settings
        self.video_feature_dir = Path(video_feature_dir) / self.dataset_name # data/mart_video_feature/BILA
        self.num_images = max_v_len - 2

        # Parameters for sequence lengths
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len
        self.max_n_sen = max_n_sen

        # Train or val mode
        self.mode = mode        # train
        self.preload = preload  # False

        # Recurrent or untied, different data styles for different models
        self.recurrent = recurrent  # True
        self.untied = untied        # False
        assert not (
            self.recurrent and self.untied
        ), "untied and recurrent cannot be True for both"

        # ---------- Load metadata ----------

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

        # ---------- Load video data ----------

        # Decide whether to load COOT embeddings or video features
        
        # COOT embeddings
        self.data_type = DataTypesConstCaption.COOT_EMB # coot_emb
        
        # map video id and clip id to clip number
        self.clip_nums = []
        for clip in tqdm(range(len(self.data))):
            self.clip_nums.append(str(clip)) # ['0', '1', ..., '999']

        self.frame_to_second = None  # Don't need this for COOT embeddings # None

        print(
            f"Dataset {self.dataset_name} #{len(self)} {self.mode} input {self.data_type}"
        )

        self.preloading_done = False

    def __len__(self):
        # return len(self.data)
        return int(len(self.data)/20)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def _load_ponnet_video_feature(
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
                img = img.reshape(-1, 150528)
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
                'attention_map_rgb', 'attention_map_depth'
            ]
            
            img_list = []
            ponnet_path = Path('data/Ponnet')
            for idx in range(self.num_images):
                img_path = str(Path(ponnet_path, df_scene[img_list_path[idx]].iloc[-1]))
                
                if img_list_path[idx] == "image_rgb":
                    rec_img = cv2.resize(cv2.imread(img_path).astype(np.float32), dsize=(16, 16))
                
                img = torch.from_numpy(cv2.resize(cv2.imread(img_path).astype(np.float32), dsize=(224, 224))).reshape(-1, 150528)
                img_list.append(img)

        return img_list, rec_img


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
        img_list, rec_img = self._load_ponnet_video_feature(raw_name)
        
        single_video_features = []
        single_video_meta = []
        
        # imageとtextを合わせた特徴やmask等の作成
        data, meta = self.clip_sentence_to_feature(
            example["clip_id"],
            example["sentence"],
            rec_img,
            img_list,
        )
        """
        # TODO : memo
        # single_video_features: video特徴量を含むdict
        ここリストにする意味ある？
        """
        
        single_video_features.append(data)
        single_video_meta.append(meta)

        return single_video_features, single_video_meta

    def clip_sentence_to_feature(
        self,
        name,
        sentence,
        gt_rec,
        img_list,
    ):
        """
        make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            video_feature: Either np.array of rgb+flow features or Dict[str, np.array] of COOT embeddings
            clip_idx: clip number in the video (needed to loat COOT features)
        """

        # 画像の特徴量 + textの特徴量を合わせた形状のfeatと画像用のmaskを作成
        feat, video_tokens, video_mask = self._load_indexed_video_feature(img_list)

        # text用のtokenとmaskを作成
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)
        
        # img_tokenとtext_tokenを連結
        input_tokens = video_tokens + text_tokens

        # token(単語含む) -> id列
        input_ids = [
            self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens
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

        data = dict(
            name=name,
            input_tokens=input_tokens,
            input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            video_feature=feat.astype(np.float32),
            gt_rec = gt_rec,
        )
        meta = dict(name=name, sentence=sentence)
        
        return data, meta


    def _load_indexed_video_feature(
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
        
        # img + text の特徴を作成 + imgの特徴量を格納
        feat = np.zeros((self.max_v_len + self.max_t_len, img_list[0].shape[1]))
        for idx in range(len(img_list)):
            feat[idx+1] = img_list[idx]

        return feat, video_tokens, mask

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
        Args:
            batch:
        Returns:
        """
        # recurrent collate function. original docstring:
        # HOW to batch clip-sentence pair? 1) directly copy the last
        # sentence, but do not count them in when
        # back-prop OR put all -1 to their text token label, treat

        # collect meta
        raw_batch_meta = [e[1] for e in batch]
        batch_meta = []
        for e in raw_batch_meta:
            cur_meta = dict(name=None, timestamp=[], gt_sentence=[])
            for d in e:
                cur_meta["clip_id"] = d["name"]
                cur_meta["gt_sentence"].append(d["sentence"])
            batch_meta.append(cur_meta)

        batch = [e[0] for e in batch]
        # Step1: pad each example to max_n_sen
        max_n_sen = max([len(e) for e in batch])
        raw_step_sizes = []

        padded_batch = []
        padding_clip_sen_data = copy.deepcopy(
            batch[0][0]
        )  # doesn"t matter which one is used
        padding_clip_sen_data["input_labels"][:] =\
            BilaDataset.IGNORE
        for ele in batch:
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
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def create_datasets_and_loaders(
    cfg: MartConfig,
    coot_feat_dir: str = MartPathConst.COOT_FEAT_DIR,
    annotations_dir: str = MartPathConst.ANNOTATIONS_DIR,
    video_feature_dir: str = MartPathConst.VIDEO_FEATURE_DIR,
    datatype: str = 'bila',
) -> Tuple[
    BilaDataset, BilaDataset, data.DataLoader, data.DataLoader
]:
    # create the dataset
    dset_name_train = cfg.dataset_train.name
    train_dataset = BilaDataset(
        dset_name_train,
        cfg.max_t_len,
        cfg.max_v_len,
        cfg.max_n_sen,
        mode="train",
        recurrent=cfg.recurrent,
        untied=cfg.untied or cfg.mtrans,
        video_feature_dir=video_feature_dir,
        annotations_dir=annotations_dir,
        preload=cfg.dataset_train.preload,
        datatype=datatype,
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
        recurrent=cfg.recurrent,
        untied=cfg.untied or cfg.mtrans,
        video_feature_dir=video_feature_dir,
        annotations_dir=annotations_dir,
        preload=cfg.dataset_val.preload,
        datatype=datatype,
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
        recurrent=cfg.recurrent,
        untied=cfg.untied or cfg.mtrans,
        video_feature_dir=video_feature_dir,
        annotations_dir=annotations_dir,
        preload=cfg.dataset_val.preload,
        datatype=datatype,
    )
    test_loader = data.DataLoader(
        test_dataset,
        collate_fn=val_dataset.collate_fn,
        batch_size=cfg.val.batch_size,
        shuffle=cfg.dataset_val.shuffle,
        num_workers=cfg.dataset_val.num_workers,
        pin_memory=cfg.dataset_val.pin_memory,
    )


    return train_dataset, val_dataset, train_loader, val_loader, test_dataset, test_loader
