# Nearest Neighbor Future Captioning: Generating Descriptions for Possible Collisions in Object Placement Tasks

[[paper](https://arxiv.org/abs/2407.13186)]

Takumi Komatsu, Motonari Kambara, Shumpei Hatanaka, Haruka Matsuo, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi, Komei Sugiura

# Abstract
Domestic service robots (DSRs) that support people in everyday environments have been widely investigated. However, their ability to predict and describe future risks resulting from their own actions remains insufficient. In this study, we focus on the linguistic explainability of DSRs. Most existing methods do not explicitly model the region of possible collisions; thus, they do not properly generate descriptions of these regions. In this paper, we propose the Nearest Neighbor Future Captioning Model that introduces the Nearest Neighbor Language Model for future captioning of possible collisions, which enhances the model output with a nearest neighbors retrieval mechanism. Furthermore, we introduce the Collision Attention Module that attends regions of possible collisions, which enables our model to generate descriptions that adequately reflect the objects associated with possible collisions. To validate our method, we constructed a new dataset containing samples of collisions that can occur when a DSR places an object in a simulation environment. The experimental results demonstrated that our method outperformed baseline methods, based on the standard metrics. In particular, on CIDEr-D, the baseline method obtained 25.09 points, whereas our method obtained 33.08 points.

## Setup
```bash
git clone https://github.com/keio-smilab24/RelationalFutureCaptioningModel.git  
cd RelationalFutureCaptioningModel
```

We assume the following environment for our experiments:
* Python 3.8.10
* Pytorch version 1.8.0 with CUDA 11.1 support


## Dataset
Download our dataset from [here](https://drive.google.com/drive/folders/1eDp7uy0nqmYPuSEYnPKgAxVzefpYTe5U).
We expect the directory structure to be the following:

```bash
./data
└── Bilas
　   ├── all_data_sorted.jsonl
　   ├── bilas_train_mecab.jsonl
　   ├── bilas_valid_mecab.jsonl
　   ├── bilas_test_mecab.jsonl
└── PonNet
　   ├── S-set3
　   　   ├── ponNetXX
　   　         ├── *.jpg # RGB, depth image
　   　   ├── sam_rgb
　   　         ├── *.jpg
　   ├── S-set4
　   　   ├── ponNetXX
　   　         ├── *.jpg # RGB, depth image
　   　   ├── sam_rgb
　   　         ├── *.jpg
```
We constructed the BILA-caption 2.0 dataset based on [BILA-caption dataset](https://arxiv.org/abs/2207.09083) and [PonNet] (https://arxiv.org/pdf/2102.06507).

## Train
```bash
python train.py -c config/bilas.yaml --datatype bilas -m 'max_v_len=8;train.num_epochs=30;ca_embedder=Base;random_seed=<seed>' --remove_not_final
```

## Make DataStore
```bash
python train.py -c config/bilas.yaml --datatype bilas -m 'ca_embedder=Base;random_seed=<seed>' --make_knn_dstore --load_model </path/to/your/checkpoint/path>
```

## Evaluation
```bash
python train.py -c config/bilas.yaml --datatype bilas -m 'ca_embedder=Base;random_seed=<seed>' --load_model <path/to/your/checkpoint/path> --test
```
