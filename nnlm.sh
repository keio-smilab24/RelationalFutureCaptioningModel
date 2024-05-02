#!/bin/bash

for seed in 43
do
python train.py -c config/bilas.yaml --datatype bilas -m 'ca_embedder=ConvNeXt;random_seed='${seed} --make_knn_dstore --load_model "results/run2024-05-02 11:28:45.938096/models/model_0.pth"
python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.25;k_num=256;ca_embedder=ConvNeXt;dstore_keys_path=checkpoints/dstore_keys.npy;dstore_vals_path=checkpoints/dstore_vals.npy;random_seed='${seed} --load_model "results/run2024-05-02 11:28:45.938096/models/model_0.pth" --do_knn --test --wandb
done
