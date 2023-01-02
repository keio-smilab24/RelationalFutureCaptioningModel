# for train
for seed in `seq 40 44`
do
    # python train.py -c config/bilas.yaml --datatype bilas -m 'train.num_epochs=25;ca_embedder=Base;random_seed='${seed} --wandb --del_weights
    python train.py -c config/bilas.yaml --datatype bilas -m 'train.num_epochs=25;ca_embedder=Base;random_seed='${seed} --wandb
done
# for test
# python show_caption.py -m base