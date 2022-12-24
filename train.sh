# for train
for seed in `seq 40 42`
do
    python train.py -c config/bilas.yaml --datatype bilas -m 'ca_embedder=Base;random_seed='${seed} --wandb
done
# for test
# python show_caption.py -m base