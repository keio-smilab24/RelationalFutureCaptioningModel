# for train
for i in `seq 1 5`
do
    python train.py -c config/bilas.yaml --datatype bilas --del_weights --wandb
done
# for test
# python show_caption.py -m base