# for train
for i in `seq 1`
do
    python train.py -c config/bilas_v6.yaml --datatype bilas --del_weights
    python train.py -c config/bilas.yaml --datatype bilas --del_weights
done
# for test
# python show_caption.py -m base