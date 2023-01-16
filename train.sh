# for train
# for seed in `seq 40 44`
# do
#     # python train.py -c config/bilas.yaml --datatype bilas -m 'train.num_epochs=25;ca_embedder=Base;random_seed='${seed} --wandb --del_weights
#     python train.py -c config/bilas.yaml --datatype bilas -m 'train.num_epochs=30;ca_embedder=Base;random_seed='${seed} --wandb --remove_not_final
# done

# for seed in `seq 40 44`
# do
#     # show seed
#     echo "--------------------------------"
#     echo ${seed}
#     echo "--------------------------------"
    
#     # make dstore
#     # python train.py -c config/bilas.yaml --datatype bilas -m 'ca_embedder=Base;random_seed='${seed} --make_knn_dstore --load_model results/proposed_K/seed${seed}/models/model_29.pth
    
#     # do knn -- 3 parameter
#     python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.25;k_num=8;ca_embedder=Base;random_seed='${seed}';dstore_keys_path=checkpoints/dstore_keys_'${seed}'.npy;dstore_vals_path=checkpoints/dstore_vals_'${seed}'.npy' --load_model results/proposed_K/seed${seed}/models/model_29.pth --do_knn --test
#     python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.25;k_num=16;ca_embedder=Base;random_seed='${seed}';dstore_keys_path=checkpoints/dstore_keys_'${seed}'.npy;dstore_vals_path=checkpoints/dstore_vals_'${seed}'.npy' --load_model results/proposed_K/seed${seed}/models/model_29.pth --do_knn --test
#     python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.25;k_num=32;ca_embedder=Base;random_seed='${seed}';dstore_keys_path=checkpoints/dstore_keys_'${seed}'.npy;dstore_vals_path=checkpoints/dstore_vals_'${seed}'.npy' --load_model results/proposed_K/seed${seed}/models/model_29.pth --do_knn --test
#     python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.25;k_num=64;ca_embedder=Base;random_seed='${seed}';dstore_keys_path=checkpoints/dstore_keys_'${seed}'.npy;dstore_vals_path=checkpoints/dstore_vals_'${seed}'.npy' --load_model results/proposed_K/seed${seed}/models/model_29.pth --do_knn --test
#     python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.25;k_num=128;ca_embedder=Base;random_seed='${seed}';dstore_keys_path=checkpoints/dstore_keys_'${seed}'.npy;dstore_vals_path=checkpoints/dstore_vals_'${seed}'.npy' --load_model results/proposed_K/seed${seed}/models/model_29.pth --do_knn --test
#     # python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.20;k_num=512;ca_embedder=Base;random_seed='${seed} --load_model results/proposed_K/seed${seed}/models/model_29.pth --do_knn --test
#     # python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=100;alpha=0.25;k_num=512;ca_embedder=Base;random_seed='${seed} --load_model results/proposed_K/seed${seed}/models/model_29.pth --do_knn --test
# done

# temp  : 100, 250, 500, 750, 1000, 2000
# alpha : 0.1, 0.25, 0.45, 0.5, 0.65, 0.75
# num-k : 8, 16, 32, 64, 128, 256, 512

# test 
# bset 500, 0.25, 64 | 500, 0.20, 512 | 100, 0.25, 512
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.25;k_num=128' --load_model results/test_ema_w29/models/model_29.pth --test --do_knn
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.25;k_num=64' --load_model results/test_ema_w29/models/model_29.pth --test --do_knn
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.20;k_num=512' --load_model results/test_ema_w29/models/model_29.pth --test --do_knn
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=100;alpha=0.25;k_num=512' --load_model results/test_ema_w29/models/model_29.pth --test --do_knn
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=100;alpha=0.25;k_num=256' --load_model results/test_ema_w29/models/model_29.pth --test --do_knn

# print(torch.argmax(pred_scores[0]), torch.argmax(pure_scores[0]), torch.argmax(knn_preds[0]))
# print(torch.max(pred_scores[0]), torch.max(pure_scores[0]), torch.max(knn_preds[0]))
# print(torch.argmax(pred_scores[0]).item(), torch.max(pred_scores[0]).item(),  torch.argmax(pure_scores[0]).item(), torch.max(pure_scores[0]).item(),  torch.argmax(knn_preds[0]).item(), torch.max(knn_preds[0]).item())
# print('\033[31m', torch.argmax(pred_scores[0]).item(), torch.max(pred_scores[0]).item(),  torch.argmax(pure_scores[0]).item(), torch.max(pure_scores[0]).item(),  torch.argmax(knn_preds[0]).item(), torch.max(knn_preds[0]).item(), '\033[0m')

# make knn dstore
# python train.py -c config/bilas.yaml --datatype bilas --make_knn_dstore --load_model results/proposed_J/seed40/models/model_29.pth

# do knn
# python train.py -c config/bilas.yaml --datatype bilas --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test

# for seed in 40 41 43 44
# do
#     # python train.py -c config/bilas.yaml --datatype bilas -m 'train.num_epochs=25;ca_embedder=Base;random_seed='${seed} --wandb --del_weights
#     python train.py -c config/bilas.yaml --datatype bilas -m 'train.num_epochs=40;ca_embedder=Base;random_seed='${seed} --wandb
# done

# base score
# 14.65%, METEOR 20.13%, ROUGE_L 31.01%, CIDEr 57.87%, JaSPICE 16.91%

# knn test
# (1) knn num 10x 50x 100x 250x 400 500○ 600○ 750x 1000○

# 400: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 500: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 600: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13% 
# 10000: Bleu_4 14.06%, METEOR 19.91%, ROUGE_L 30.32%, CIDEr 55.68%, JaSPICE 14.07%

# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=1000;alpha=0.25;k_num=10' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=1000;alpha=0.25;k_num=50' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=1000;alpha=0.25;k_num=100' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=1000;alpha=0.25;k_num=250' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=1000;alpha=0.25;k_num=400' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=1000;alpha=0.25;k_num=600' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=1000;alpha=0.25;k_num=750' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=1000;alpha=0.25;k_num=1000' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test

# (2) temp 10x 50x 100x 250○ 500○ 600○ 700○ 800○ 900○ 1000○ 1500○ 2000 5000x 10000x

# 250: Bleu_4 14.53%, METEOR 20.40%, ROUGE_L 31.13%, CIDEr 58.51%, JaSPICE 15.97%
# 500: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 600: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 700 : Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 800: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 900: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 1000: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 1500: Bleu_4 14.53%, METEOR 20.40%, ROUGE_L 31.13%, CIDEr 58.51%, JaSPICE 15.97%,

# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=600;alpha=0.25;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=700;alpha=0.25;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=800;alpha=0.25;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=900;alpha=0.25;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=10000;alpha=0.25;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test

# (3) alpha 0.1x 0.15 0.20 0.25○ 0.5○ 0.75○

# 0.15: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 0.20: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 0.25 : Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 0.30: Bleu_4 14.57%, METEOR 20.17%, ROUGE_L 30.81%, CIDEr 57.81%, JaSPICE 16.13%
# 0.40: Bleu_4 14.53%, METEOR 20.40%, ROUGE_L 31.13%, CIDEr 58.51%, JaSPICE 15.97%
# 0.5 : Bleu_4 14.53%, METEOR 20.40%, ROUGE_L 31.13%, CIDEr 58.51%, JaSPICE 15.97%
# 0.75 : Bleu_4 13.60%, METEOR 20.19%, ROUGE_L 30.42%, CIDEr 55.09%, JaSPICE 14.39%,

# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.15;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.20;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.30;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test
# python train.py -c config/bilas.yaml --datatype bilas -m 'knn_temperature=500;alpha=0.40;k_num=500' --load_model results/proposed_J/seed40/models/model_29.pth --do_knn --test


# for test
# python show_caption.py -m base