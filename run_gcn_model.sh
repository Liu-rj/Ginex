# Ogbn-papers100M
python prepare_dataset.py --dataset ogbn-papers100M --path /nvme2n1/offgs_dataset/ogbn-papers100M-offgs
python create_neigh_cache.py --neigh-cache-size 5e9 --dataset ogbn-papers100M
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model GCN
# # acc
# sudo env PATH=$PATH python run_ginex_acc.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 25e9 --feature-cache-size 25e9 --sb-size 10000 --num-epochs 50 --verbose --model GCN --gpu 1


# MAG240M
python prepare_dataset.py --dataset mag240m --path /nvme2n1/offgs_dataset/mag240m-offgs
python create_neigh_cache.py --neigh-cache-size 10e9 --dataset mag240m
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model GCN
# # acc
# sudo env PATH=$PATH python run_ginex_acc.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 50e9 --feature-cache-size 50e9 --sb-size 10000 --num-epochs 50 --verbose --model GCN --gpu 1
