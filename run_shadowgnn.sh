# Ogbn-papers100M
# python prepare_dataset.py --dataset ogbn-papers100M --path /nvme2n1/offgs_dataset/ogbn-papers100M-offgs
# python create_neigh_cache.py --neigh-cache-size 5e9 --dataset ogbn-papers100M
sudo env PATH=$PATH python run_ginex_shadow.py --dataset ogbn-papers100M --num-hiddens 128 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode --sizes "10,10,10" --batch-size 1024
sudo env PATH=$PATH python run_ginex_shadow.py --dataset ogbn-papers100M --num-hiddens 128 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model SAGE --sizes "10,10,10" --batch-size 1024
# # acc
# sudo env PATH=$PATH python run_ginex_shadow.py --dataset ogbn-papers100M --num-hiddens 128 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 50 --verbose --model SAGE --sizes "10,10,10" --batch-size 1024

# MAG240M
# python prepare_dataset.py --dataset mag240m --path /nvme2n1/offgs_dataset/mag240m-offgs
# python create_neigh_cache.py --neigh-cache-size 10e9 --dataset mag240m
sudo env PATH=$PATH python run_ginex_shadow.py --dataset mag240m --num-hiddens 128 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode --sizes "10,10,10" --batch-size 1024
sudo env PATH=$PATH python run_ginex_shadow.py --dataset mag240m --num-hiddens 128 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model SAGE --sizes "10,10,10" --batch-size 1024
# # acc
# sudo env PATH=$PATH python run_ginex_shadow.py --dataset mag240m --num-hiddens 128 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 50 --verbose --model SAGE --sizes "10,10,10" --batch-size 1024
