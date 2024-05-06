# Ogbn-papers100M
python create_neigh_cache.py --neigh-cache-size 5e9 --dataset ogbn-papers100M
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 3 --verbose --train-only

python create_neigh_cache.py --neigh-cache-size 15e9 --dataset ogbn-papers100M
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 10000 --num-epochs 3 --verbose --train-only

python create_neigh_cache.py --neigh-cache-size 25e9 --dataset ogbn-papers100M
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 25e9 --feature-cache-size 25e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 25e9 --feature-cache-size 25e9 --sb-size 10000 --num-epochs 3 --verbose --train-only


# MAG240M
python create_neigh_cache.py --neigh-cache-size 10e9 --dataset mag240m
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only

python create_neigh_cache.py --neigh-cache-size 30e9 --dataset mag240m
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 30e9 --feature-cache-size 30e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 30e9 --feature-cache-size 30e9 --sb-size 10000 --num-epochs 3 --verbose --train-only

python create_neigh_cache.py --neigh-cache-size 50e9 --dataset mag240m
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 50e9 --feature-cache-size 50e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 50e9 --feature-cache-size 50e9 --sb-size 10000 --num-epochs 3 --verbose --train-only


# Friendster
python create_neigh_cache.py --neigh-cache-size 4e9 --dataset friendster
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 4e9 --feature-cache-size 4e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 4e9 --feature-cache-size 4e9 --sb-size 10000 --num-epochs 3 --verbose --train-only

python create_neigh_cache.py --neigh-cache-size 10e9 --dataset friendster
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only

python create_neigh_cache.py --neigh-cache-size 15e9 --dataset friendster
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 10000 --num-epochs 3 --verbose --train-only


# IGB-HOM
python create_neigh_cache.py --neigh-cache-size 15e9 --dataset igb-full
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 20000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 20000 --num-epochs 3 --verbose --train-only

python create_neigh_cache.py --neigh-cache-size 45e9 --dataset igb-full
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 45e9 --feature-cache-size 45e9 --sb-size 20000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 45e9 --feature-cache-size 45e9 --sb-size 20000 --num-epochs 3 --verbose --train-only

python create_neigh_cache.py --neigh-cache-size 75e9 --dataset igb-full
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 75e9 --feature-cache-size 75e9 --sb-size 20000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 75e9 --feature-cache-size 75e9 --sb-size 20000 --num-epochs 3 --verbose --train-only
