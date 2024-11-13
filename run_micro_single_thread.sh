# Ogbn-papers100M
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model SAGE


# MAG240M
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model SAGE


# Friendster
# sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 4e9 --feature-cache-size 4e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model SAGE


# IGB-HOM
# sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 20000 --num-epochs 3 --verbose --train-only --model SAGE
