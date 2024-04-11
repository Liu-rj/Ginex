# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages cgexec -g memory:8gb /opt/conda/envs/npc/bin/python run_baseline.py

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages cgexec -g memory:8gb /opt/conda/envs/npc/bin/python run_ginex.py --neigh-cache-size 6000000000 --feature-cache-size 6000000000 --sb-size 1500


# sudo env PATH=$PATH python run_baseline.py

# sudo env PATH=$PATH python run_ginex.py --neigh-cache-size 6000000000 --feature-cache-size 6000000000 --sb-size 1500

# sudo env PATH=$PATH python run_ginex.py --dataset=ogbn-products --neigh-cache-size 1000000000 --feature-cache-size 1000000000 --sb-size 1500



# Ogbn-papers100M
python prepare_dataset.py --dataset ogbn-papers100M --path /nvme2n1/offgs_dataset/ogbn-papers100M-offgs
python create_neigh_cache.py --neigh-cache-size 5e9 --dataset ogbn-papers100M
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 256 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model SAGE
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 64 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model GAT


# MAG240M
python prepare_dataset.py --dataset mag240m --path /nvme1n1/offgs_dataset/mag240m-offgs
python create_neigh_cache.py --neigh-cache-size 10e9 --dataset mag240m
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 256 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model SAGE
sudo env PATH=$PATH python run_ginex.py --dataset mag240m --num-hiddens 64 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model GAT


# Friendster
python prepare_dataset.py --dataset friendster --path /nvme2n1/offgs_dataset/friendster-offgs
python create_neigh_cache.py --neigh-cache-size 4e9 --dataset friendster
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 4e9 --feature-cache-size 4e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 4e9 --feature-cache-size 4e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model SAGE
sudo env PATH=$PATH python run_ginex.py --dataset friendster --num-hiddens 64 --dropout 0 --neigh-cache-size 4e9 --feature-cache-size 4e9 --sb-size 10000 --num-epochs 3 --verbose --train-only --model GAT


# IGB-HOM
python prepare_dataset.py --dataset igb-full --path /nvme1n1/offgs_dataset/igb-full-offgs
python create_neigh_cache.py --neigh-cache-size 15e9 --dataset igb-full
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 20000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 20000 --num-epochs 3 --verbose --train-only --model SAGE
sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 64 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 20000 --num-epochs 3 --verbose --train-only --model GAT
