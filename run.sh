# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages cgexec -g memory:8gb /opt/conda/envs/npc/bin/python run_baseline.py

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages cgexec -g memory:8gb /opt/conda/envs/npc/bin/python run_ginex.py --neigh-cache-size 6000000000 --feature-cache-size 6000000000 --sb-size 1500


# sudo env PATH=$PATH python run_baseline.py

# sudo env PATH=$PATH python run_ginex.py --neigh-cache-size 6000000000 --feature-cache-size 6000000000 --sb-size 1500

# sudo env PATH=$PATH python run_ginex.py --dataset=ogbn-products --neigh-cache-size 1000000000 --feature-cache-size 1000000000 --sb-size 1500




# python prepare_dataset.py --dataset friendster
# python create_neigh_cache.py --neigh-cache-size 10e9 --dataset friendster
# sudo env PATH=$PATH python run_ginex.py --dataset friendster --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
# sudo env PATH=$PATH python run_ginex.py --dataset friendster --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only

# python create_neigh_cache.py --neigh-cache-size 6e9 --dataset friendster
# sudo env PATH=$PATH python run_ginex.py --dataset friendster --neigh-cache-size 6e9 --feature-cache-size 6e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
# sudo env PATH=$PATH python run_ginex.py --dataset friendster --neigh-cache-size 6e9 --feature-cache-size 6e9 --sb-size 10000 --num-epochs 3 --verbose --train-only


# python prepare_dataset.py --dataset mag240m
# python create_neigh_cache.py --neigh-cache-size 20e9 --dataset mag240m
# sudo env PATH=$PATH python run_ginex.py --dataset mag240m --neigh-cache-size 20e9 --feature-cache-size 20e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
# sudo env PATH=$PATH python run_ginex.py --dataset mag240m --neigh-cache-size 20e9 --feature-cache-size 20e9 --sb-size 10000 --num-epochs 3 --verbose --train-only


# python prepare_dataset.py --dataset ogbn-papers100M
# python create_neigh_cache.py --neigh-cache-size 10e9 --dataset ogbn-papers100M
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 512 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 512 --dropout 0.2 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 3 --verbose --train-only

# python create_neigh_cache.py --neigh-cache-size 5e9 --dataset ogbn-papers100M
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 512 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex.py --dataset ogbn-papers100M --num-hiddens 512 --dropout 0.2 --neigh-cache-size 5e9 --feature-cache-size 5e9 --sb-size 10000 --num-epochs 3 --verbose --train-only
