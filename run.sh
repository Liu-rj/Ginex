# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages cgexec -g memory:8gb /opt/conda/envs/npc/bin/python run_baseline.py

# sudo PYTHONPATH=/opt/conda/envs/npc/lib/python3.9/site-packages cgexec -g memory:8gb /opt/conda/envs/npc/bin/python run_ginex.py --neigh-cache-size 6000000000 --feature-cache-size 6000000000 --sb-size 1500


# sudo env PATH=$PATH python run_baseline.py

# sudo env PATH=$PATH python run_ginex.py --neigh-cache-size 6000000000 --feature-cache-size 6000000000 --sb-size 1500

# sudo env PATH=$PATH python run_ginex.py --dataset=ogbn-products --neigh-cache-size 1000000000 --feature-cache-size 1000000000 --sb-size 1500



# python prepare_dataset.py --dataset=ogbn-products
# python create_neigh_cache.py --neigh-cache-size 200000000 --dataset=ogbn-products
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset=ogbn-products --neigh-cache-size 200000000 --feature-cache-size 200000000 --sb-size 1500

# python create_neigh_cache.py --neigh-cache-size 600000000 --dataset=ogbn-products
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset=ogbn-products --neigh-cache-size 600000000 --feature-cache-size 600000000 --sb-size 1500





# python prepare_dataset.py --dataset=ogbn-papers100M
# python create_neigh_cache.py --neigh-cache-size 10000000000 --dataset=ogbn-papers100M
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset=ogbn-papers100M --neigh-cache-size 10000000000 --feature-cache-size 10000000000 --sb-size 1500

# python create_neigh_cache.py --neigh-cache-size 32000000000 --dataset=ogbn-papers100M
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset=ogbn-papers100M --neigh-cache-size 32000000000 --feature-cache-size 32000000000 --sb-size 1500




# python prepare_dataset.py --dataset=friendster
# python create_neigh_cache.py --neigh-cache-size 6400000000 --dataset=friendster
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset=friendster --neigh-cache-size 6400000000 --feature-cache-size 6400000000 --sb-size 1500

# python create_neigh_cache.py --neigh-cache-size 19200000000 --dataset=friendster
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset=friendster --neigh-cache-size 19200000000 --feature-cache-size 19200000000 --sb-size 1500
