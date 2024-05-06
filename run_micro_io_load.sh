# Friendster
# sudo env PATH=$PATH python run_ginex_single_thread.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 4e9 --feature-cache-size 4e9 --sb-size 10000 --num-epochs 1 --verbose --train-only
# sudo env PATH=$PATH python run_ginex_single_thread.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 10e9 --feature-cache-size 10e9 --sb-size 10000 --num-epochs 1 --verbose --train-only
# sudo env PATH=$PATH python run_ginex_single_thread.py --dataset friendster --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 10000 --num-epochs 1 --verbose --train-only


# IGB-HOM
# sudo env PATH=$PATH python run_ginex_single_thread.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 15e9 --feature-cache-size 15e9 --sb-size 20000 --num-epochs 1 --verbose --train-only
# sudo env PATH=$PATH python run_ginex_single_thread.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 45e9 --feature-cache-size 45e9 --sb-size 20000 --num-epochs 1 --verbose --train-only

sudo env PATH=$PATH python run_ginex.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 75e9 --feature-cache-size 75e9 --sb-size 20000 --num-epochs 1 --verbose --train-only --sample-mode
sudo env PATH=$PATH python run_ginex_single_thread.py --dataset igb-full --num-hiddens 256 --dropout 0 --neigh-cache-size 75e9 --feature-cache-size 75e9 --sb-size 20000 --num-epochs 1 --verbose --train-only
