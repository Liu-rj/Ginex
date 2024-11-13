import argparse
import time
import os
import glob
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
import threading
from queue import Queue
from sage import SAGE
from gat import GAT
from gcn import GCN
import csv

from lib.data import *
from lib.cache import *
from lib.utils import *
from lib.neighbor_sampler import GinexNeighborSampler


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--model', type=str, default='SAGE')
argparser.add_argument('--batch-size', type=int, default=1024)
argparser.add_argument('--num-workers', type=int, default=os.cpu_count()*2)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--dropout', type=float, default=0.2)
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--exp-name', type=str, default=None)
argparser.add_argument('--sizes', type=str, default='20,15,10')
argparser.add_argument('--sb-size', type=int, default='1000')
argparser.add_argument('--feature-cache-size', type=float, default=500000000)
argparser.add_argument('--trace-load-num-threads', type=int, default=4)
argparser.add_argument('--neigh-cache-size', type=float, default=45000000000)
argparser.add_argument('--ginex-num-threads', type=int, default=os.cpu_count()*8)
argparser.add_argument('--verbose', dest='verbose', default=False, action='store_true')
argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
args = argparser.parse_args()
print(args)
args.neigh_cache_size = int(args.neigh_cache_size)
args.feature_cache_size = int(args.feature_cache_size)

# Set args/environment variables/path
os.environ["GINEX_NUM_THREADS"] = str(args.ginex_num_threads)
dataset_path = os.path.join('./dataset', args.dataset + '-ginex')
split_idx_path = os.path.join(dataset_path, "split_idx.pth")

# Prepare dataset
if args.verbose:
    tqdm.write("Preparing dataset...")
if args.exp_name is None:
    # now = datetime.now()
    # args.exp_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    args.exp_name = f"{args.dataset}-{args.neigh_cache_size:g}-{args.feature_cache_size:g}-{args.sb_size}-{args.sizes}"
os.makedirs(os.path.join('./trace', args.exp_name), exist_ok=True)
sizes = [int(size) for size in args.sizes.split(",")]
dataset = GinexDataset(path=dataset_path, split_idx_path=split_idx_path)
num_nodes = dataset.num_nodes
num_features = dataset.num_features
features = dataset.features_path
num_classes = dataset.num_classes
mmapped_features = dataset.get_mmapped_features()
indptr, indices = dataset.get_adj_mat()
labels = dataset.get_labels()
label_offset = dataset.conf['label_offset']

if args.verbose:
    tqdm.write("Done!")

# Define model
device = torch.device('cuda:%d' % args.gpu)
torch.cuda.set_device(device)
if args.model == 'SAGE':
    model = SAGE(num_features, args.num_hiddens, num_classes, num_layers=len(sizes), dropout=args.dropout)
elif args.model == 'GAT':
    model = GAT(num_features, args.num_hiddens, num_classes, num_layers=len(sizes), dropout=args.dropout)
elif args.model == 'GCN':
    model = GCN(num_features, args.num_hiddens, num_classes, num_layers=len(sizes), dropout=args.dropout)
else:
    raise ValueError('Invalid model name')
model = model.to(device)


def changeset_compute(i, last, mode="train"):
    if mode == "train":
        node_idx = dataset.shuffled_train_idx
    elif mode == "valid":
        node_idx = dataset.val_idx
    elif mode == "test":
        node_idx = dataset.test_idx

    torch.cuda.synchronize()
    tic = time.time()
    effective_sb_size = (
        int(
            (node_idx.numel() % (args.sb_size * args.batch_size) + args.batch_size - 1)
            / args.batch_size
        )
        if last
        else args.sb_size
    )
    cache = FeatureCache(
        args.feature_cache_size,
        effective_sb_size,
        num_nodes,
        mmapped_features,
        num_features,
        args.exp_name,
        i - 1,
        args.verbose,
    )
    # Pass 1 and 2 are executed before starting sb sample.
    # We overlap only the pass 3 of changeset precomputation,
    # which is the most time consuming part, with sb sample.
    iterptr, iters, initial_cache_indices = cache.pass_1_and_2()
    torch.cuda.synchronize()
    cache_init = time.time() - tic

    # Only changset precomputation at the last superbatch in epoch
    torch.cuda.synchronize()
    tic = time.time()
    cache.pass_3(iterptr, iters, initial_cache_indices)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    changeset_precompute = time.time() - tic

    return cache, initial_cache_indices.cpu(), (cache_init, changeset_precompute)


def inspect(i, last, mode="train"):
    if mode == "train":
        node_idx = dataset.shuffled_train_idx
    elif mode == "valid":
        node_idx = dataset.val_idx
    elif mode == "test":
        node_idx = dataset.test_idx

    torch.cuda.empty_cache()

    # Load neighbor cache
    tic = time.time()
    neighbor_cache_path = (
        str(dataset_path) + "/nc" + "_size_" + str(args.neigh_cache_size) + ".dat"
    )
    neighbor_cache_conf_path = (
        str(dataset_path) + "/nc" + "_size_" + str(args.neigh_cache_size) + "_conf.json"
    )
    neighbor_cache_numel = json.load(open(neighbor_cache_conf_path, "r"))["shape"][0]
    neighbor_cachetable_path = (
        str(dataset_path) + "/nctbl" + "_size_" + str(args.neigh_cache_size) + ".dat"
    )
    neighbor_cachetable_conf_path = (
        str(dataset_path)
        + "/nctbl"
        + "_size_"
        + str(args.neigh_cache_size)
        + "_conf.json"
    )
    neighbor_cachetable_numel = json.load(open(neighbor_cachetable_conf_path, "r"))[
        "shape"
    ][0]
    neighbor_cache = load_int64(neighbor_cache_path, neighbor_cache_numel)
    neighbor_cachetable = load_int64(
        neighbor_cachetable_path, neighbor_cachetable_numel
    )
    load_cache = time.time() - tic

    start_idx = i * args.batch_size * args.sb_size
    end_idx = min((i + 1) * args.batch_size * args.sb_size, node_idx.numel())
    loader = GinexNeighborSampler(
        indptr,
        dataset.indices_path,
        args.exp_name,
        i,
        node_idx=node_idx[start_idx:end_idx],
        sizes=sizes,
        num_nodes=num_nodes,
        cache_data=neighbor_cache,
        address_table=neighbor_cachetable,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    inner_sampling_time = 0
    tic = time.time()
    for step, duration in enumerate(loader):
        inner_sampling_time += duration

    sampling = time.time() - tic

    tensor_free(neighbor_cache)
    tensor_free(neighbor_cachetable)

    return load_cache, sampling, inner_sampling_time


def switch(cache, initial_cache_indices):
    cache.fill_cache(initial_cache_indices)
    del initial_cache_indices
    return cache


def trace_load(i, sb):
    return (
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_ids_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_adjs_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_update_' + str(i) + '.pth'),
            )


def gather(gather_q, n_id, cache, batch_size):
    batch_inputs = gather_ginex(features, n_id, num_features, cache)
    batch_labels = labels[n_id[:batch_size]]
    gather_q.put((batch_inputs, batch_labels))


def execute(i, cache, pbar, total_loss, total_correct, last, mode="train"):
    decompose_recorder = [0] * 10

    if last:
        if mode == "train":
            num_iter = int(
                (
                    dataset.shuffled_train_idx.numel()
                    % (args.sb_size * args.batch_size)
                    + args.batch_size
                    - 1
                )
                / args.batch_size
            )
        elif mode == "valid":
            num_iter = int(
                (
                    dataset.val_idx.numel() % (args.sb_size * args.batch_size)
                    + args.batch_size
                    - 1
                )
                / args.batch_size
            )
        elif mode == "test":
            num_iter = int(
                (
                    dataset.test_idx.numel() % (args.sb_size * args.batch_size)
                    + args.batch_size
                    - 1
                )
                / args.batch_size
            )
    else:
        num_iter = args.sb_size

    for idx in range(num_iter):
        # tic = time.time()
        # with open("/proc/sys/vm/drop_caches", "w") as stream:
        #     stream.write("1\n")
        # drop_cache_time += time.time() - tic

        # Sample
        tic = time.time()
        q_value = trace_load(idx, i - 1)
        n_id, adjs, (in_indices, in_positions, out_indices) = q_value
        decompose_recorder[0] += time.time() - tic
        batch_size = adjs[-1].size[1]

        # Gather
        tic = time.time()
        batch_inputs, cache_miss, io_traffic = gather_ginex(
            features, n_id, num_features, cache
        )
        batch_labels = labels[n_id[:batch_size] - label_offset]
        decompose_recorder[1] += time.time() - tic
        decompose_recorder[7] += cache_miss
        decompose_recorder[8] += io_traffic

        # Cache
        tic = time.time()
        cache.update(batch_inputs, in_indices, in_positions, out_indices)
        decompose_recorder[2] += time.time() - tic

        # Transfer
        torch.cuda.synchronize()
        tic = time.time()
        adjs = [adj.to(device) for adj in adjs]
        torch.cuda.synchronize()
        decompose_recorder[3] += time.time() - tic
        tic = time.time()
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        torch.cuda.synchronize()
        decompose_recorder[4] += time.time() - tic
        decompose_recorder[9] += batch_inputs.shape[0]

        # Forward
        tic = time.time()
        out = model(batch_inputs_cuda, adjs)
        loss = F.nll_loss(out, batch_labels_cuda.long())
        # Backward
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        decompose_recorder[5] += time.time() - tic

        # Free
        tic = time.time()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        del n_id
        del in_indices
        del in_positions
        del out_indices
        del adjs
        tensor_free(batch_inputs)
        decompose_recorder[6] += time.time() - tic
        pbar.update(batch_size)

    return total_loss, total_correct, decompose_recorder


def train(epoch):
    model.train()

    dataset.make_new_shuffled_train_idx()
    num_iter = int(
        (dataset.shuffled_train_idx.numel() + args.batch_size - 1) / args.batch_size
    )

    pbar = tqdm(total=dataset.train_idx.numel(), position=0, leave=True, ncols=100)
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0
    num_sb = int(
        (dataset.train_idx.numel() + args.batch_size * args.sb_size - 1)
        / (args.batch_size * args.sb_size)
    )

    # sampling_time, train_time = 0, 0, 0
    info_recoder = [0] * 3
    sample_decompose = [0] * 5
    train_decompose = [0] * 10

    for i in range(num_sb + 1):
        if args.verbose:
            tqdm.write(
                "Running {}th superbatch of total {} superbatches".format(i, num_sb)
            )

        # # Superbatch sample
        # if i < num_sb and epoch == 0:
        #     if args.verbose:
        #         tqdm.write("Step 1: Superbatch Sample")
        #     tic = time.time()
        #     decompose_time = inspect(i, last=(i == num_sb), mode="train")
        #     torch.cuda.synchronize()
        #     info_recoder[0] += time.time() - tic
        #     for j in range(2):
        #         sample_decompose[j] += decompose_time[j]
        #     sample_decompose[-1] += decompose_time[-1]  # inner sample time
        #     if args.verbose:
        #         tqdm.write("Step 1: Done")

        if i == 0:
            continue

        node_idx = dataset.shuffled_train_idx
        effective_sb_size = int((node_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size) if i==num_sb else args.sb_size
        cache = FeatureCache(args.feature_cache_size, effective_sb_size, num_nodes, mmapped_features, num_features, args.exp_name, i - 1, args.verbose)
        # Pass 1 and 2 are executed before starting sb sample.
        # We overlap only the pass 3 of changeset precomputation, 
        # which is the most time consuming part, with sb sample.
        iterptr, iters, initial_cache_indices = cache.pass_1_and_2()

        # Switch
        if args.verbose:
            tqdm.write("Step 2: Switch")
        tic = time.time()
        cache = switch(cache, initial_cache_indices.cpu())
        torch.cuda.synchronize()
        info_recoder[1] += time.time() - tic
        if args.verbose:
            tqdm.write("Step 2: Done")

        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")

        # Main loop
        if args.verbose:
            tqdm.write("Step 3: Main Loop")
        tic = time.time()
        total_loss, total_correct, decompose_time = execute(
            i,
            cache,
            pbar,
            total_loss,
            total_correct,
            last=(i == num_sb),
            mode="train",
        )
        torch.cuda.synchronize()
        info_recoder[2] += time.time() - tic
        for j in range(len(decompose_time)):
            train_decompose[j] += decompose_time[j]
        if args.verbose:
            tqdm.write("Step 3: Done")

        # Delete obsolete runtime files
        # New: Do not delete trace file for reuse
        # delete_trace(i)

    pbar.close()

    loss = total_loss / num_iter
    approx_acc = total_correct / dataset.train_idx.numel()

    print(
        f"Load Cache time: {sample_decompose[0]:.3f} s\t"
        f"Sampling time: {sample_decompose[1]:.3f} s\t"
        f"Inner sampling time: {sample_decompose[4]:.3f} s\t"
        f"Cache Init time: {sample_decompose[2]:.3f} s\t"
        f"Changeset Precomputaion time: {sample_decompose[3]:.3f} s"
    )

    print(
        f"Load Graph time: {train_decompose[0]:.3f} s\t"
        f"Load feature time: {train_decompose[1]:.3f} s\t"
        f"Cache Update time: {train_decompose[2]:.3f} s\t"
        f"Graph Transfer time: {train_decompose[3]:.3f} s\t"
        f"Feature Transfer time: {train_decompose[4]:.3f} s\t"
        f"Train time: {train_decompose[5]:.3f} s\t"
        f"Free time: {train_decompose[6]:.3f} s\t"
        f"Cache Miss Num: {train_decompose[7]}\t"
        f"IO Traffic: {train_decompose[8]}\t"
        f"Feature Transfer Num: {train_decompose[9]}\t"
    )

    return loss, approx_acc, info_recoder, sample_decompose, train_decompose


if __name__ == "__main__":
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0

    epoch_info_recoder = [[] for i in range(4)]
    epoch_sample_decompose = [[] for i in range(5)]
    epoch_train_decompose = [[] for i in range(10)]

    for epoch in range(args.num_epochs):
        if args.verbose:
            tqdm.write("\n==============================")
            tqdm.write("Running Epoch {}...".format(epoch))

        loss, acc, info_recoder, sample_decompose, train_decompose = train(epoch)

        for i, record in enumerate(info_recoder):
            epoch_info_recoder[i].append(record)
        if epoch > 0:
            epoch_info_recoder[0][-1] = epoch_info_recoder[0][0]
        epoch_total_time = np.sum([record[-1] for record in epoch_info_recoder[:-1]])
        epoch_info_recoder[-1].append(epoch_total_time)
        for i, record in enumerate(sample_decompose):
            if record == 0 and epoch != 0:
                epoch_sample_decompose[i].append(epoch_sample_decompose[i][-1])
            else:
                epoch_sample_decompose[i].append(record)
        for i, record in enumerate(train_decompose):
            epoch_train_decompose[i].append(record)

    log_info = [
        args.dataset,
        args.sizes,
        args.batch_size,
        args.feature_cache_size,
        "SAGE",
        args.num_epochs,
    ]

    with open("logs/end_to_end_single_thread_decompose.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_end_to_end = log_info.copy()
        for epoch_info in epoch_info_recoder:
            log_end_to_end.append(round(np.mean(epoch_info), 2))
        writer.writerow(log_end_to_end)

    with open("logs/sample_single_thread_decompose.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_sample = log_info.copy()
        for epoch_info in epoch_sample_decompose:
            log_sample.append(round(np.mean(epoch_info), 2))
        writer.writerow(log_sample)

    with open("logs/train_single_thread_decompose.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_train = log_info.copy()
        for epoch_info in epoch_train_decompose:
            log_train.append(round(np.mean(epoch_info), 2))
        writer.writerow(log_train)
