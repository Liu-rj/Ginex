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

from lib.data import *
from lib.cache import *
from lib.utils import *
from lib.neighbor_sampler import GinexNeighborSampler


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument("--num-epochs", type=int, default=3)
argparser.add_argument("--batch-size", type=int, default=1024)
argparser.add_argument("--num-workers", type=int, default=os.cpu_count() * 2)
argparser.add_argument("--num-hiddens", type=int, default=256)
argparser.add_argument("--dataset", type=str, default="ogbn-papers100M")
argparser.add_argument("--exp-name", type=str, default=None)
argparser.add_argument("--sizes", type=str, default="10,10,10")
argparser.add_argument("--sb-size", type=int, default="1000")
argparser.add_argument("--feature-cache-size", type=float, default=500000000)
argparser.add_argument("--trace-load-num-threads", type=int, default=4)
argparser.add_argument("--neigh-cache-size", type=int, default=45000000000)
argparser.add_argument("--ginex-num-threads", type=int, default=os.cpu_count() * 8)
argparser.add_argument("--verbose", dest="verbose", default=False, action="store_true")
argparser.add_argument(
    "--train-only", dest="train_only", default=False, action="store_true"
)
args = argparser.parse_args()
print(args)

# Set args/environment variables/path
os.environ["GINEX_NUM_THREADS"] = str(args.ginex_num_threads)
dataset_path = os.path.join("./dataset", args.dataset + "-ginex")
split_idx_path = os.path.join(dataset_path, "split_idx.pth")

# Prepare dataset
if args.verbose:
    tqdm.write("Preparing dataset...")
if args.exp_name is None:
    now = datetime.now()
    args.exp_name = now.strftime("%Y_%m_%d_%H_%M_%S")
os.makedirs(os.path.join("./trace", args.exp_name), exist_ok=True)
sizes = [int(size) for size in args.sizes.split(",")]
dataset = GinexDataset(path=dataset_path, split_idx_path=split_idx_path)
num_nodes = dataset.num_nodes
num_features = dataset.num_features
features = dataset.features_path
num_classes = dataset.num_classes
mmapped_features = dataset.get_mmapped_features()
indptr, indices = dataset.get_adj_mat()
labels = dataset.get_labels()

if args.verbose:
    tqdm.write("Done!")

# Define model
device = torch.device("cuda:%d" % args.gpu)
torch.cuda.set_device(device)
model = SAGE(num_features, args.num_hiddens, num_classes, num_layers=len(sizes))
model = model.to(device)


def inspect(i, last, mode="train"):
    drop_cache_time = 0
    changeset_precompute = 0
    cache_init = 0
    sampling = 0
    load_cache = 0
    inner_sampling_time = 0

    # Same effect of `sysctl -w vm.drop_caches=1`
    # Requires sudo
    tic = time.time()
    with open("/proc/sys/vm/drop_caches", "w") as stream:
        stream.write("1\n")
    drop_cache_time += time.time() - tic

    if mode == "train":
        node_idx = dataset.shuffled_train_idx
    elif mode == "valid":
        node_idx = dataset.val_idx
    elif mode == "test":
        node_idx = dataset.test_idx

    # No changeset precomputation when i == 0
    if i != 0:
        torch.cuda.synchronize()
        tic = time.time()
        effective_sb_size = (
            int(
                (
                    node_idx.numel() % (args.sb_size * args.batch_size)
                    + args.batch_size
                    - 1
                )
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
        cache_init += time.time() - tic

        # Only changset precomputation at the last superbatch in epoch
        if last:
            torch.cuda.synchronize()
            tic = time.time()
            cache.pass_3(iterptr, iters, initial_cache_indices)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            changeset_precompute += time.time() - tic

            print(
                f"Lode Cache time: {load_cache:.3f} s\t"
                f"sampling time: {sampling:.3f} s\t"
                f"Inner sampling time: {inner_sampling_time:.3f} s\t"
                f"Cache Init time: {cache_init:.3f} s\t"
                f"Changeset Precomputaion time: {changeset_precompute:.3f} s"
            )

            return cache, initial_cache_indices.cpu(), drop_cache_time
        else:
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
        num_workers=4,
    )

    tic = time.time()
    for step, duration in enumerate(loader):
        sampling += time.time() - tic
        inner_sampling_time += duration

        # tic = time.time()
        # with open("/proc/sys/vm/drop_caches", "w") as stream:
        #     stream.write("1\n")
        # drop_cache_time += time.time() - tic

        if i != 0 and step == 0:
            torch.cuda.synchronize()
            tic = time.time()
            cache.pass_3(iterptr, iters, initial_cache_indices)
            torch.cuda.synchronize()
            changeset_precompute += time.time() - tic

        tic = time.time()

    tensor_free(neighbor_cache)
    tensor_free(neighbor_cachetable)

    print(
        f"Load Cache time: {load_cache:.3f} s\t"
        f"Sampling time: {sampling:.3f} s\t"
        f"Inner sampling time: {inner_sampling_time:.3f} s\t"
        f"Cache Init time: {cache_init:.3f} s\t"
        f"Changeset Precomputaion time: {changeset_precompute:.3f} s"
    )

    if i != 0:
        return cache, initial_cache_indices.cpu(), drop_cache_time
    else:
        return None, None, drop_cache_time


def switch(cache, initial_cache_indices):
    cache.fill_cache(initial_cache_indices)
    del initial_cache_indices
    return cache


def trace_load(q, indices, sb):
    for i in indices:
        q.put(
            (
                torch.load(
                    "./trace/"
                    + args.exp_name
                    + "/"
                    + "sb_"
                    + str(sb)
                    + "_ids_"
                    + str(i)
                    + ".pth"
                ),
                torch.load(
                    "./trace/"
                    + args.exp_name
                    + "/"
                    + "sb_"
                    + str(sb)
                    + "_adjs_"
                    + str(i)
                    + ".pth"
                ),
                torch.load(
                    "./trace/"
                    + args.exp_name
                    + "/"
                    + "sb_"
                    + str(sb)
                    + "_update_"
                    + str(i)
                    + ".pth"
                ),
            )
        )


def gather(gather_q, n_id, cache, batch_size):
    batch_inputs = gather_ginex(features, n_id, num_features, cache)
    batch_labels = labels[n_id[:batch_size]]
    gather_q.put((batch_inputs, batch_labels))


def delete_trace(i):
    n_id_filelist = glob.glob(
        "./trace/" + args.exp_name + "/sb_" + str(i - 1) + "_ids_*"
    )
    adjs_filelist = glob.glob(
        "./trace/" + args.exp_name + "/sb_" + str(i - 1) + "_adjs_*"
    )
    cache_filelist = glob.glob(
        "./trace/" + args.exp_name + "/sb_" + str(i - 1) + "_update_*"
    )

    for n_id_file in n_id_filelist:
        try:
            os.remove(n_id_file)
        except:
            tqdm.write("Error while deleting file : ", n_id_file)

    for adjs_file in adjs_filelist:
        try:
            os.remove(adjs_file)
        except:
            tqdm.write("Error while deleting file : ", adjs_file)

    for cache_file in cache_filelist:
        try:
            os.remove(cache_file)
        except:
            tqdm.write("Error while deleting file : ", cache_file)


def execute(i, cache, pbar, total_loss, total_correct, last, mode="train"):
    drop_cache_time = 0
    feat_load_time = 0
    graph_load_time = 0
    cache_update_time = 0
    train_time = 0
    free_time = 0
    graph_transfer_time, feat_transfer_time = 0, 0
    input_feat_size = 0
    input_node_num = 0

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

    # Multi-threaded load of sets of (ids, adj, update)
    q = list()
    loader = list()
    for t in range(args.trace_load_num_threads):
        q.append(Queue(maxsize=2))
        loader.append(
            threading.Thread(
                target=trace_load,
                args=(
                    q[t],
                    list(range(t, num_iter, args.trace_load_num_threads)),
                    i - 1,
                ),
                daemon=True,
            )
        )
        loader[t].start()

    n_id_q = Queue(maxsize=2)
    adjs_q = Queue(maxsize=2)
    in_indices_q = Queue(maxsize=2)
    in_positions_q = Queue(maxsize=2)
    out_indices_q = Queue(maxsize=2)
    gather_q = Queue(maxsize=1)

    for idx in range(num_iter):
        tic = time.time()
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")
        drop_cache_time += time.time() - tic

        batch_size = args.batch_size
        if idx == 0:
            # Sample
            tic = time.time()
            q_value = q[idx % args.trace_load_num_threads].get()
            graph_load_time += time.time() - tic
            if q_value:
                n_id, adjs, (in_indices, in_positions, out_indices) = q_value
                batch_size = adjs[-1].size[1]
                n_id_q.put(n_id)
                adjs_q.put(adjs)
                in_indices_q.put(in_indices)

                in_positions_q.put(in_positions)
                out_indices_q.put(out_indices)

            # Gather
            tic = time.time()
            batch_inputs = gather_ginex(features, n_id, num_features, cache)
            batch_labels = labels[n_id[:batch_size]]
            feat_load_time += time.time() - tic

            # Cache
            tic = time.time()
            cache.update(batch_inputs, in_indices, in_positions, out_indices)
            cache_update_time += time.time() - tic

        if idx != 0:
            # Gather
            tic = time.time()
            (batch_inputs, batch_labels) = gather_q.get()
            feat_load_time += time.time() - tic

            # Cache
            in_indices = in_indices_q.get()
            in_positions = in_positions_q.get()
            out_indices = out_indices_q.get()

            tic = time.time()
            cache.update(batch_inputs, in_indices, in_positions, out_indices)
            cache_update_time += time.time() - tic

        if idx != num_iter - 1:
            # Sample
            tic = time.time()
            q_value = q[(idx + 1) % args.trace_load_num_threads].get()
            graph_load_time += time.time() - tic
            if q_value:
                n_id, adjs, (in_indices, in_positions, out_indices) = q_value
                batch_size = adjs[-1].size[1]
                n_id_q.put(n_id)
                adjs_q.put(adjs)
                in_indices_q.put(in_indices)
                in_positions_q.put(in_positions)
                out_indices_q.put(out_indices)

            # Gather
            gather_loader = threading.Thread(
                target=gather, args=(gather_q, n_id, cache, batch_size), daemon=True
            )
            gather_loader.start()

        # Transfer
        torch.cuda.synchronize()
        tic = time.time()
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        torch.cuda.synchronize()
        feat_transfer_time += time.time() - tic
        tic = time.time()
        adjs_host = adjs_q.get()
        adjs = [adj.to(device) for adj in adjs_host]
        torch.cuda.synchronize()
        graph_transfer_time += time.time() - tic
        input_node_num += batch_inputs.shape[0]
        input_feat_size += batch_inputs.shape[0] * batch_inputs.shape[1]

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
        train_time += time.time() - tic

        # Free
        tic = time.time()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        n_id = n_id_q.get()
        del n_id
        if idx == 0:
            in_indices = in_indices_q.get()
            in_positions = in_positions_q.get()
            out_indices = out_indices_q.get()
        del in_indices
        del in_positions
        del out_indices
        del adjs_host
        tensor_free(batch_inputs)
        free_time += time.time() - tic
        pbar.update(batch_size)

    print(
        f"Load Graph time: {graph_load_time:.3f} s\t"
        f"Load feature time: {feat_load_time:.3f} s\t"
        f"Cache Update time: {cache_update_time:.3f} s\t"
        f"Graph Transfer time: {graph_transfer_time:.3f} s\t"
        f"Feature Transfer time: {feat_transfer_time:.3f} s\t"
        f"Train time: {train_time:.3f} s\t"
        f"Free time: {free_time:.3f} s"
    )

    print(
        f"Feature Transfer size: {input_feat_size}\t"
        f"Input Node num: {input_node_num}"
    )

    return total_loss, total_correct, drop_cache_time


def train(epoch):
    model.train()

    dataset.make_new_shuffled_train_idx()
    num_iter = int(
        (dataset.shuffled_train_idx.numel() + args.batch_size - 1) / args.batch_size
    )

    pbar = tqdm(total=dataset.train_idx.numel(), position=0, leave=True)
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0
    num_sb = int(
        (dataset.train_idx.numel() + args.batch_size * args.sb_size - 1)
        / (args.batch_size * args.sb_size)
    )

    sampling_time, train_time, drop_cache_time = 0, 0, 0

    for i in range(num_sb + 1):
        tic = time.time()
        with open("/proc/sys/vm/drop_caches", "w") as stream:
            stream.write("1\n")
        drop_cache_time += time.time() - tic

        if args.verbose:
            tqdm.write(
                "Running {}th superbatch of total {} superbatches".format(i, num_sb)
            )

        # Superbatch sample
        if args.verbose:
            tqdm.write("Step 1: Superbatch Sample")
        tic = time.time()
        cache, initial_cache_indices, duration = inspect(
            i, last=(i == num_sb), mode="train"
        )
        torch.cuda.synchronize()
        sampling_time += time.time() - tic - duration
        drop_cache_time += duration
        if args.verbose:
            tqdm.write("Step 1: Done")

        if i == 0:
            continue

        # Switch
        if args.verbose:
            tqdm.write("Step 2: Switch")
        cache = switch(cache, initial_cache_indices)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write("Step 2: Done")

        # Main loop
        if args.verbose:
            tqdm.write("Step 3: Main Loop")
        tic = time.time()
        total_loss, total_correct, duration = execute(
            i, cache, pbar, total_loss, total_correct, last=(i == num_sb), mode="train"
        )
        torch.cuda.synchronize()
        train_time += time.time() - tic - duration
        drop_cache_time += duration
        if args.verbose:
            tqdm.write("Step 3: Done")

        # Delete obsolete runtime files
        delete_trace(i)

    pbar.close()

    loss = total_loss / num_iter
    approx_acc = total_correct / dataset.train_idx.numel()

    return loss, approx_acc, sampling_time, train_time, drop_cache_time


# @torch.no_grad()
# def inference(mode='test'):
#     model.eval()

#     if mode == 'test':
#         pbar = tqdm(total=dataset.test_idx.numel(), position=0, leave=True)
#         num_sb = int((dataset.test_idx.numel()+args.batch_size*args.sb_size-1)/(args.batch_size*args.sb_size))
#         num_iter = int((dataset.test_idx.numel()+args.batch_size-1) / args.batch_size)
#     elif mode == 'valid':
#         pbar = tqdm(total=dataset.val_idx.numel(), position=0, leave=True)
#         num_sb = int((dataset.val_idx.numel()+args.batch_size*args.sb_size-1)/(args.batch_size*args.sb_size))
#         num_iter = int((dataset.val_idx.numel()+args.batch_size-1) / args.batch_size)

#     pbar.set_description('Evaluating')

#     total_loss = total_correct = 0

#     for i in range(num_sb + 1):
#         if args.verbose:

#             tqdm.write ('Running {}th superbatch of total {} superbatches'.format(i, num_sb))

#         # Superbatch sample
#         if args.verbose:
#             tqdm.write ('Step 1: Superbatch Sample')
#         cache, initial_cache_indices = inspect(i, last=(i==num_sb), mode=mode)
#         torch.cuda.synchronize()
#         if args.verbose:
#             tqdm.write ('Step 1: Done')

#         if i == 0:
#             continue

#         # Switch
#         if args.verbose:
#             tqdm.write ('Step 2: Switch')
#         cache = switch(cache, initial_cache_indices)
#         torch.cuda.synchronize()
#         if args.verbose:
#             tqdm.write ('Step 2: Done')

#         # Main loop
#         if args.verbose:
#             tqdm.write ('Step 3: Main Loop')
#         total_loss, total_correct = execute(i, cache, pbar, total_loss, total_correct, last=(i==num_sb), mode=mode)
#         if args.verbose:
#             tqdm.write ('Step 3: Done')

#         # Delete obsolete runtime files
#         delete_trace(i)

#     pbar.close()

#     loss = total_loss / num_iter
#     if mode == 'test':
#         approx_acc = total_correct / dataset.test_idx.numel()
#     elif mode == 'valid':
#         approx_acc = total_correct / dataset.val_idx.numel()

#     return loss, approx_acc


if __name__ == "__main__":
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0
    epoch_time_list, sampling_list, train_list, cache_list = [], [], [], []
    for epoch in range(args.num_epochs):
        if args.verbose:
            tqdm.write("\n==============================")
            tqdm.write("Running Epoch {}...".format(epoch))

        torch.cuda.synchronize()
        start = time.time()
        loss, acc, sampling_time, train_time, drop_cache_time = train(epoch)
        torch.cuda.synchronize()
        epoch_time = time.time() - start - drop_cache_time
        epoch_time_list.append(epoch_time)
        sampling_list.append(sampling_time)
        train_list.append(train_time)
        tqdm.write(
            f"Sampling time: {sampling_time:.3f} s\t"
            f"Train time: {train_time:.3f} s\t"
            f"Drop Cache time: {drop_cache_time:.3f} s\t"
            f"Epoch time: {epoch_time:.3f} s"
        )
        tqdm.write(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}")

        # if epoch > 3 and not args.train_only:
        #     val_loss, val_acc = inference(mode='valid')
        #     test_loss, test_acc = inference(mode='test')
        #     tqdm.write ('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(val_loss, val_acc, test_loss, test_acc))

        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc
        #         final_test_acc = test_acc

    print(f"Avg Sampling Time: {np.mean(sampling_list[1:]):.3f}")
    print(f"Avg Train Time: {np.mean(train_list[1:]):.3f}")
    print(f"Avg Epoch Time: {np.mean(epoch_time_list[1:]):.3f}")
    # if not args.train_only:
    #     tqdm.write(f'Final Test acc: {final_test_acc:.4f}')
