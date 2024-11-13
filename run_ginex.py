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
argparser.add_argument('--sample-mode', dest='sample_mode', default=False, action='store_true')
argparser.add_argument("--log", type=str, default="logs/train_multi_thread.csv")
args = argparser.parse_args()
print(args)
args.neigh_cache_size = int(args.neigh_cache_size)
args.feature_cache_size = int(args.feature_cache_size)

# Set args/environment variables/path
os.environ['GINEX_NUM_THREADS'] = str(args.ginex_num_threads)
dataset_path = os.path.join('./dataset', args.dataset + '-ginex')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

# Prepare dataset
if args.verbose:
    tqdm.write('Preparing dataset...')
if args.exp_name is None:
    # now = datetime.now()
    # args.exp_name = now.strftime('%Y_%m_%d_%H_%M_%S')
    args.exp_name = f"{args.dataset}-{args.neigh_cache_size:g}-{args.feature_cache_size:g}-{args.sb_size}-{args.sizes}"
os.makedirs(os.path.join('./trace', args.exp_name), exist_ok=True)
sizes = [int(size) for size in args.sizes.split(',')]
dataset = GinexDataset(path=dataset_path, split_idx_path=split_idx_path)
num_nodes = dataset.num_nodes
num_features = dataset.num_features
features = dataset.features_path
num_classes = dataset.num_classes
mmapped_features = dataset.get_mmapped_features()
indptr, indices = dataset.get_adj_mat()
labels = dataset.get_labels()
label_offset = dataset.conf['label_offset']
print(f'Feature Dimension: {num_features}')

if args.verbose:
    tqdm.write('Done!')

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


def inspect(i, last, mode='train'):
    # Same effect of `sysctl -w vm.drop_caches=1`
    # Requires sudo
    with open('/proc/sys/vm/drop_caches', 'w') as stream:
        stream.write('1\n')

    if mode == 'train':
        node_idx = dataset.shuffled_train_idx
    elif mode == 'valid':
        node_idx = dataset.val_idx
    elif mode == 'test':
        node_idx = dataset.test_idx

    load_cache, sample, changeset_compute = 0, 0, 0

    # No changeset precomputation when i == 0
    if i != 0:
        torch.cuda.synchronize()
        tic = time.time()
        effective_sb_size = int((node_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size) if last else args.sb_size
        cache = FeatureCache(args.feature_cache_size, effective_sb_size, num_nodes, mmapped_features, num_features, args.exp_name, i - 1, args.verbose)
        # Pass 1 and 2 are executed before starting sb sample.
        # We overlap only the pass 3 of changeset precomputation, 
        # which is the most time consuming part, with sb sample.
        iterptr, iters, initial_cache_indices = cache.pass_1_and_2()
        
        # Only changset precomputation at the last superbatch in epoch
        if last:
            cache.pass_3(iterptr, iters, initial_cache_indices)
            torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        changeset_compute = time.time() - tic
        
        if last:
            return cache, initial_cache_indices.cpu(), (load_cache, sample, changeset_compute)

    # Load neighbor cache
    torch.cuda.synchronize()
    tic = time.time()
    neighbor_cache_path = str(dataset_path) + '/nc' + '_size_' + f"{args.neigh_cache_size:g}" + '.dat'
    neighbor_cache_conf_path = str(dataset_path) + '/nc' + '_size_' + f"{args.neigh_cache_size:g}" + '_conf.json'
    neighbor_cache_numel = json.load(open(neighbor_cache_conf_path, 'r'))['shape'][0]
    neighbor_cachetable_path = str(dataset_path) + '/nctbl' + '_size_' + f"{args.neigh_cache_size:g}" + '.dat'
    neighbor_cachetable_conf_path = str(dataset_path) + '/nctbl' + '_size_' + f"{args.neigh_cache_size:g}" + '_conf.json'
    neighbor_cachetable_numel = json.load(open(neighbor_cachetable_conf_path, 'r'))['shape'][0]
    neighbor_cache = load_int64(neighbor_cache_path, neighbor_cache_numel)
    neighbor_cachetable = load_int64(neighbor_cachetable_path, neighbor_cachetable_numel)
    torch.cuda.synchronize()
    load_cache = time.time() - tic

    changeset_compute_in_sample = 0
    torch.cuda.synchronize()
    tic = time.time()
    start_idx = i * args.batch_size * args.sb_size 
    end_idx = min((i+1) * args.batch_size * args.sb_size, node_idx.numel())
    loader = GinexNeighborSampler(indptr, dataset.indices_path, args.exp_name, i, node_idx=node_idx[start_idx:end_idx],
                                       sizes=sizes, num_nodes = num_nodes,
                                       cache_data = neighbor_cache, address_table = neighbor_cachetable,
                                       batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers, prefetch_factor=1<<20)

    num_mini_batches = (node_idx[start_idx:end_idx].numel() + args.batch_size - 1) // args.batch_size
    for step, _ in tqdm(enumerate(loader), total=num_mini_batches, ncols=100):
        if i != 0 and step == 0:
            torch.cuda.synchronize()
            tic = time.time()
            cache.pass_3(iterptr, iters, initial_cache_indices)
            torch.cuda.synchronize()
            changeset_compute_in_sample = time.time() - tic

    tensor_free(neighbor_cache)
    tensor_free(neighbor_cachetable)

    torch.cuda.synchronize()
    sample = time.time() - tic - changeset_compute_in_sample
    changeset_compute += changeset_compute_in_sample

    if i != 0:
        return cache, initial_cache_indices.cpu(), (load_cache, sample, changeset_compute)
    else:
        return None, None, (load_cache, sample, changeset_compute)


def switch(cache, initial_cache_indices):
    cache.fill_cache(initial_cache_indices); del(initial_cache_indices)
    return cache


def trace_load(q, indices, sb):
    for i in indices:
        q.put((
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_ids_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_adjs_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_update_' + str(i) + '.pth'),
            ))


def gather(gather_q, n_id, cache, batch_size):
    batch_inputs, cache_miss, io_traffic = gather_ginex(features, n_id, num_features, cache)
    batch_labels = labels[n_id[:batch_size] - label_offset]
    gather_q.put((batch_inputs, batch_labels, cache_miss, io_traffic))


def delete_trace(i):
    n_id_filelist = glob.glob('./trace/' + args.exp_name + '/sb_' + str(i - 1) + '_ids_*')
    adjs_filelist = glob.glob('./trace/' + args.exp_name + '/sb_' + str(i - 1) + '_adjs_*')
    cache_filelist = glob.glob('./trace/' + args.exp_name + '/sb_' + str(i - 1) + '_update_*')

    for n_id_file in n_id_filelist:
        try:
            os.remove(n_id_file)
        except:
            tqdm.write('Error while deleting file : ', n_id_file)

    for adjs_file in adjs_filelist:
        try:
            os.remove(adjs_file)
        except:
            tqdm.write('Error while deleting file : ', adjs_file)

    for cache_file in cache_filelist:
        try:
            os.remove(cache_file)
        except:
            tqdm.write('Error while deleting file : ', cache_file)


def execute(i, cache, pbar, total_loss, total_correct, last, mode='train'):
    if last:
        if mode == 'train':
            num_iter = int((dataset.shuffled_train_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
        elif mode == 'valid':
            num_iter = int((dataset.val_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
        elif mode == 'test':
            num_iter = int((dataset.test_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
    else:
        num_iter = args.sb_size

    # Multi-threaded load of sets of (ids, adj, update)
    q = list()
    loader = list()
    for t in range(args.trace_load_num_threads):
        q.append(Queue(maxsize=2))
        loader.append(threading.Thread(target=trace_load, args=(q[t], list(range(t, num_iter, args.trace_load_num_threads)), i-1), daemon=True))
        loader[t].start()

    n_id_q = Queue(maxsize=2)
    adjs_q = Queue(maxsize=2)
    in_indices_q = Queue(maxsize=2)
    in_positions_q = Queue(maxsize=2)
    out_indices_q = Queue(maxsize=2)
    gather_q = Queue(maxsize=1)
    
    total_num_infeats = 0
    total_cache_miss, total_io, load_time = 0, 0, 0

    for idx in range(num_iter):
        batch_size = args.batch_size
        if idx == 0:
            # Sample
            q_value = q[idx % args.trace_load_num_threads].get()
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
            batch_inputs, cache_miss, io_traffic = gather_ginex(features, n_id, num_features, cache)
            batch_labels = labels[n_id[:batch_size] - label_offset]
            load_time += time.time() - tic
            total_cache_miss += cache_miss
            total_io += io_traffic

            # Cache
            cache.update(batch_inputs, in_indices, in_positions, out_indices)

        if idx != 0:
            # Gather
            tic = time.time()
            (batch_inputs, batch_labels, cache_miss, io_traffic) = gather_q.get()
            load_time += time.time() - tic
            total_cache_miss += cache_miss
            total_io += io_traffic

            # Cache
            in_indices = in_indices_q.get()
            in_positions = in_positions_q.get()
            out_indices = out_indices_q.get()
            cache.update(batch_inputs, in_indices, in_positions, out_indices)

        if idx != num_iter-1:
            # Sample
            q_value = q[(idx + 1) % args.trace_load_num_threads].get()
            if q_value:
                n_id, adjs, (in_indices, in_positions, out_indices) = q_value
                batch_size = adjs[-1].size[1]
                n_id_q.put(n_id)
                adjs_q.put(adjs)
                in_indices_q.put(in_indices)
                in_positions_q.put(in_positions)
                out_indices_q.put(out_indices)

            # Gather
            gather_loader = threading.Thread(target=gather, args=(gather_q, n_id, cache, batch_size), daemon=True)
            gather_loader.start()

        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs_host = adjs_q.get()
        adjs = [adj.to(device) for adj in adjs_host]
        total_num_infeats += batch_inputs.numel()

        # Forward
        out = model(batch_inputs_cuda, adjs)
        loss = F.nll_loss(out, batch_labels_cuda.long())

        # Backward
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Free
        total_loss += float(loss.item())
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum().item())
        n_id = n_id_q.get()
        del(n_id)
        if idx == 0:
            in_indices = in_indices_q.get()
            in_positions = in_positions_q.get()
            out_indices = out_indices_q.get()
        del(in_indices)
        del(in_positions)
        del(out_indices)
        del(adjs_host)
        tensor_free(batch_inputs)
        pbar.update(batch_size)

    return total_loss, total_correct, total_num_infeats, (load_time, total_cache_miss, total_io)


def train(epoch):
    model.train()

    dataset.make_new_shuffled_train_idx()
    num_iter = int((dataset.shuffled_train_idx.numel()+args.batch_size-1) / args.batch_size)

    pbar = tqdm(total=dataset.train_idx.numel(), position=0, leave=True, ncols=100)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_num_infeats = 0
    total_loss = total_correct = 0
    num_sb = int((dataset.train_idx.numel()+args.batch_size*args.sb_size-1)/(args.batch_size*args.sb_size))
    
    load_cache, sample, changeset_compute = 0, 0, 0
    total_cache_miss, total_io, total_load_time = 0, 0, 0

    for i in range(num_sb + 1):
        if args.verbose:
            tqdm.write ('Running {}th superbatch of total {} superbatches'.format(i, num_sb))

        if args.sample_mode:
            # Superbatch sample
            if args.verbose:
                tqdm.write ('Step 1: Superbatch Sample')
            cache, initial_cache_indices, sample_decompose = inspect(i, last=(i==num_sb), mode='train')
            load_cache += sample_decompose[0]
            sample += sample_decompose[1]
            changeset_compute += sample_decompose[2]
            torch.cuda.synchronize()
            if args.verbose:
                tqdm.write ('Step 1: Done')

        if i == 0:
            continue

        if not args.sample_mode:
            node_idx = dataset.shuffled_train_idx
            effective_sb_size = int((node_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size) if i==num_sb else args.sb_size
            cache = FeatureCache(args.feature_cache_size, effective_sb_size, num_nodes, mmapped_features, num_features, args.exp_name, i - 1, args.verbose)
            # Pass 1 and 2 are executed before starting sb sample.
            # We overlap only the pass 3 of changeset precomputation, 
            # which is the most time consuming part, with sb sample.
            iterptr, iters, initial_cache_indices = cache.pass_1_and_2()

            # Switch
            if args.verbose:
                tqdm.write ('Step 2: Switch')
            cache = switch(cache, initial_cache_indices.cpu())
            torch.cuda.synchronize()
            if args.verbose:
                tqdm.write ('Step 2: Done')

            # Main loop
            if args.verbose:
                tqdm.write ('Step 3: Main Loop')
            total_loss, total_correct, num_infeats, decompose_info = execute(i, cache, pbar, total_loss, total_correct, last=(i==num_sb), mode='train')
            load_time, cache_miss, io_traffic = decompose_info
            total_load_time += load_time
            total_cache_miss += cache_miss
            total_io += io_traffic
            
            total_num_infeats += num_infeats
            if args.verbose:
                tqdm.write ('Step 3: Done')

        # Delete obsolete runtime files
        # delete_trace(i)

    pbar.close()

    loss = total_loss / num_iter
    approx_acc = total_correct / dataset.train_idx.numel()

    return loss, approx_acc, total_num_infeats, (total_load_time, total_cache_miss, total_io), (load_cache, sample, changeset_compute)


@torch.no_grad()
def inference(mode='test'):
    model.eval()

    if mode == 'test':
        pbar = tqdm(total=dataset.test_idx.numel(), position=0, leave=True)
        num_sb = int((dataset.test_idx.numel()+args.batch_size*args.sb_size-1)/(args.batch_size*args.sb_size))
        num_iter = int((dataset.test_idx.numel()+args.batch_size-1) / args.batch_size)
    elif mode == 'valid':
        pbar = tqdm(total=dataset.val_idx.numel(), position=0, leave=True)
        num_sb = int((dataset.val_idx.numel()+args.batch_size*args.sb_size-1)/(args.batch_size*args.sb_size))
        num_iter = int((dataset.val_idx.numel()+args.batch_size-1) / args.batch_size)

    pbar.set_description('Evaluating')

    total_loss = total_correct = 0
    total_num_infeats = 0

    for i in range(num_sb + 1):
        if args.verbose:

            tqdm.write ('Running {}th superbatch of total {} superbatches'.format(i, num_sb))
        
        # Superbatch sample
        if args.verbose:
            tqdm.write ('Step 1: Superbatch Sample')
        cache, initial_cache_indices = inspect(i, last=(i==num_sb), mode=mode)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write ('Step 1: Done')

        if i == 0:
            continue

        # Switch
        if args.verbose:
            tqdm.write ('Step 2: Switch')
        cache = switch(cache, initial_cache_indices)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write ('Step 2: Done')

        # Main loop
        if args.verbose:
            tqdm.write ('Step 3: Main Loop')
        total_loss, total_correct, num_infeats = execute(i, cache, pbar, total_loss, total_correct, last=(i==num_sb), mode=mode)
        total_num_infeats += num_infeats
        if args.verbose:
            tqdm.write ('Step 3: Done')

        # Delete obsolete runtime files
        delete_trace(i)

    pbar.close()

    loss = total_loss / num_iter
    if mode == 'test':
        approx_acc = total_correct / dataset.test_idx.numel()
    elif mode == 'valid':
        approx_acc = total_correct / dataset.val_idx.numel()

    return loss, approx_acc, total_num_infeats


if __name__=='__main__':
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epoch_info_recoder = [[] for i in range(3)]
    epoch_sample_decompose = [[] for i in range(3)]

    epoch_time_list = []
    best_val_acc = final_test_acc = 0
    for epoch in range(args.num_epochs):
        if args.verbose:
            tqdm.write('\n==============================')
            tqdm.write('Running Epoch {}...'.format(epoch))

        start = time.time()
        loss, acc, num_infeats, decompose_info, sample_decompose = train(epoch)
        
        for i, record in enumerate(decompose_info):
            epoch_info_recoder[i].append(record)
        
        for i, record in enumerate(sample_decompose):
            epoch_sample_decompose[i].append(record)
        
        end = time.time()
        tqdm.write(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}, #Infeats: {num_infeats}')
        tqdm.write('Epoch time: {:.4f} s'.format(end - start))
        tqdm.write(f'Load time: {decompose_info[0]:.4f} s, Cache Miss: {decompose_info[1]}, IO Traffic: {decompose_info[2]}')
        tqdm.write(f'Load Cache: {sample_decompose[0]:.4f} s, Sample: {sample_decompose[1]:.4f} s, Changeset Compute: {sample_decompose[2]:.4f} s')
        epoch_time_list.append(end - start)

        if epoch > 3 and not args.train_only:
            val_loss, val_acc, num_infeats_valid = inference(mode='valid')
            test_loss, test_acc, num_infeats_test = inference(mode='test')
            tqdm.write ('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(val_loss, val_acc, test_loss, test_acc))
            tqdm.write('Valid #Infeats: {0}, Test #Infeats: {1}'.format(num_infeats_valid, num_infeats_test))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    
    avg_epoch_time = np.mean(epoch_time_list[1:]) if len(epoch_time_list) > 1 else epoch_time_list[0]
    print(f'Average Epoch Time: {avg_epoch_time} s')
    
    with open(args.log, "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        log_info = [
            args.dataset,
            args.sizes,
            args.batch_size,
            args.sb_size,
            f"{args.neigh_cache_size:g}",
            f"{args.feature_cache_size:g}",
            args.model,
            args.num_hiddens,
            args.dropout,
            args.num_epochs,
            args.train_only,
            args.sample_mode,
            round(avg_epoch_time, 2),
        ]
        for epoch_info in epoch_info_recoder:
            log_info.append(round(np.mean(epoch_info[1:]), 2))
        for epoch_info in epoch_sample_decompose:
            log_info.append(round(np.mean(epoch_info), 2))
        writer.writerow(log_info)

    if not args.train_only:
        tqdm.write('Final Test acc: {final_test_acc:.4f}')
