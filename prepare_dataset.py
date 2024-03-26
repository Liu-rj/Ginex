import argparse
from ogb.nodeproppred import PygNodePropPredDataset
import scipy
import numpy as np
import json
import torch
import os
from load_graph import *


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="ogbn-papers100M")
argparser.add_argument(
    "--path", type=str, default="/nvme2n1/offgs_dataset/ogbn-papers100M-offgs"
)
args = argparser.parse_args()

# # Download/load dataset
# print('Loading dataset...')
# root = './dataset'
# os.makedirs(root, exist_ok=True)
# dataset = PygNodePropPredDataset(args.dataset, root)
# dataset_path = os.path.join(root, args.dataset + '-ginex')
# print('Done!')

# # Construct sparse formats
# print('Creating coo/csc/csr format of dataset...')
# num_nodes = dataset[0].num_nodes
# coo = dataset[0].edge_index.numpy()
# v = np.ones_like(coo[0])
# coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
# csc = coo.tocsc()
# csr = coo.tocsr()
# print('Done!')

# # Save csc-formatted dataset
# indptr = csc.indptr.astype(np.int64)
# indices = csc.indices.astype(np.int64)
# features = dataset[0].x
# labels = dataset[0].y

root = "./dataset"
os.makedirs(root, exist_ok=True)
dataset_path = os.path.join(root, args.dataset + "-ginex")

# label_offset = 0
# if args.dataset.startswith("ogbn"):
#     dataset = load_ogb(args.dataset, "/efs/rjliu/dataset")
# elif args.dataset.startswith("igb"):
#     dataset = load_igb(args)
# elif args.dataset == "mag240m":
#     dataset = load_mag240m("/efs/rjliu/dataset/mag240m", only_graph=False)
#     label_offset = dataset[-1]
#     dataset = dataset[:-1]
# elif args.dataset == "friendster":
#     dataset = load_friendster("/efs/rjliu/dataset/friendster", 128, 20)
# else:
#     raise NotImplementedError

dataset = load_offgs_dataset(args.path)

g, features, labels, n_classes, splitted_idx, label_offset = dataset
print(g)
print(g.formats())
g.create_formats_()
print(g.formats())
indptr, indices, _ = g.adj_tensors("csc")

train_ratio = splitted_idx["train"].numel() / g.num_nodes()
print(f"Train ratio: {train_ratio}")

os.makedirs(dataset_path, exist_ok=True)
indptr_path = os.path.join(dataset_path, "indptr.dat")
indices_path = os.path.join(dataset_path, "indices.dat")
features_path = os.path.join(dataset_path, "features.dat")
labels_path = os.path.join(dataset_path, "labels.dat")
conf_path = os.path.join(dataset_path, "conf.json")
split_idx_path = os.path.join(dataset_path, "split_idx.pth")

print("Saving indptr...")
indptr_np = indptr.numpy()
print(indptr_np.dtype)
indptr_mmap = np.memmap(
    indptr_path, mode="w+", shape=indptr_np.shape, dtype=indptr_np.dtype
)
indptr_mmap[:] = indptr[:]
indptr_mmap.flush()
print("Done!")

print("Saving indices...")
indices_np = indices.numpy()
print(indices_np.dtype)
indices_mmap = np.memmap(
    indices_path, mode="w+", shape=indices_np.shape, dtype=indices_np.dtype
)
indices_mmap[:] = indices[:]
indices_mmap.flush()
print("Done!")

print("Saving features...")
features_mmap = np.memmap(
    features_path, mode="w+", shape=features.shape, dtype=np.float32
)
features_mmap[:] = features[:]
features_mmap.flush()
print("Done!")

print("Saving labels...")
labels = labels.type(torch.float32)
labels_mmap = np.memmap(labels_path, mode="w+", shape=labels.shape, dtype=np.float32)
labels_mmap[:] = labels[:]
labels_mmap.flush()
print("Done!")

print("Making conf file...")
mmap_config = dict()
mmap_config["num_nodes"] = int(g.num_nodes())
mmap_config["indptr_shape"] = tuple(indptr.shape)
mmap_config["indptr_dtype"] = str(indptr_mmap.dtype)
mmap_config["indices_shape"] = tuple(indices.shape)
mmap_config["indices_dtype"] = str(indices_mmap.dtype)
mmap_config["features_shape"] = tuple(features_mmap.shape)
mmap_config["features_dtype"] = str(features_mmap.dtype)
mmap_config["labels_shape"] = tuple(labels_mmap.shape)
mmap_config["labels_dtype"] = str(labels_mmap.dtype)
mmap_config["num_classes"] = int(n_classes)
mmap_config["label_offset"] = int(label_offset)
json.dump(mmap_config, open(conf_path, "w"))
print("Done!")

print("Saving split index...")
torch.save(splitted_idx, split_idx_path)
print("Done!")

# Calculate and save score for neighbor cache construction
print("Calculating score for neighbor cache construction...")
score_path = os.path.join(dataset_path, "nc_score.pth")
csc_indptr_tensor = indptr
csr_indptr_tensor = g.adj_tensors("csr")[0]

eps = 0.00000001
in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1]) + eps
out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps
score = out_num_neighbors / in_num_neighbors
print("Done!")

print("Saving score...")
torch.save(score, score_path)
print("Done!")
