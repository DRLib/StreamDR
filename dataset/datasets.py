#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os

import h5py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from utils.nn_utils import compute_knn_graph
from utils.umap_utils import fuzzy_simplicial_set, construct_edge_dataset

MACHINE_EPSILON = np.finfo(np.double).eps


class TextDataset(Dataset):
    def __init__(self, dataset_name, root_dir, train=True, data_file_path=None, train_data=None, test_data=None):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        if train and train_data is None:
            self.data_file_path = os.path.join(root_dir, dataset_name + ".h5") if data_file_path is None else data_file_path
        self.train = train
        self.data = None
        self.targets = None
        self.data_num = 0
        self.min_neighbor_num = 0
        self.symmetry_knn_indices = None
        self.symmetry_knn_weights = None
        self.symmetry_knn_dists = None
        self.transform = None
        self.__load_data(train_data, test_data)

    def __len__(self):
        return self.data.shape[0]

    def __load_data(self, train_data, test_data):
        if self.train and train_data is not None:
            self.data = train_data[0]
            self.targets = train_data[1]
            return
        elif not self.train and test_data is not None:
            self.data = test_data[0]
            self.targets = test_data[1]
            return

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            train_data, train_labels = \
                load_local_h5_by_path(self.data_file_path, ['x', 'y'])
            self.data = train_data
            self.targets = train_labels
        else:
            self.data = test_data

        self.data_num = self.data.shape[0]

    def __getitem__(self, index):
        text, target = self.data[index], int(self.targets[index])
        text = torch.tensor(text, dtype=torch.float)

        return text, target

    def _check_exists(self):
        return os.path.exists(self.data_file_path)

    def update_transform(self, new_transform):
        self.transform = new_transform

    def get_data(self, index):
        res = self.data[index]
        return torch.tensor(res, dtype=torch.float)

    def get_all_data(self, data_num=-1):
        if data_num == -1:
            return self.data
        else:
            return self.data[torch.randperm(self.data_num)[:data_num], :]

    def simple_preprocess(self, knn_indices, knn_distances):
        min_dis = np.expand_dims(np.min(knn_distances, axis=1), 1).repeat(knn_indices.shape[1], 1)
        max_dis = np.expand_dims(np.max(knn_distances, axis=1), 1).repeat(knn_indices.shape[1], 1)
        knn_distances = (knn_distances - min_dis) / (max_dis - min_dis)
        normal_knn_weights = np.exp(-knn_distances ** 2)
        normal_knn_weights /= np.expand_dims(np.sum(normal_knn_weights, axis=1), 1). \
            repeat(knn_indices.shape[1], 1)

        n_samples, n_neighbors = knn_indices.shape
        csr_row = np.expand_dims(np.arange(0, n_samples, 1), 1).repeat(n_neighbors, 1).ravel()
        csr_nn_weights = csr_matrix((normal_knn_weights.ravel(), (csr_row, knn_indices[:, :n_neighbors].ravel())),
                                    shape=(n_samples, n_samples))
        symmetric_nn_weights = csr_nn_weights + csr_nn_weights.T
        nn_indices, nn_weights, self.min_neighbor_num, raw_weights, _ = get_kw_from_coo(symmetric_nn_weights,
                                                                                        n_neighbors,
                                                                                        n_samples)

        self.symmetry_knn_indices = np.array(nn_indices, dtype=object)
        self.symmetry_knn_weights = np.array(nn_weights, dtype=object)


class ProbTextDataset(TextDataset):
    def __init__(self, dataset_name, root_dir, train=True, repeat=1, data_file_path=None, train_data=None,
                 test_data=None):
        TextDataset.__init__(self, dataset_name, root_dir, train, data_file_path, train_data, test_data)
        self.repeat = repeat
        self.edge_data = None
        self.edge_num = None
        self.edge_weight = None
        self.raw_knn_weights = None

    def build_fuzzy_simplicial_set(self, knn_cache_path, pairwise_cache_path, n_neighbors, metric="euclidean",
                                   max_candidates=60):
        knn_indices, knn_distances = compute_knn_graph(self.data, knn_cache_path, n_neighbors, pairwise_cache_path,
                                                       metric, max_candidates)
        umap_graph, sigmas, rhos, self.raw_knn_weights = fuzzy_simplicial_set(
            X=self.data,
            n_neighbors=n_neighbors,
            knn_indices=knn_indices,
            knn_dists=knn_distances)
        return umap_graph, sigmas, rhos

    def umap_process(self, knn_cache_path, pairwise_cache_path, n_neighbors, embedding_epoch, metric="euclidean",
                     max_candidates=60, return_meta=False):
        umap_graph, sigmas, rhos = self.build_fuzzy_simplicial_set(knn_cache_path, pairwise_cache_path, n_neighbors,
                                                                   metric, max_candidates)
        self.edge_data, self.edge_num, self.edge_weight = construct_edge_dataset(
            self.data, umap_graph, embedding_epoch)

        if return_meta:
            return self.edge_data, self.edge_num, sigmas, rhos

        return self.edge_data, self.edge_num

    def __getitem__(self, index):
        to_data, from_data = self.edge_data[0][index], self.edge_data[1][index]

        return torch.tensor(to_data, dtype=torch.float), torch.tensor(from_data, dtype=torch.float)

    def __len__(self):
        return self.edge_num


class CDRTextDataset(TextDataset):
    def __init__(self, dataset_name, root_dir, train=True, data_file_path=None, train_data=None, test_data=None):
        TextDataset.__init__(self, dataset_name, root_dir, train, data_file_path, train_data, test_data)

    def __getitem__(self, index):
        text, target = self.data[index], int(self.targets[index])
        x, x_sim, idx, sim_idx = self.transform(text, index)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
            x_sim = torch.tensor(x_sim, dtype=torch.float)
        return [x, x_sim, idx, sim_idx], target

    def sample_data(self, indices):
        x = self.data[indices]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        return x

    def add_new_data(self, data, labels=None):
        if data is not None:
            self.data = data if self.data is None else np.concatenate([self.data, data], axis=0)
        if labels is not None:
            self.targets = labels if self.targets is None else np.append(self.targets, labels)

    def slide_window(self, out_num):
        if out_num <= 0:
            return
        self.data = self.data[out_num:]
        self.targets = self.targets[out_num:]


class ProbCDRTextDataset(CDRTextDataset):
    def __init__(self, dataset_name, root_dir, train=True, data_file_path=None, train_data=None, test_data=None):
        CDRTextDataset.__init__(self, dataset_name, root_dir, train, data_file_path, train_data, test_data)
        self.umap_graph = None
        self.raw_knn_weights = None
        self.min_neighbor_num = None

    def build_fuzzy_simplicial_set(self, knn_indices, knn_distances, n_neighbors):
        self.umap_graph, sigmas, rhos, self.raw_knn_weights, knn_dist = fuzzy_simplicial_set(
            X=self.data, n_neighbors=n_neighbors, knn_indices=knn_indices,
            knn_dists=knn_distances, return_dists=True)

        self.symmetry_knn_dists = knn_dist.tocoo()
        return self.umap_graph, sigmas, rhos

    def prob_process(self, knn_indices, knn_distances, n_neighbors, return_meta=False):
        self.umap_graph, sigmas, rhos = self.build_fuzzy_simplicial_set(knn_indices, knn_distances, n_neighbors)

        self.data_num = knn_indices.shape[0]
        n_samples = self.data_num

        nn_indices, nn_weights, self.min_neighbor_num, raw_weights, nn_dists \
            = get_kw_from_coo(self.umap_graph, n_neighbors, n_samples, self.symmetry_knn_dists)

        self.symmetry_knn_indices = np.array(nn_indices, dtype=object)
        self.symmetry_knn_weights = np.array(nn_weights, dtype=object)
        self.symmetry_knn_dists = np.array(nn_dists, dtype=object)

        if return_meta:
            return None, None, sigmas, rhos

        return None, None


def get_kw_from_coo(csr_graph, n_neighbors, n_samples, dist_csr=None):
    nn_indices = []
    nn_weights = []
    raw_weights = []
    nn_dists = []

    tmp_min_neighbor_num = n_neighbors
    for i in range(1, n_samples + 1):
        pre = csr_graph.indptr[i - 1]
        idx = csr_graph.indptr[i]
        cur_indices = csr_graph.indices[pre:idx]
        if dist_csr is not None:
            nn_dists.append(dist_csr.data[pre:idx])
        tmp_min_neighbor_num = min(tmp_min_neighbor_num, idx - pre)
        cur_weights = csr_graph.data[pre:idx]

        nn_indices.append(cur_indices)
        cur_sum = np.sum(cur_weights)
        nn_weights.append(cur_weights / cur_sum)
        raw_weights.append(cur_weights)
    return nn_indices, nn_weights, tmp_min_neighbor_num, raw_weights, nn_dists


def load_local_h5_by_path(dataset_path, keys):
    f = h5py.File(dataset_path, "r")
    res = []
    for key in keys:
        if key in f.keys():
            res.append(f[key][:])
        else:
            res.append(None)
    f.close()
    return res
