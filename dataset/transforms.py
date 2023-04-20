#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math
import random

import numpy as np
import torch
from torchvision import transforms as transforms


def assign_weighted_neighbor_samples(symmetry_knn_indices, symmetry_knn_weights, n_neighbors, epoch_num, similar_num=1,
                                     ex_factor=None, prev_nums=None):
    if ex_factor is None:
        ex_factor = 1

    n_samples = symmetry_knn_indices.shape[0]
    if prev_nums is None or len(prev_nums) == 0:
        repo_size = epoch_num * similar_num * ex_factor
        sample_num_per_neighbor = symmetry_knn_weights * repo_size
        neighbor_sample_repo = np.empty((n_samples, repo_size), dtype=np.int)
    else:
        sample_num_per_neighbor = symmetry_knn_weights * epoch_num * similar_num
        repo_size = math.ceil(epoch_num * similar_num * np.max(np.array(ex_factor)))
        neighbor_sample_repo = np.empty((n_samples, repo_size), dtype=np.int)
        base = n_samples
        for i in range(len(prev_nums)):
            num, fa = prev_nums[i], ex_factor[i]
            base -= num
            sample_num_per_neighbor[base:base+num] *= fa

    for i in range(n_samples):
        sample_num_per_neighbor[i] = np.ceil(sample_num_per_neighbor[i]).astype(np.int)
        if np.sum(sample_num_per_neighbor[i]) < epoch_num * similar_num:
            sample_num_per_neighbor[i] = \
                np.ones_like(sample_num_per_neighbor[i]) * np.ceil(epoch_num * similar_num / n_neighbors)

        tmp_num = len(symmetry_knn_indices[i])
        tmp_repo = np.repeat(symmetry_knn_indices[i].astype(np.int), sample_num_per_neighbor[i][:tmp_num].astype(np.int).squeeze())
        num = min(repo_size, len(tmp_repo))
        np.random.shuffle(tmp_repo)
        neighbor_sample_repo[i, :num] = tmp_repo[:num].astype(np.int)

    return neighbor_sample_repo


class CDRDataTransform(object):
    def __init__(self, epoch_num, similar_num, train_dataset, is_image, transform, n_neighbors, norm_nn_indices,
                 norm_nn_weights, ex_factor=1):

        self.n_neighbors = n_neighbors
        self.transform = transform
        self.train_dataset = train_dataset
        self.n_samples = norm_nn_indices.shape[0]

        self.neighbor_sample_repo = None
        self.neighbor_sample_index = None
        self.pre_norm_nn_indices = norm_nn_indices
        self.pre_norm_nn_weights = norm_nn_weights

        self.build_neighbor_repo(epoch_num, n_neighbors, norm_nn_indices, norm_nn_weights, ex_factor)

        self.is_image = is_image
        if self.is_image:
            self.transform = transforms.ToTensor()

    def build_neighbor_repo(self, epoch_num, n_neighbors, norm_nn_indices=None, norm_nn_weights=None, ex_factor=1):
        if norm_nn_indices is None:
            norm_nn_indices = self.pre_norm_nn_indices

        if norm_nn_weights is None:
            norm_nn_weights = self.pre_norm_nn_weights

        self.neighbor_sample_repo = assign_weighted_neighbor_samples(norm_nn_indices,
                                                                     norm_nn_weights, n_neighbors,
                                                                     epoch_num, ex_factor)
        self.neighbor_sample_index = np.zeros(self.n_samples, dtype=np.int)

    def update(self, train_dataset, epoch_num, norm_nn_indices, norm_nn_weights):
        self.pre_norm_nn_weights = norm_nn_weights
        self.pre_norm_nn_indices = norm_nn_indices
        self.train_dataset = train_dataset
        self.n_samples = train_dataset.data.shape[0]
        self.neighbor_sample_repo = assign_weighted_neighbor_samples(norm_nn_indices,
                                                                     norm_nn_weights, self.n_neighbors,
                                                                     epoch_num)
        self.neighbor_sample_index = np.zeros(self.n_samples, dtype=np.int)

    def _neighbor_index_fixed(self, index):
        sim_index = self.neighbor_sample_repo[index][self.neighbor_sample_index[index]]
        self.neighbor_sample_index[index] += 1
        return sim_index

    def __call__(self, sample, index):
        x = sample
        if self.transform is not None:
            x = self.transform(sample)
        else:
            x = torch.tensor(x, dtype=torch.float)

        sim_index = self._neighbor_index_fixed(index)

        x_sim = self.train_dataset.get_data(sim_index)
        if self.transform is not None:
            x_sim = self.transform(x_sim)

        return x.float(), x_sim.float(), index, sim_index

