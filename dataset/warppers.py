#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math

import numba.typed.typedlist
from numba import jit
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler

from dataset.samplers import CustomSampler
from dataset.transforms import CDRDataTransform
from dataset.datasets import *
from utils.logger import InfoLogger
from utils.nn_utils import compute_knn_graph
from utils.umap_utils import fuzzy_simplicial_set_partial, simple_fuzzy


def build_dataset(data_file_path, dataset_name, root_dir):
    train_dataset = ProbCDRTextDataset(dataset_name, root_dir, True, data_file_path)
    data_augment = None
    return data_augment, None, train_dataset


class DataRepo:
    def __init__(self, n_neighbor):
        self.n_neighbor = n_neighbor
        self._total_n_samples = 0
        self._total_data = None
        self._total_label = None
        self._total_embeddings = None
        self._knn_manager = KNNManager(n_neighbor)

    def slide_window(self, out_num, *args):
        if out_num <= 0:
            return out_num

        self._total_data = self.get_total_data()[out_num:]
        self._total_label = self.get_total_label()[out_num:]
        self._total_embeddings = self.get_total_embeddings()[out_num:]
        self._knn_manager.slide_window(out_num)
        self._total_n_samples = self._total_data.shape[0]

        return out_num

    def post_slide(self, out_num):
        self._total_embeddings = self.get_total_embeddings()[out_num:]

    def get_n_samples(self):
        return self._total_n_samples

    def get_total_data(self):
        return self._total_data

    def get_total_label(self):
        return self._total_label

    def get_total_embeddings(self):
        return self._total_embeddings

    def get_knn_indices(self):
        return self._knn_manager.knn_indices

    def get_knn_dists(self):
        return self._knn_manager.knn_dists

    def update_embeddings(self, new_embeddings):
        self._total_embeddings = new_embeddings

    def add_new_data(self, data=None, embeddings=None, labels=None, knn_indices=None, knn_dists=None):
        if data is not None:
            if self._total_data is None:
                self._total_data = data
            else:
                self._total_data = np.append(self._total_data, data, axis=0)
            self._total_n_samples += data.shape[0]

        if embeddings is not None:
            if self._total_embeddings is None:
                self._total_embeddings = embeddings
            else:
                self._total_embeddings = np.concatenate([self._total_embeddings, embeddings], axis=0)

        if labels is not None:
            if self._total_label is None:
                self._total_label = np.array(labels)
            else:
                if isinstance(labels, list):
                    self._total_label = np.concatenate([self._total_label, labels])
                else:
                    self._total_label = np.append(self._total_label, labels)

        if knn_indices is not None and knn_dists is not None:
            self._knn_manager.add_new_kNN(knn_indices, knn_dists)


class DataSetWrapper(DataRepo):

    def __init__(self, similar_num, batch_size, n_neighbor, window_size):
        DataRepo.__init__(self, n_neighbor)
        self.similar_num = similar_num
        self.batch_size = batch_size
        self.batch_num = 0
        self.test_batch_num = 0
        self.train_dataset = None
        self.n_neighbor = n_neighbor
        self._window_size = window_size
        self.symmetric_nn_indices = None
        self.symmetric_nn_weights = None
        self.raw_knn_weights = None

    def get_data_loaders(self, epoch_num, dataset_name, root_dir, n_neighbors, knn_cache_path=None,
                         pairwise_cache_path=None, is_image=True, data_file_path=None, multi=False):
        self.n_neighbor = n_neighbors
        data_augment, test_dataset, train_dataset = build_dataset(data_file_path, dataset_name, root_dir)
        self.train_dataset = train_dataset
        if self._knn_manager.is_empty():
            knn_indices, knn_distances = compute_knn_graph(train_dataset.data, None, n_neighbors,
                                                           None, accelerate=False)
            self._knn_manager.add_new_kNN(knn_indices, knn_distances)

        self.distance2prob(train_dataset)

        train_num = self.update_transform(data_augment, epoch_num, is_image, train_dataset)

        train_indices = list(range(train_num))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset, train_indices, [])

        return train_loader, train_num

    def update_transform(self, data_augment, epoch_num, is_image, train_dataset):
        train_dataset.update_transform(CDRDataTransform(epoch_num, self.similar_num, train_dataset, is_image,
                                                        data_augment, self.n_neighbor,
                                                        self.symmetric_nn_indices[:self.get_n_samples()],
                                                        self.symmetric_nn_weights[:self.get_n_samples()]))
        train_num = train_dataset.data_num

        self.batch_num = math.floor(train_num / self.batch_size)
        return train_num

    def distance2prob(self, train_dataset):
        train_dataset.prob_process(self._knn_manager.knn_indices, self._knn_manager.knn_dists, self.n_neighbor)

        self._update_knn_stat(train_dataset)

    def _update_knn_stat(self, train_dataset):
        self.symmetric_nn_indices = train_dataset.symmetry_knn_indices
        self.symmetric_nn_weights = train_dataset.symmetry_knn_weights
        self.raw_knn_weights = train_dataset.raw_knn_weights

    def get_train_validation_data_loaders(self, train_dataset, train_indices, val_indices):
        InfoLogger.info("Train num = {} Val num = {}".format(len(train_indices), len(val_indices)))

        train_sampler = CustomSampler(train_indices, False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  drop_last=True, shuffle=True)

        valid_loader = None
        return train_loader, valid_loader


class StreamingDatasetWrapper(DataSetWrapper):
    def __init__(self, batch_size, n_neighbor, window_size, device=None):
        DataSetWrapper.__init__(self, 1, batch_size, n_neighbor, window_size)
        self._cached_neighbor_change_indices = []
        self.cur_neighbor_changed_indices = None
        self.__replaced_raw_weights = []
        self.__sigmas = None
        self.__rhos = None
        self._concat_num = 1000
        self._tmp_neighbor_weights = np.ones((1, self.n_neighbor))
        self._unfitted_data_num = 0
        self.out_sum = 0

    def slide_window(self, out_num, *args):
        if out_num <= 0:
            return
        super().slide_window(out_num)
        cached_candidate_indices, cached_candidate_dists, cached_candidate_idx = args

        knn_indices = self.get_knn_indices()
        knn_dists = self.get_knn_dists()
        knn_indices -= out_num
        out_indices = np.argwhere(self.get_knn_indices() < 0)
        changed_data_idx = out_indices[:, 0]

        total_embeddings = self.get_total_embeddings()
        new_k = 5 * self.n_neighbor + 1
        dist_cal_set = []
        for i, item in enumerate(changed_data_idx):
            if item in dist_cal_set:
                continue
            n = len(cached_candidate_indices[item])
            idx = cached_candidate_idx[item]
            while (idx < n) and (cached_candidate_indices[item][idx] < 0):
                idx += 1

            if idx < n:
                knn_indices[item, out_indices[i, 1]] = cached_candidate_indices[item][idx]
                knn_dists[item, out_indices[i, 1]] = cached_candidate_dists[item][idx]
                cached_candidate_idx[item] = idx
            else:
                dist_cal_set.append(item)
                dist = cdist(total_embeddings[item][np.newaxis, :], total_embeddings).squeeze()

                tmp_sorted_indices = np.argsort(dist)
                knn_indices[item] = tmp_sorted_indices[1:1 + self.n_neighbor]
                knn_dists[item] = dist[knn_indices[item]]
                cached_candidate_indices[item] = tmp_sorted_indices[1 + self.n_neighbor:new_k]
                cached_candidate_dists[item] = dist[cached_candidate_indices[item]]
                cached_candidate_idx[item] = 0

        self._knn_manager.update_knn_graph(knn_indices, knn_dists)
        self.__sigmas = self.__sigmas[out_num:]
        self.__rhos = self.__rhos[out_num:]
        self.raw_knn_weights = self.raw_knn_weights[out_num:]

        self.train_dataset.slide_window(out_num)

        if len(self._cached_neighbor_change_indices) > 0:
            self._cached_neighbor_change_indices -= out_num

        self._cached_neighbor_change_indices = np.append(self._cached_neighbor_change_indices,
                                                         np.unique(changed_data_idx))
        return out_num

    def update_unfitted_data_num(self, unfitted_num):
        self._unfitted_data_num = unfitted_num

    def distance2prob(self, train_dataset):
        sigmas, rhos = train_dataset.prob_process(self._knn_manager.knn_indices, self._knn_manager.knn_dists,
                                                  self.n_neighbor, return_meta=True)[-2:]
        self._update_knn_stat(train_dataset)
        return sigmas, rhos

    def get_data_loaders(self, epoch_num, dataset_name, root_dir, n_neighbors, knn_cache_path=None,
                         pairwise_cache_path=None, is_image=True, data_file_path=None, multi=False):
        self.n_neighbor = n_neighbors
        train_dataset = ProbCDRTextDataset(None, None, True,
                                           train_data=[self.get_total_data(), self.get_total_label()])

        self.train_dataset = train_dataset
        if self._knn_manager.is_empty():
            knn_indices, knn_distances = compute_knn_graph(train_dataset.data, None, n_neighbors,
                                                           None, accelerate=False)
            self._knn_manager.add_new_kNN(knn_indices, knn_distances)

        self.__sigmas, self.__rhos = self.distance2prob(train_dataset)

        train_num = self.update_transform(None, epoch_num, is_image, train_dataset)
        train_indices = list(range(train_num))

        train_loader = self._get_train_data_loader(train_dataset, train_indices)

        return train_loader, train_num

    def update_data_loaders(self, epoch_nums, sampled_indices, multi=False):
        self.train_dataset.transform.update(self.train_dataset, epoch_nums,
                                            self.symmetric_nn_indices[:self.get_n_samples()],
                                            self.symmetric_nn_weights[:self.get_n_samples()])

        train_loader = self._get_train_data_loader(self.train_dataset, sampled_indices)
        return train_loader, len(sampled_indices)

    def get_dataset(self):
        train_dataset = ProbCDRTextDataset(None, None, True,
                                           train_data=[self.get_total_data(), self.get_total_label()])

        data_augment = None
        return data_augment, train_dataset

    def _get_train_data_loader(self, train_dataset, train_indices, shuffle=True):
        InfoLogger.info("Train num = {}".format(len(train_indices)))

        train_sampler = CustomSampler(train_indices, shuffle)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  drop_last=True, shuffle=False)
        return train_loader

    def update_knn_graph(self, pre_n_samples, new_data, data_num_list, candidate_indices=None, candidate_dists=None,
                         cut_num=None, update_similarity=True, symmetric=True):
        knn_distances = self._knn_manager.knn_dists if cut_num is None else self._knn_manager.knn_dists[:cut_num]
        knn_indices = self._knn_manager.knn_indices if cut_num is None else self._knn_manager.knn_indices[:cut_num]

        new_data_num = new_data.shape[0]
        neighbor_changed_indices = np.arange(pre_n_samples, pre_n_samples + new_data.shape[0]).tolist()

        self.__replaced_raw_weights = []
        neighbor_changed_indices = self._knn_manager.update_previous_kNN(new_data.shape[0], pre_n_samples,
                                                                         candidate_indices,
                                                                         candidate_dists,
                                                                         neighbor_changed_indices)

        knn_changed_neighbor_meta = self._knn_manager.get_pre_neighbor_changed_positions()
        if len(knn_changed_neighbor_meta) > 0:
            changed_neighbor_sims = self.raw_knn_weights[knn_changed_neighbor_meta[:, 1]]
            self.__replaced_raw_weights = changed_neighbor_sims[:, -1] / np.sum(changed_neighbor_sims, axis=1)

        self.raw_knn_weights = np.concatenate([self.raw_knn_weights, np.ones((new_data_num, self.n_neighbor))],
                                              axis=0)
        self.__sigmas = np.concatenate([self.__sigmas, np.ones(new_data_num)])
        self.__rhos = np.concatenate([self.__rhos, np.ones(new_data_num)])

        if not update_similarity:
            if pre_n_samples >= self._window_size:
                cur_update_indices = np.arange(self._window_size - new_data.shape[0], self._window_size).tolist()
            else:
                cur_update_indices = np.arange(pre_n_samples, pre_n_samples + new_data.shape[0]).tolist()
        else:
            cur_update_indices = neighbor_changed_indices

        if not symmetric:
            cur_sigmas, cur_rhos, updated_raw_knn_weights = \
                simple_fuzzy(knn_indices[cur_update_indices], knn_distances[cur_update_indices])
            self.raw_knn_weights[cur_update_indices] = updated_raw_knn_weights
        else:
            umap_graph, cur_sigmas, cur_rhos, self.raw_knn_weights = \
                fuzzy_simplicial_set_partial(knn_indices, knn_distances, self.raw_knn_weights,
                                             cur_update_indices, apply_set_operations=symmetric,
                                             return_coo_results=symmetric)

        self.__sigmas[cur_update_indices] = cur_sigmas
        self.__rhos[cur_update_indices] = cur_rhos

        if not update_similarity:
            self._cached_neighbor_change_indices = np.union1d(self._cached_neighbor_change_indices,
                                                              neighbor_changed_indices)

        return None

    def update_cached_neighbor_similarities(self):
        if len(self._cached_neighbor_change_indices) <= 0:
            return

        valid_indices = np.where(self._cached_neighbor_change_indices >= 0)[0]
        if len(valid_indices) <= 0:
            self._cached_neighbor_change_indices = np.array([])
            return

        self._cached_neighbor_change_indices = np.unique(self._cached_neighbor_change_indices[valid_indices]).astype(
            int)

        umap_graph, sigmas, rhos, self.raw_knn_weights = \
            fuzzy_simplicial_set_partial(self._knn_manager.knn_indices, self._knn_manager.knn_dists,
                                         self.raw_knn_weights,
                                         self._cached_neighbor_change_indices)

        updated_sym_nn_indices, updated_symm_nn_weights = extract_csr(umap_graph, self._cached_neighbor_change_indices)

        self.__sigmas[self._cached_neighbor_change_indices] = sigmas
        self.__rhos[self._cached_neighbor_change_indices] = rhos
        self.symmetric_nn_weights[self._cached_neighbor_change_indices] = updated_symm_nn_weights
        self.symmetric_nn_indices[self._cached_neighbor_change_indices] = updated_sym_nn_indices
        self._cached_neighbor_change_indices = []

    def get_pre_neighbor_changed_info(self):
        pre_changed_neighbor_meta = self._knn_manager.get_pre_neighbor_changed_positions()
        neighbor_changed_indices = pre_changed_neighbor_meta[:, 1] if len(pre_changed_neighbor_meta) > 0 else []
        replaced_raw_weights = self.__replaced_raw_weights
        replaced_indices = pre_changed_neighbor_meta[:, 3] if len(pre_changed_neighbor_meta) > 0 else []
        anchor_positions = pre_changed_neighbor_meta[:, 2] if len(pre_changed_neighbor_meta) > 0 else []
        return neighbor_changed_indices, replaced_raw_weights, replaced_indices, anchor_positions

    def add_new_data(self, data=None, embeddings=None, labels=None, knn_indices=None, knn_dists=None):
        super().add_new_data(data, embeddings, labels, knn_indices, knn_dists)
        if self.train_dataset is not None:
            self.train_dataset.add_new_data(data, labels)

    def cal_old2new_relationship(self, old_n_samples, reduction="max"):
        old_data = self.get_total_data()[:old_n_samples]
        new_data = self.get_total_data()[old_n_samples:]
        old_rhos = self.__rhos[:old_n_samples]
        old_sigmas = self.__sigmas[:old_n_samples]
        dists = cdist(old_data, new_data)
        normed_dists = dists - old_rhos[:, np.newaxis]
        normed_dists[normed_dists < 0] = 0
        total_relationships = np.exp(-normed_dists / old_sigmas[:, np.newaxis])

        if reduction == "mean":
            relationships = np.mean(total_relationships, axis=1)
        elif reduction == "max":
            relationships = np.max(total_relationships, axis=1)
        else:
            raise RuntimeError("'reduction' should be one of 'mean/max'")

        return 1 - relationships

    def update_previous_info(self, pre_num, new_self, out_during_update, skipped_slide_num):
        if pre_num <= out_during_update:
            return
        self.__sigmas[:pre_num - out_during_update] = new_self.__sigmas[out_during_update:pre_num]
        self.__rhos[:pre_num - out_during_update] = new_self.__rhos[out_during_update:pre_num]
        self.raw_knn_weights[:pre_num - out_during_update] = new_self.raw_knn_weights[out_during_update:pre_num]
        self._knn_manager.knn_indices[:pre_num - out_during_update] = new_self.get_knn_indices()[
                                                                      out_during_update:pre_num]
        self._knn_manager.knn_dists[:pre_num - out_during_update] = new_self.get_knn_dists()[out_during_update:pre_num]

    def get_data_neighbor_mean_std_dist(self):
        knn_dists = self._knn_manager.knn_dists[
                    :-self._unfitted_data_num] if self._unfitted_data_num > 0 else self._knn_manager.knn_dists
        mean_per_data = np.mean(knn_dists, axis=1)
        return np.mean(mean_per_data), np.std(mean_per_data)

    def get_embedding_neighbor_mean_std_dist(self):
        total_embeddings = self._total_embeddings[:-self._unfitted_data_num, np.newaxis, :] \
            if self._unfitted_data_num > 0 else self._total_embeddings[:, np.newaxis, :]
        knn_indices = self._knn_manager.knn_indices[:-self._unfitted_data_num] \
            if self._unfitted_data_num > 0 else self._knn_manager.knn_indices
        pre_low_neighbor_embedding_dists = \
            np.linalg.norm(total_embeddings - np.reshape(
                self._total_embeddings[np.ravel(knn_indices)],
                (self.get_n_samples() - self._unfitted_data_num, self.n_neighbor, -1)), axis=-1)

        mean_per_data = np.mean(pre_low_neighbor_embedding_dists, axis=1)

        return np.mean(mean_per_data), np.std(mean_per_data)


class KNNManager:
    def __init__(self, k):
        self.k = k
        self.knn_indices = None
        self.knn_dists = None
        self._pre_neighbor_changed_meta = []

    def update_knn_graph(self, knn_indices, knn_dists):
        self.knn_indices = knn_indices
        self.knn_dists = knn_dists

    def slide_window(self, out_num):
        if self.knn_indices is None or out_num <= 0:
            return

        self.knn_indices = self.knn_indices[out_num:]
        self.knn_dists = self.knn_dists[out_num:]

    def is_empty(self):
        return self.knn_indices is None

    def get_pre_neighbor_changed_positions(self):
        return self._pre_neighbor_changed_meta

    def add_new_kNN(self, new_knn_indices, new_knn_dists):
        if self.knn_indices is None:
            self.knn_indices = new_knn_indices
            self.knn_dists = new_knn_dists
            return

        if new_knn_indices is not None:
            self.knn_indices = np.concatenate([self.knn_indices, new_knn_indices], axis=0)
        if new_knn_dists is not None:
            self.knn_dists = np.concatenate([self.knn_dists, new_knn_dists], axis=0)

    def update_previous_kNN(self, new_data_num, pre_n_samples, candidate_indices, candidate_dists,
                            neighbor_changed_indices=None, symm=True):
        c_neighbor_changed_indices, self._pre_neighbor_changed_meta, self.knn_indices, self.knn_dists = \
            _do_update(new_data_num, pre_n_samples, numba.typed.typedlist.List(candidate_indices),
                       numba.typed.typedlist.List(candidate_dists), self.knn_indices,
                       self.knn_dists, symm)
        self._pre_neighbor_changed_meta = np.array(self._pre_neighbor_changed_meta, dtype=int)
        if neighbor_changed_indices is not None:
            c_neighbor_changed_indices.extend(neighbor_changed_indices)
        return c_neighbor_changed_indices


@jit(nopython=True)
def _do_update(new_data_num, pre_n_samples, candidate_indices_list, candidate_dists_list, knn_indices, knn_dists,
               symm=True):
    pre_neighbor_changed_meta = []
    neighbor_changed_indices = []
    for i in range(new_data_num):
        candidate_indices = candidate_indices_list[i]
        candidate_dists = candidate_dists_list[i]

        for j, data_idx in enumerate(candidate_indices):
            if knn_dists[data_idx][-1] <= candidate_dists[j]:
                continue

            if data_idx not in neighbor_changed_indices:
                neighbor_changed_indices.append(data_idx)
            insert_index = knn_dists.shape[1] - 1
            while insert_index >= 0 and candidate_dists[j] <= knn_dists[data_idx][insert_index]:
                insert_index -= 1

            if symm and knn_indices[data_idx][-1] not in neighbor_changed_indices:
                neighbor_changed_indices.append(knn_indices[data_idx][-1])

            pre_neighbor_changed_meta.append(
                [pre_n_samples + i, data_idx, insert_index + 1, knn_indices[data_idx][-1]])
            knn_dists[data_idx][insert_index + 2:] = knn_dists[data_idx][insert_index + 1:-1]
            knn_dists[data_idx][insert_index + 1] = candidate_dists[j]
            knn_indices[data_idx][insert_index + 2:] = knn_indices[data_idx][insert_index + 1:-1]
            knn_indices[data_idx][insert_index + 1] = pre_n_samples

    return neighbor_changed_indices, pre_neighbor_changed_meta, knn_indices, knn_dists


# @jit
def extract_csr(csr_graph, indices, norm=True):
    nn_indices = []
    nn_weights = []

    for i in indices:
        pre = csr_graph.indptr[i]
        idx = csr_graph.indptr[i + 1]
        cur_indices = csr_graph.indices[pre:idx]
        cur_weights = csr_graph.data[pre:idx]

        nn_indices.append(cur_indices)
        if norm:
            nn_weights.append(cur_weights / np.sum(cur_weights))
        else:
            nn_weights.append(cur_weights)
    return nn_indices, nn_weights
