import random
import time
import torch
import numba
import numpy as np
import scipy

from scipy import optimize
from utils.loss_grads import nce_loss_single, nce_loss_grad
from utils.umap_utils import find_ab_params


def initial_embedding_with_weighted_mean(neighbor_sims, neighbor_embeddings):
    normed_neighbor_sims = neighbor_sims / np.sum(neighbor_sims)
    cur_initial_embedding = np.sum(normed_neighbor_sims[:, np.newaxis] * neighbor_embeddings, axis=0)[np.newaxis, :]
    return cur_initial_embedding


@numba.jit(nopython=True)
def _local_move(embedding, new_embeddings, neighbor_embeddings, anchor_position, i, replaced_sim, target_sims):
    anchor_sims = target_sims[i, anchor_position[i]] / np.sum(target_sims[i])
    back = replaced_sim * (neighbor_embeddings - embedding)
    embedding -= back
    move = anchor_sims * (new_embeddings - embedding)
    embedding += move
    return embedding


class EmbeddingOptimizer:
    def __init__(self, neg_num=50, min_dist=0.1, temperature=0.15,
                 skip_opt=False, timeout_thresh=5.0):

        self.nce_opt_update_thresh = 5
        self.__local_move_thresh_w = 3  # 1.5 for high accuracy, 3 for high speed
        self.__neg_num = neg_num
        self.__temperature = temperature
        self.__a, self.__b = find_ab_params(1.0, min_dist)
        self.skip_opt = skip_opt
        self.skip_optimizer = SkipOptimizer(timeout_thresh) if skip_opt else None
        self._nce_grads = nce_loss_grad

        self.pre_num = 0
        self.num = 0
        self.std_num = 0

    def optimize_new_data_embedding(self, neighbor_sims, neighbor_embeddings, other_embeddings):
        initial_embedding = initial_embedding_with_weighted_mean(neighbor_sims, neighbor_embeddings)
        neg_embeddings = other_embeddings[random.sample(list(np.arange(other_embeddings.shape[0])), self.__neg_num)]
        res = scipy.optimize.minimize(nce_loss_single, initial_embedding, method="BFGS", jac=self._nce_grads,
                                      args=(neighbor_embeddings, neg_embeddings, self.__a, self.__b, self.__temperature),
                                      options={'gtol': 1e-5, 'disp': False, 'return_all': False, 'eps': 1e-10})
        optimized_e = res.x
        update_step = optimized_e - initial_embedding
        update_step[update_step > self.nce_opt_update_thresh] = 0
        update_step[update_step < -self.nce_opt_update_thresh] = 0
        initial_embedding += update_step
        return initial_embedding

    def update_old_data_embedding(self, new_data_num, total_embeddings, update_indices, knn_indices,
                                  knn_dists, corr_target_sims, anchor_positions, replaced_indices,
                                  replaced_raw_weights):
        old_embeddings = total_embeddings[:-new_data_num]
        if self.skip_opt:
            total_embeddings = self._pre_skip_opt(total_embeddings, knn_indices)

        local_move_mask, row_indices = self._cal_local_move_mask(total_embeddings, update_indices,
                                                                 anchor_positions, knn_indices)

        if self.skip_opt:
            update_indices, anchor_positions, replaced_indices, replaced_raw_weights, local_move_mask = \
                self._mid_skip_opt(update_indices, knn_indices, knn_dists, anchor_positions, replaced_indices,
                                   replaced_raw_weights, local_move_mask, row_indices)

        total_embeddings[update_indices] = self._update(update_indices, knn_indices[update_indices], total_embeddings,
                                                        corr_target_sims, anchor_positions,
                                                        replaced_indices, replaced_raw_weights, local_move_mask)

        if self.skip_opt:
            self._end_skip_opt()

        return total_embeddings

    def _pre_skip_opt(self, embeddings, knn_indices):
        assert self.skip_optimizer is not None
        timeouts_indices = self.skip_optimizer.get_timeouts_indices()
        if len(timeouts_indices) == 0:
            return embeddings

        neg_indices = random.sample(list(np.arange(embeddings.shape[0])), self.__neg_num)
        for i, item in enumerate(timeouts_indices):
            embeddings[item] = self._nce_optimize_step(embeddings[item], embeddings[knn_indices[item]],
                                                       embeddings[neg_indices])
        return embeddings

    def _mid_skip_opt(self, update_indices, knn_indices, knn_dists, anchor_positions, replaced_indices,
                      replaced_raw_weights, local_move_mask, row_indices):

        assert self.skip_optimizer is not None
        embedding_update_mask = self.skip_optimizer.skip(update_indices, knn_indices, knn_dists, anchor_positions,
                                                         local_move_mask, row_indices)

        update_indices = update_indices[embedding_update_mask]
        anchor_positions = anchor_positions[embedding_update_mask].astype(int)
        replaced_indices = replaced_indices[embedding_update_mask].astype(int)
        replaced_raw_weights = replaced_raw_weights[embedding_update_mask]
        local_move_mask = local_move_mask[embedding_update_mask]

        return update_indices, anchor_positions, replaced_indices, replaced_raw_weights, local_move_mask

    def _end_skip_opt(self):
        self.skip_optimizer.update_records()

    def _cal_local_move_mask(self, total_embeddings, update_indices, anchor_positions, knn_indices):
        update_num = len(update_indices)
        dist2neighbors = np.reshape(np.linalg.norm(np.reshape(total_embeddings[knn_indices[update_indices]] -
                                                              total_embeddings[update_indices][:, np.newaxis, :],
                                                              (-1, total_embeddings.shape[1])), axis=-1),
                                    (update_num, -1))

        tmp_indices = np.arange(update_num)
        dists2new_embeddings = dist2neighbors[tmp_indices, anchor_positions]
        dist2other_neighbors = (np.sum(dist2neighbors, axis=1) - dists2new_embeddings) / (knn_indices.shape[1] - 1)
        local_move_mask = dists2new_embeddings <= dist2other_neighbors * self.__local_move_thresh_w
        return local_move_mask, tmp_indices

    def update_all_skipped_data(self, embeddings, knn_indices):
        if not self.skip_opt:
            return
        update_indices = self.skip_optimizer.update_all_skipped_data()
        neg_indices = random.sample(list(np.arange(embeddings.shape[0])), self.__neg_num)

        for i, item in enumerate(update_indices):
            embeddings[item] = self._nce_optimize_step(embeddings[item], embeddings[knn_indices[item]],
                                                       embeddings[neg_indices])

        return embeddings

    def _update(self, update_indices, corr_knn_indices, total_embeddings, target_sims, anchor_position,
                replaced_neighbor_indices, replaced_sims, local_move_mask):
        new_embeddings = total_embeddings[-1]
        total_n_samples = total_embeddings.shape[0]
        neg_indices = None

        for i, item in enumerate(update_indices):
            if local_move_mask[i]:
                total_embeddings[item] = _local_move(total_embeddings[item], new_embeddings,
                                                     total_embeddings[replaced_neighbor_indices[i]], anchor_position, i,
                                                     replaced_sims[i], target_sims)
            else:
                if neg_indices is None:
                    neg_indices = random.sample(list(np.arange(total_n_samples)), self.__neg_num)
                total_embeddings[item] = self._nce_optimize_step(total_embeddings[item],
                                                                 total_embeddings[corr_knn_indices[i]],
                                                                 total_embeddings[neg_indices])

        return total_embeddings[update_indices]

    def _nce_optimize_step(self, optimize_embedding, positive_embeddings, neg_embeddings):
        res = scipy.optimize.minimize(nce_loss_single, optimize_embedding,
                                      method="BFGS", jac=self._nce_grads,
                                      args=(positive_embeddings, neg_embeddings, self.__a, self.__b, self.__temperature),
                                      options={'gtol': 1e-4, 'disp': False, 'return_all': False, 'eps': 1e-10})

        optimized_e = res.x
        update_step = optimized_e - optimize_embedding
        update_step[update_step > self.nce_opt_update_thresh] = 0
        update_step[update_step < -self.nce_opt_update_thresh] = 0
        optimize_embedding += update_step
        return optimize_embedding


class SkipOptimizer:
    def __init__(self, timeout_thresh=1.0):
        self.timeout_thresh = timeout_thresh
        self.delayed_meta = None
        self._oldest_time = 0

        self.timeout_meta_indices = []
        self._left_mask = None
        self.updated_data_indices = []
        self.skipped_data_indices = []

        self.thresh = 1.5  # 0

    def get_timeouts_indices(self):
        cur_time = time.time()
        if self.delayed_meta is None or cur_time - self._oldest_time < self.timeout_thresh:
            if self.delayed_meta is not None:
                self._left_mask = np.ones(self.delayed_meta.shape[0]).astype(bool)
            self.timeout_meta_indices = []
            return []

        self._left_mask = np.ones(self.delayed_meta.shape[0]).astype(bool)
        self.timeout_meta_indices = np.where(cur_time - self.delayed_meta[:, 1] >= self.timeout_thresh)[0]
        self._left_mask[self.timeout_meta_indices] = False
        return self.delayed_meta[self.timeout_meta_indices][:, 0]

    def skip(self, update_indices, knn_indices, knn_dists, anchor_positions, local_move_mask, row_indices):
        optimize_mask = self._cal_opt_mask(update_indices, knn_indices, knn_dists, anchor_positions, row_indices)
        embedding_update_mask = (local_move_mask + optimize_mask) > 0
        self.updated_data_indices = update_indices[embedding_update_mask]
        self.skipped_data_indices = update_indices[~embedding_update_mask]
        return embedding_update_mask

    def _cal_opt_mask(self, update_indices, knn_indices, knn_dists, anchor_positions, row_indices):
        max_dists = knn_dists[update_indices, -1]
        neighbors_knn_dist = knn_dists[knn_indices[update_indices]]

        max_neighbor_knn_dist = np.max(neighbors_knn_dist, axis=-1)
        tmp_mask = np.ones_like(max_neighbor_knn_dist).astype(bool)
        tmp_mask[row_indices, anchor_positions] = False
        other_neighbor_knn_dist = np.reshape(max_neighbor_knn_dist[tmp_mask], (tmp_mask.shape[0], -1))
        lower_q, higher_q = np.quantile(other_neighbor_knn_dist, [0.25, 0.75], axis=1)

        int_r = higher_q - lower_q
        optimize_mask = max_dists < higher_q + self.thresh * int_r
        return optimize_mask

    def update_records(self):
        left_delayed_meta = []
        if self.delayed_meta is not None:
            _, position_1, _ = np.intersect1d(self.delayed_meta[:, 0], self.updated_data_indices, return_indices=True)
            self._left_mask[position_1] = False
            left_delayed_meta = self.delayed_meta[self._left_mask]

        cur_time = time.time()
        total_skipped_meta = None
        if len(left_delayed_meta) > 0:
            if len(left_delayed_meta.shape) < 2:
                left_delayed_meta = left_delayed_meta[np.newaxis, :]

            self._oldest_time = np.min(left_delayed_meta[:, 1])

            skipped_data_indices = np.setdiff1d(self.skipped_data_indices, left_delayed_meta[:, 0])
            skipped_data_meta = np.zeros(shape=(len(skipped_data_indices), 2), dtype=int)
            skipped_data_meta[:, 0] = skipped_data_indices
            skipped_data_meta[:, 1] = cur_time

            total_skipped_meta = np.concatenate([left_delayed_meta, skipped_data_meta], axis=0)
        elif len(self.skipped_data_indices) > 0:
            self._oldest_time = cur_time
            total_skipped_meta = np.zeros(shape=(len(self.skipped_data_indices), 2), dtype=int)
            total_skipped_meta[:, 0] = self.skipped_data_indices
            total_skipped_meta[:, 1] = cur_time

        self.delayed_meta = total_skipped_meta

    def update_all_skipped_data(self):
        if len(self.delayed_meta.shape) < 2:
            return [int(self.delayed_meta[0])]
        else:
            return self.delayed_meta[:, 0].astype(int)


class NNEmbedder:
    def __init__(self, device):
        self._model = None
        self._device = device

    def update_model(self, new_model):
        self._model = new_model.to(self._device)

    def embed(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float, device=self._device)
        with torch.no_grad():
            data_embeddings = self._model(data).cpu()

        return data_embeddings.numpy()
