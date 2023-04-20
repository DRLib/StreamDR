import os
import numpy as np
from pynndescent import NNDescent
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from annoy import AnnoyIndex
from utils.logger import InfoLogger


def compute_accurate_knn(flattened_data, k, neighbors_cache_path=None, pairwise_cache_path=None, metric="euclidean",
                         include_self=False):
    cur_path = None
    if neighbors_cache_path is not None:
        cur_path = neighbors_cache_path.replace(".npy", "_ac.npy")

    if cur_path is not None and os.path.exists(cur_path):
        knn_indices, knn_distances = np.load(cur_path)
        InfoLogger.info("directly load accurate neighbor_graph from {}".format(cur_path))
    else:
        preload = flattened_data.shape[0] <= 30000

        pairwise_distance = get_pairwise_distance(flattened_data, metric, pairwise_cache_path, preload=preload)
        sorted_indices = np.argsort(pairwise_distance, axis=1)
        if include_self:
            knn_indices = sorted_indices[:, :k]
        else:
            knn_indices = sorted_indices[:, 1:k + 1]
        knn_distances = []
        for i in range(knn_indices.shape[0]):
            knn_distances.append(pairwise_distance[i, knn_indices[i]])
        knn_distances = np.array(knn_distances)
        if cur_path is not None:
            np.save(cur_path, [knn_indices, knn_distances])
            InfoLogger.info("successfully compute accurate neighbor_graph and save to {}".format(cur_path))
    return knn_indices, knn_distances


def compute_knn_graph(all_data, neighbors_cache_path, k, pairwise_cache_path,
                      metric="euclidean", max_candidates=60, accelerate=False, include_self=False):
    flattened_data = all_data.reshape((len(all_data), np.product(all_data.shape[1:])))
    if not accelerate:
        knn_indices, knn_distances = compute_accurate_knn(flattened_data, k, neighbors_cache_path, pairwise_cache_path,
                                                          include_self=include_self)
        return knn_indices, knn_distances

    if neighbors_cache_path is not None and os.path.exists(neighbors_cache_path):
        neighbor_graph = np.load(neighbors_cache_path)
        knn_indices, knn_distances = neighbor_graph
        InfoLogger.info("directly load approximate neighbor_graph from {}".format(neighbors_cache_path))
    else:
        n_trees = 5 + int(round((all_data.shape[0]) ** 0.5 / 20.0))
        n_iters = max(5, int(round(np.log2(all_data.shape[0]))))

        nnd = NNDescent(
            flattened_data,
            n_neighbors=k + 1,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=max_candidates,
            verbose=False
        )

        knn_indices, knn_distances = nnd.neighbor_graph
        knn_indices = knn_indices[:, 1:]
        knn_distances = knn_distances[:, 1:]

        if neighbors_cache_path is not None:
            np.save(neighbors_cache_path, [knn_indices, knn_distances])
        InfoLogger.info("successfully compute approximate neighbor_graph and save to {}".format(neighbors_cache_path))
    return knn_indices, knn_distances


def get_pairwise_distance(flattened_data, metric="euclidean", pairwise_distance_cache_path=None, preload=False):
    if pairwise_distance_cache_path is not None and preload and os.path.exists(pairwise_distance_cache_path):
        pairwise_distance = np.load(pairwise_distance_cache_path)
        InfoLogger.info("directly load pairwise distance from {}".format(pairwise_distance_cache_path))
    else:
        pairwise_distance = pairwise_distances(flattened_data, metric=metric, squared=False)
        pairwise_distance[pairwise_distance < 1e-12] = 0.0
        if preload and pairwise_distance_cache_path is not None:
            np.save(pairwise_distance_cache_path, pairwise_distance)
            InfoLogger.info(
                "successfully compute pairwise distance and save to {}".format(pairwise_distance_cache_path))
    return pairwise_distance


class StreamingANNSearchAnnoy:
    def __init__(self, beta=5, update_iter=500, automatic_beta=False):
        self._searcher = None
        self._beta = beta
        self._update_iter = update_iter
        self._fitted_num = 0
        self._opt_embedding_indices = np.array([], dtype=int)
        self._optimized_data = None
        self._infer_embedding_indices = np.array([], dtype=int)
        self._inferred_data = None
        self._automatic_beta = automatic_beta

    def search(self, k, pre_embeddings, pre_data, query_embeddings, query_data, unfitted_num, update=False):
        if update:
            self._build_annoy_index(pre_embeddings[-unfitted_num:])
        elif (pre_embeddings.shape[0] - self._fitted_num) >= self._update_iter:
            self._build_annoy_index(pre_embeddings)

        if not self._automatic_beta:
            new_k = self._beta * k
        else:
            new_k = int(0.15 * np.sqrt(pre_data.shape[0]) * k)

        query_num = query_data.shape[0]
        if query_num == 1:
            candidate_indices = self._searcher.get_nns_by_vector(query_embeddings.squeeze(), new_k)
            candidate_indices = np.array(candidate_indices, dtype=int)
            if not update:
                candidate_indices = np.union1d(candidate_indices,
                                               np.arange(self._fitted_num, pre_data.shape[0]).astype(int))

            candidate_data = pre_data[candidate_indices]

            dists = cdist(query_data, candidate_data).squeeze()
            sorted_indices = np.argsort(dists).astype(int)
            final_indices = candidate_indices[sorted_indices[:k]][np.newaxis, :]
            final_dists = dists[sorted_indices[:k]][np.newaxis, :]
            candidate_indices = [candidate_indices[sorted_indices]]
            dists = [dists[sorted_indices]]
        else:
            # ====================================for batch process=====================================
            final_indices = np.empty((query_num, k), dtype=int)
            final_dists = np.empty((query_num, k), dtype=float)
            final_candidate_indices = []
            dists = []
            unfitted_data_indices = np.arange(self._fitted_num, pre_data.shape[0]).astype(int)
            for i in range(query_num):
                candidate_indices = self._searcher.get_nns_by_vector(query_embeddings[i], new_k)
                candidate_indices = np.array(candidate_indices, dtype=int)

                if not update:
                    candidate_indices = np.union1d(candidate_indices, unfitted_data_indices)

                cur_dists = cdist(query_data[i][np.newaxis, :], pre_data[candidate_indices]).squeeze()
                sorted_indices = np.argsort(cur_dists)
                final_candidate_indices.append(candidate_indices[sorted_indices])
                dists.append(cur_dists[sorted_indices])
                final_indices[i] = candidate_indices[sorted_indices[:k]]
                final_dists[i] = cur_dists[sorted_indices[:k]]

            candidate_indices = final_candidate_indices
            # ====================================for batch process=====================================

        return final_indices, final_dists, candidate_indices, dists

    def _build_annoy_index(self, embeddings):
        if self._searcher is None:
            self._searcher = AnnoyIndex(embeddings.shape[1], 'euclidean')
        else:
            self._searcher.unbuild()

        for i in range(embeddings.shape[0]):
            self._searcher.add_item(i, embeddings[i])

        self._searcher.build(10)
        self._fitted_num = embeddings.shape[0]
        self._opt_embedding_indices = np.array([], dtype=int)
        self._optimized_data = None
        self._infer_embedding_indices = np.array([], dtype=int)
        self._inferred_data = None


def heapK(ary, nums, k):
    if nums <= k:
        return ary

    ks = ary[:k]
    build_heap(ks, k)

    for index in range(k, nums):
        ele = ary[index]
        if ks[0] > ele:
            ks[0] = ele
            downAdjust(ks, 0, k)

    return ks


def build_heap(ary_list, k):
    index = k // 2 - 1
    while index >= 0:
        downAdjust(ary_list, index, k)
        index -= 1


def downAdjust(ary_list, parent_index, k):
    tmp = ary_list[parent_index]
    child_index = 2 * parent_index + 1

    while child_index < k:
        if child_index + 1 < k and ary_list[child_index + 1] > ary_list[child_index]:
            child_index += 1

        if tmp >= ary_list[child_index]:
            break

        ary_list[parent_index] = ary_list[child_index]
        parent_index = child_index
        child_index = 2 * parent_index + 1

    ary_list[parent_index] = tmp


def find_k_minimums(data, k):
    nums = data.shape[0]
    return heapK(data, nums, k)
