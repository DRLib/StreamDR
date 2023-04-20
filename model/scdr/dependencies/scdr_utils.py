import math
import random
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor


class KeyPointsGenerator:
    DBSCAN = "dbscan"

    @staticmethod
    def generate(data, key_rate, method=DBSCAN, cluster_inner_random=True, prob=None, min_num=0, cover_all=False,
                 **kwargs):
        if key_rate >= 1:
            return data, np.arange(0, data.shape[0], 1), None, None
        if method == KeyPointsGenerator.DBSCAN:
            return KeyPointsGenerator._generate_dbscan_based(data, key_rate, cluster_inner_random,
                                                             prob, min_num, cover_all, **kwargs)
        else:
            raise RuntimeError("Unsupported key data generating method. Please ensure that 'method' is one of random/"
                               "kmeans/dbscan.")

    @staticmethod
    def _generate_dbscan_based(data, key_rate, is_random=True, prob=None, min_num=0, batch_whole=False, **kwargs):
        n_samples = data.shape[0]
        eps = kwargs['eps']
        min_samples = kwargs["min_samples"]
        dbs = DBSCAN(eps, min_samples=min_samples)
        dbs.fit(data)
        key_data_num = max(int(n_samples * key_rate), min_num)

        if not batch_whole:
            return KeyPointsGenerator._sample_in_each_cluster(data, dbs.labels_, key_data_num, None, prob, is_random)
        else:
            return KeyPointsGenerator._cover_all_seq(data, dbs.labels_, key_data_num, min_samples * 3)

    @staticmethod
    def _sample_in_each_cluster(data, labels, key_data_num, centroids=None, prob=None, is_random=True):
        key_indices = []
        total_cluster_indices = []
        cluster_indices = []
        n_samples = len(labels)
        key_rate = key_data_num / len(np.argwhere(labels >= 0).squeeze())

        unique_labels = np.unique(labels)
        for item in unique_labels:
            if item < 0:
                continue
            cur_indices = np.argwhere(labels == item).squeeze()
            total_cluster_indices.append(cur_indices)

        if is_random:
            for item in np.unique(labels):
                if item < 0:
                    continue
                cur_indices = np.where(labels == item)[0]
                cur_key_num = int(math.ceil(len(cur_indices) * key_rate))
                cur_prob = None
                if prob is not None:
                    cur_prob = prob[cur_indices]
                    cur_prob /= np.sum(cur_prob)
                sampled_indices = np.random.choice(cur_indices, cur_key_num, p=cur_prob, replace=False)
                cluster_indices.append(np.arange(len(key_indices), len(key_indices) + len(sampled_indices)))
                key_indices.extend(sampled_indices)
        else:
            if prob is not None:
                raise RuntimeError("Custom sampling probability conflicts with distance-based sampling!")
            for i, item in enumerate(np.unique(labels)):
                if item < 0:
                    continue
                cur_indices = np.where(labels == item)[0]
                cur_center = centroids[i] if centroids is not None else np.mean(data[cur_indices], axis=-1)
                cur_dists = cdist(cur_center, data[cur_indices])
                sorted_indices = np.argsort(cur_dists)
                cur_key_num = int(len(cur_indices) * key_rate)
                sampled_indices = cur_indices[sorted_indices[np.linspace(0, len(cur_indices), cur_key_num, dtype=int)]]
                cluster_indices.append(np.arange(len(key_indices), len(key_indices) + len(sampled_indices)))
                key_indices.extend(sampled_indices)

        exclude_indices = []
        total_indices = np.arange(len(key_indices))
        for item in cluster_indices:
            exclude_indices.append(np.setdiff1d(total_indices, item))

        return data[key_indices], key_indices, cluster_indices, exclude_indices, total_cluster_indices

    @staticmethod
    def _cover_all_seq(data, labels, key_data_num, min_samples=20):
        n_samples = len(np.argwhere(labels >= 0).squeeze())

        key_indices = []
        cluster_indices = []
        exclude_indices = []
        total_cluster_indices = []
        batch_cluster_num = []
        unique_labels = np.unique(labels)
        valid_num = 0
        selected_labels = []
        for item in unique_labels:
            if item < 0:
                continue
            cur_indices = np.argwhere(labels == item).squeeze()
            if len(cur_indices) < min_samples:
                continue
            selected_labels.append(item)
            valid_num += len(cur_indices)
            np.random.shuffle(cur_indices)
            total_cluster_indices.append(cur_indices)
            batch_cluster_num.append(len(cur_indices))

        batch_num = int(np.floor(valid_num / key_data_num))
        batch_cluster_num = np.floor(np.array(batch_cluster_num) / batch_num).astype(int)

        for i in range(batch_num):
            cur_k_indices = []
            cur_c_indices = []
            idx = 0
            for item in selected_labels:
                num = batch_cluster_num[idx]
                end_idx = min((i + 1) * num, len(total_cluster_indices[idx]))
                select_indices = np.arange(i * num, end_idx)

                if end_idx < (i + 1) * num:
                    left = (i + 1) * num - end_idx
                    select_indices = np.append(select_indices, np.arange(left))

                cur_indices = total_cluster_indices[idx][select_indices]
                cur_c_indices.append(np.arange(len(cur_k_indices), len(cur_k_indices) + len(select_indices)))
                cur_k_indices.extend(cur_indices)
                idx += 1

            key_indices.append(cur_k_indices)
            cluster_indices.append(cur_c_indices)

        for i in range(batch_num):
            cur_e_indices = []
            total_indices = np.arange(len(key_indices[i]))
            for item in cluster_indices[i]:
                cur_e_indices.append(np.setdiff1d(total_indices, item))
            exclude_indices.append(cur_e_indices)

        return None, key_indices, cluster_indices, exclude_indices, total_cluster_indices


def dist_to_nearest_cluster_centroids(fitted_data, cluster_indices):
    centroids = []
    all_indices = []
    for item in cluster_indices:
        centroids.append(np.mean(fitted_data[item], axis=0))
        all_indices.extend(item)

    dist_matrix = cdist(fitted_data[all_indices], np.array(centroids))
    min_dist = np.min(dist_matrix, axis=1)
    return np.array(centroids), np.mean(min_dist), np.std(min_dist)


class ClusterRepDataSampler:
    def __init__(self, sample_rate=0, min_num=100, cover_all=False):
        self.__sample_rate = sample_rate
        self.__min_num = min_num
        self.__cover_all = cover_all

    def sample(self, fitted_embeddings, eps, min_samples, labels=None):
        _, rep_old_indices, cluster_indices, exclude_indices, total_cluster_indices = \
            KeyPointsGenerator.generate(fitted_embeddings, self.__sample_rate, method=KeyPointsGenerator.DBSCAN,
                                        min_num=self.__min_num, cover_all=self.__cover_all, eps=eps,
                                        min_samples=min_samples, labels=labels)

        if not self.__cover_all:
            rep_old_indices = [rep_old_indices]
            cluster_indices = [cluster_indices]
            exclude_indices = [exclude_indices]

        rep_batch_nums = len(rep_old_indices)

        return rep_batch_nums, rep_old_indices, cluster_indices, exclude_indices, total_cluster_indices


class EmbeddingQualitySupervisor:
    def __init__(self, global_change_num_thresh, model_update_thresh,
                 e_thresh=None, data_reduction="mean", embedding_reduction="mean"):
        self.__last_update_time = None
        self.__e_thresh = e_thresh
        self.__interval_seconds = 6000
        self.__manifold_change_num_thresh = global_change_num_thresh
        self.__model_update_thresh = model_update_thresh
        self.__new_manifold_data_num = 0
        self.__bad_embedding_data_num = 0
        self.__need_update_num = 0
        self._data_reduction = data_reduction
        self._embedding_reduction = embedding_reduction

        self._lof = OptLocalOutlierFactor(n_neighbors=10, novelty=True, metric="euclidean",
                                          contamination=0.1)

    def update_threshes(self, e_thresh):
        self._update_e_thresh(e_thresh)

    def _update_e_thresh(self, new_e_thresh):
        self.__e_thresh = new_e_thresh

    def update_model_update_time(self, update_time):
        self.__last_update_time = update_time

    def _judge_model_update(self, data_num):
        self.__need_update_num += data_num

        if self.__need_update_num > self.__model_update_thresh:
            self.__need_update_num = 0
            return True

        return False

    def _judge_model_replace(self):
        update = False
        if self.__last_update_time is not None and time.time() - self.__last_update_time >= self.__interval_seconds:
            update = True
        elif self.__new_manifold_data_num >= self.__manifold_change_num_thresh:
            update = True

        if update:
            self.__new_manifold_data_num = 0

        return update

    def pattern_change_detection(self, knn_indices, knn_dists, pre_data=None):
        manifold_change_list = []
        data_num = 0

        if pre_data is not None:
            self._lof.opt_fit(*pre_data)

        for i in range(knn_indices.shape[0]):
            label = self._lof.predict_novel(knn_indices[i], knn_dists[i])
            data_num += 1
            if label == -1:
                self.__new_manifold_data_num += 1
                manifold_change_list.append(True)
            else:
                manifold_change_list.append(False)

        return manifold_change_list, manifold_change_list, self._judge_model_replace(), \
            self._judge_model_update(data_num)

    def slide_window(self, out_num):
        self._lof.n_samples_fit_ = max(0, self._lof.n_samples_fit_ - out_num)


class OptLocalOutlierFactor(LocalOutlierFactor):
    def __init__(
            self,
            n_neighbors=20,
            *,
            algorithm="auto",
            leaf_size=30,
            metric="minkowski",
            p=2,
            metric_params=None,
            contamination="auto",
            novelty=False,
            n_jobs=None,
    ):
        LocalOutlierFactor.__init__(self, n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p,
                                    metric_params=metric_params, contamination=contamination, novelty=novelty,
                                    n_jobs=n_jobs)
        self._valid_rate = 0.5
        self.n_samples_fit_ = 0

    def predict_novel(self, nn_indices, nn_dists):
        valid_indices = np.where(nn_indices < self.n_samples_fit_)[0]
        if len(valid_indices) / self.n_neighbors_ < self._valid_rate:
            return -1

        scores = self.opt_score_samples(nn_indices[valid_indices], nn_dists[valid_indices]) - self.offset_

        return -1 if scores < 0 else 1

    def opt_score_samples(self, neighbors_indices_X, distances_X):
        dist_k = self._distances_fit_X_[neighbors_indices_X, self.n_neighbors_ - 1]
        reach_dist_array = np.maximum(distances_X, dist_k)

        X_lrd = 1.0 / (np.mean(reach_dist_array) + 1e-10)

        lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd

        return -np.mean(lrd_ratios_array)

    def opt_fit(self, knn_indices, knn_dists, fitted_num):
        self.n_samples_fit_ = fitted_num
        self._distances_fit_X_ = knn_dists
        self.n_neighbors_ = knn_indices.shape[1]
        self._lrd = self._local_reachability_density(
            knn_dists, knn_indices
        )
        lrd_ratios_array = (
            self._lrd[knn_indices] / self._lrd[:, np.newaxis]
        )

        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array[:fitted_num], axis=1)

        if self.contamination == "auto":
            # inliers score around -1 (the higher, the less abnormal).
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(
                self.negative_outlier_factor_, 100.0 * self.contamination
            )

        return self

