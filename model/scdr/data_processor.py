import time
from multiprocessing import Process

import numpy as np
import torch

from model.scdr.dependencies.embedding_optimizer import EmbeddingOptimizer
from model.scdr.dependencies.scdr_utils import EmbeddingQualitySupervisor
from utils.nn_utils import StreamingANNSearchAnnoy, compute_knn_graph
from utils.queue_set import ModelUpdateQueueSet, DataProcessorQueue

OPTIMIZE_NEW_DATA_EMBEDDING = False
OPTIMIZE_NEIGHBORS = True


class DataProcessor:
    def __init__(self, n_neighbors, batch_size, model_update_queue_set, window_size, global_pattern_change_thresh, device="cuda:0"):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.model_update_queue_set = model_update_queue_set
        self.device = device
        self._window_size = window_size
        self.nn_embedder = None

        self.data_num_when_update = 0

        self._do_model_update = True
        self.skip_opt = True
        self.model_update_intervals = 6000
        self.model_update_num_thresh = 30
        self.global_change_num_thresh = global_pattern_change_thresh
        self.opt_neg_num = 50
        self._knn_update_iter = 1000
        self.knn_searcher_approx = StreamingANNSearchAnnoy()

        self.stream_dataset = None
        self.embedding_quality_supervisor = None
        self.embedding_optimizer = None

        self.initial_data_buffer = None
        self.initial_label_buffer = None

        self.fitted_data_num = 0
        self.current_model_unfitted_num = 0
        self._need_replace_model = False
        self._update_after_replace_signal = False
        self._model_just_replaced = False
        self._model_is_updating = False
        self._model_update_delayed = False
        self._last_update_meta = None
        self._total_processed_data_num = 0
        self._out_since_last_send_update = 0
        self._skipped_slide_num = []
        self._cached_candidate_indices = None
        self._cached_candidate_dists = None
        self._cached_candidate_idx = None
        self._is_new_manifold = None
        self._candidate_num = 0
        self._process_num = 0
        self.update_count = 0

    def init_stream_dataset(self, stream_dataset):
        self.stream_dataset = stream_dataset
        self.stream_dataset.initial_data_num = self.stream_dataset.get_n_samples()
        self._total_processed_data_num = self.stream_dataset.get_n_samples()
        self.fitted_data_num = self.stream_dataset.get_n_samples()
        self._candidate_num = (self.knn_searcher_approx._beta - 1) * self.n_neighbors
        self._cached_candidate_indices = np.ones((self.fitted_data_num, self._candidate_num), dtype=int) * -1
        self._cached_candidate_dists = np.ones((self.fitted_data_num, self._candidate_num), dtype=int) * -1
        self._cached_candidate_idx = [1] * len(self._cached_candidate_indices)
        self._is_new_manifold = [False] * self.fitted_data_num
        self.initial_embedding_optimizer_and_quality_supervisor()

    def process(self, data, data_embeddings, labels=None):
        self._process_num += 1
        self._total_processed_data_num += data.shape[0]
        pre_n_samples = self.stream_dataset.get_n_samples()
        pre_embeddings = self.stream_dataset.get_total_embeddings()

        update = False
        if self._model_just_replaced:
            update = True
            self._model_just_replaced = False

        knn_indices, knn_dists, candidate_indices, candidate_dists = \
            self.knn_searcher_approx.search(self.n_neighbors, pre_embeddings,
                                            self.stream_dataset.get_total_data(),
                                            data_embeddings, data, self.current_model_unfitted_num, update)

        self._cached_candidate_idx.append(0)
        self._cached_candidate_indices = \
            np.concatenate([self._cached_candidate_indices,
                            candidate_indices[0][np.newaxis,
                            self.n_neighbors:self.n_neighbors + self._candidate_num]], axis=0)

        self._cached_candidate_dists = \
            np.concatenate([self._cached_candidate_dists,
                            candidate_dists[0][np.newaxis,
                            self.n_neighbors:self.n_neighbors + self._candidate_num]], axis=0)

        self.stream_dataset.add_new_data(data, None, labels, knn_indices, knn_dists)

        self.stream_dataset.update_knn_graph(pre_n_samples, data, None, candidate_indices, candidate_dists,
                                             update_similarity=False, symmetric=False)

        if self._process_num % self._knn_update_iter == 0:
            acc_knn_indices, acc_knn_dists = compute_knn_graph(self.stream_dataset.get_total_data(), None,
                                                               self.n_neighbors, None)
            self.stream_dataset._knn_manager.update_knn_graph(acc_knn_indices, acc_knn_dists)

        if update:
            fit_data = [self.stream_dataset.get_knn_indices(), self.stream_dataset.get_knn_dists(),
                        self.fitted_data_num]
        else:
            fit_data = None

        p_need_optimize, manifold_change, need_replace_model, need_update_model = \
            self.embedding_quality_supervisor.pattern_change_detection(knn_indices, knn_dists, fit_data)

        self.stream_dataset.add_new_data(embeddings=data_embeddings)

        if self._do_model_update and (need_update_model or self._model_update_delayed):
            self._send_update_signal()

        need_replace_model = need_replace_model and self._last_update_meta is not None
        if self._do_model_update and (need_replace_model or self._need_replace_model):
            if self.model_update_queue_set.training_data_queue.empty() or self._update_after_replace_signal:

                self._replace_model()
                need_replace_model = True
            else:
                self._need_replace_model = True
                need_replace_model = False

        if not need_replace_model:
            neighbor_changed_indices, replaced_raw_weights, replaced_indices, anchor_positions = \
                self.stream_dataset.get_pre_neighbor_changed_info()
            if len(neighbor_changed_indices) > 0:
                optimized_embeddings = self.embedding_optimizer.update_old_data_embedding(
                    data.shape[0], self.stream_dataset.get_total_embeddings(), neighbor_changed_indices,
                    self.stream_dataset.get_knn_indices(), self.stream_dataset.get_knn_dists(),
                    self.stream_dataset.raw_knn_weights[neighbor_changed_indices], anchor_positions,
                    replaced_indices, replaced_raw_weights)
                self.stream_dataset.update_embeddings(optimized_embeddings)

        ret = self.stream_dataset.get_total_embeddings()
        return ret, 0

    def _replace_model(self):
        newest_model, embeddings, total_data_idx = self._last_update_meta
        self.nn_embedder.update_model(newest_model)

        new_data_embeddings = self.embed_updating_collected_data(total_data_idx)
        total_embeddings = np.concatenate([embeddings, new_data_embeddings], axis=0)

        self.stream_dataset.update_embeddings(total_embeddings)
        self.stream_dataset.update_unfitted_data_num(new_data_embeddings.shape[0])

        self._model_embeddings = total_embeddings
        self._model_just_replaced = True
        self.current_model_unfitted_num = new_data_embeddings.shape[0]
        self._last_update_meta = None
        self._need_replace_model = False
        self._update_after_replace_signal = False

    def _send_update_signal(self):
        if self._model_is_updating:
            self._model_update_delayed = True
            return

        pre_fitted_num = self.fitted_data_num
        self.fitted_data_num = self.stream_dataset.get_n_samples()

        while not self.model_update_queue_set.training_data_queue.empty():
            self.model_update_queue_set.training_data_queue.get()

        self.model_update_queue_set.training_data_queue.put(
            [self.stream_dataset, None, pre_fitted_num,
             self.stream_dataset.get_n_samples() - pre_fitted_num,
             self._total_processed_data_num, self._out_since_last_send_update])
        self._model_is_updating = True
        self.model_update_queue_set.MODEL_UPDATING.value = 1
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.UPDATE)
        self.update_count += 1
        self._model_update_delayed = False
        self._out_since_last_send_update = 0

    def get_final_embeddings(self):
        embeddings = self.stream_dataset.get_total_embeddings()
        return embeddings

    def initial_embedding_optimizer_and_quality_supervisor(self):
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist = \
            self.stream_dataset.get_embedding_neighbor_mean_std_dist()

        e_thresh = pre_neighbor_embedding_m_dist + 3 * pre_neighbor_embedding_s_dist
        self.embedding_quality_supervisor = EmbeddingQualitySupervisor(self.global_change_num_thresh,
                                                                       self.model_update_num_thresh,
                                                                       e_thresh)
        self.embedding_quality_supervisor._lof.opt_fit(self.stream_dataset.get_knn_indices(),
                                                       self.stream_dataset.get_knn_dists(),
                                                       self.stream_dataset.get_n_samples())
        self.embedding_quality_supervisor.update_model_update_time(time.time())
        self.embedding_optimizer = EmbeddingOptimizer(neg_num=self.opt_neg_num, skip_opt=self.skip_opt)

    def infer_embeddings(self, data):
        return self.nn_embedder.embed(data)

    def embed_updating_collected_data(self, data_num_when_update):
        data = self.stream_dataset.get_total_data()[-(self._total_processed_data_num - data_num_when_update):]
        return self.infer_embeddings(data)

    def update_scdr(self, newest_model, embeddings, stream_dataset, total_data_idx):
        pre_embeddings = self.stream_dataset.get_total_embeddings()
        self.stream_dataset.update_previous_info(embeddings.shape[0], stream_dataset,
                                                 self._out_since_last_send_update,
                                                 self._skipped_slide_num[0] if len(self._skipped_slide_num) > 0 else 0)
        self._last_update_meta = [newest_model, embeddings, total_data_idx]
        self.stream_dataset.update_embeddings(pre_embeddings)
        self._skipped_slide_num = []
        self._model_is_updating = False
        if self._need_replace_model:
            self._update_after_replace_signal = True

        return pre_embeddings

    def update_thresholds(self):
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist = \
            self.stream_dataset.get_embedding_neighbor_mean_std_dist()

        if self.embedding_quality_supervisor is not None:
            e_thresh = pre_neighbor_embedding_m_dist + 3 * pre_neighbor_embedding_s_dist
            self.embedding_quality_supervisor.update_threshes(e_thresh)

    def ending(self):
        return ""


class DataProcessorProcess(DataProcessor, Process):
    def __init__(self, embedding_data_queue, n_neighbors, batch_size,
                 model_update_queue_set, window_size, global_pattern_change_thresh, device="cuda:0"):
        self.name = "data update process"
        DataProcessor.__init__(self, n_neighbors, batch_size, model_update_queue_set, window_size,
                               global_pattern_change_thresh, device)
        Process.__init__(self, name=self.name)
        self._embedding_data_queue: DataProcessorQueue = embedding_data_queue
        self._newest_model = None
        self.sub_time = 0
        self.dataset_slide_time = 0
        self.last_time = 0

    def run(self) -> None:
        while True:

            if not self.model_update_queue_set.embedding_queue.empty():
                embeddings, infer_model, stream_dataset, total_data_idx = \
                    self.model_update_queue_set.embedding_queue.get()
                self.update_scdr(infer_model, embeddings, stream_dataset, total_data_idx)

            data, data_embedding, label, is_end = self._embedding_data_queue.get()

            if is_end:
                break

            self._embedding_data_queue.processing()
            self._newest_model = None

            out_num = self.slide_window()

            total_embeddings, other_time = super().process(data, data_embedding, label)
            self._embedding_data_queue.put_res([total_embeddings, self._newest_model, other_time, out_num])
            self._embedding_data_queue.processed()

        self.ending()

    def _replace_model(self):
        newest_model, embeddings, total_data_idx = self._last_update_meta
        self._newest_model = newest_model

        data = self.stream_dataset.get_total_data()[-(self._total_processed_data_num - total_data_idx):]
        data = torch.tensor(data, dtype=torch.float)
        with torch.no_grad():
            tmp_model = newest_model
            new_data_embeddings = tmp_model(data).numpy()

        total_embeddings = np.concatenate([embeddings, new_data_embeddings], axis=0)[
                           -self.stream_dataset.get_n_samples():]
        self.stream_dataset.update_embeddings(total_embeddings)
        self.stream_dataset.update_unfitted_data_num(new_data_embeddings.shape[0])

        self._model_just_replaced = True
        self.current_model_unfitted_num = new_data_embeddings.shape[0]
        self._last_update_meta = None
        self._need_replace_model = False
        self._update_after_replace_signal = False

    def slide_window(self):
        out_num = max(0, self.stream_dataset.get_n_samples() - self._window_size)
        if out_num <= 0:
            return out_num

        self._cached_candidate_indices = self._cached_candidate_indices[out_num:]
        self._cached_candidate_indices -= out_num
        self._cached_candidate_dists = self._cached_candidate_dists[out_num:]
        self._cached_candidate_idx = self._cached_candidate_idx[out_num:]

        self.stream_dataset.slide_window(out_num, self._cached_candidate_indices, self._cached_candidate_dists,
                                         self._cached_candidate_idx)

        self.embedding_quality_supervisor.slide_window(out_num)
        self._is_new_manifold = self._is_new_manifold[out_num:]
        self.fitted_data_num = max(0, self.fitted_data_num - out_num)
        self._out_since_last_send_update += out_num

        if self._model_is_updating and len(self._skipped_slide_num) == 0:
            self._skipped_slide_num.append(out_num)

        return out_num

