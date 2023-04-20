import os.path

import numpy as np

from dataset.warppers import StreamingDatasetWrapper
from model.scdr.data_processor import DataProcessor, DataProcessorProcess
from model.scdr.dependencies.embedding_optimizer import NNEmbedder
from utils.queue_set import ModelUpdateQueueSet, DataProcessorQueue


class StreamDRBase:
    def __init__(self, n_neighbors, batch_size, global_pattern_change_thresh, model_update_queue_set, window_size=2000,
                 ckpt_path=None, device="cuda:0"):
        self.model_update_queue_set = model_update_queue_set
        self.device = device
        self.nn_embedder = NNEmbedder(device)
        self.data_processor = DataProcessor(n_neighbors, batch_size, model_update_queue_set, window_size,
                                            global_pattern_change_thresh, device)
        self.ckpt_path = ckpt_path if (ckpt_path is not None and os.path.exists(ckpt_path)) else None

        self.stream_dataset = StreamingDatasetWrapper(batch_size, n_neighbors, window_size, self.device)
        self.initial_data_buffer = None
        self.initial_label_buffer = None
        self.model_trained = False

    def fit_new_data(self, data, labels=None, end=False):
        other_time = 0
        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None, 0, False

            self.stream_dataset.add_new_data(data=self.initial_data_buffer, labels=self.initial_label_buffer)
            total_embeddings = self._initial_project_model()
            self.data_processor.init_stream_dataset(self.stream_dataset)
            self.data_processor.nn_embedder = self.nn_embedder
        else:
            if data is None:
                return None, 0, False
            self._listen_model_update()

            data_embeddings = self.nn_embedder.embed(data)
            total_embeddings, other_time = self.data_processor.process(data, data_embeddings, labels)

        return total_embeddings, other_time, True

    def _listen_model_update(self):
        if not self.model_update_queue_set.embedding_queue.empty():
            embeddings, infer_model, stream_dataset, total_data_idx = self.model_update_queue_set.embedding_queue.get()
            self.update_scdr(infer_model, embeddings, stream_dataset, total_data_idx)

    def get_final_embeddings(self):
        return self.data_processor.get_final_embeddings()

    def _caching_initial_data(self, data, labels):
        self.initial_data_buffer = data if self.initial_data_buffer is None \
            else np.concatenate([self.initial_data_buffer, data], axis=0)
        if labels is not None:
            self.initial_label_buffer = labels if self.initial_label_buffer is None \
                else np.concatenate([self.initial_label_buffer, labels], axis=0)

        return True

    def _initial_project_model(self):
        self.model_update_queue_set.training_data_queue.put(
            [self.stream_dataset, None, self.ckpt_path])
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.UPDATE)
        self.model_update_queue_set.INITIALIZING.value = True
        embeddings, model, stream_dataset, _ = self.model_update_queue_set.embedding_queue.get()
        self.stream_dataset = stream_dataset
        self.stream_dataset.update_unfitted_data_num(0)
        self.stream_dataset.add_new_data(embeddings=embeddings)
        self.nn_embedder.update_model(model)

        self.model_trained = True
        self.model_update_queue_set.INITIALIZING.value = False

        return embeddings

    def save_model(self):
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.SAVE)

    def update_scdr(self, newest_model, embeddings, stream_dataset, total_data_idx):
        return self.data_processor.update_scdr(newest_model, embeddings, stream_dataset, total_data_idx)

    def ending(self):
        return self.data_processor.ending(), []


class StreamDRParallel(StreamDRBase):
    def __init__(self, embedding_data_queue, n_neighbors, batch_size, global_pattern_change_thresh,
                 model_update_queue_set, ckpt_path=None, device="cuda:0", window_size=2000):
        StreamDRBase.__init__(self, n_neighbors, batch_size, global_pattern_change_thresh, model_update_queue_set,
                              window_size, ckpt_path, device)
        self.data_processor = DataProcessorProcess(embedding_data_queue, n_neighbors, batch_size, model_update_queue_set
                                                   , window_size, global_pattern_change_thresh, device)
        self._embedding_data_queue: DataProcessorQueue = embedding_data_queue

    def fit_new_data(self, data, labels=None, end=False):
        if not self.model_trained:

            if not self._caching_initial_data(data, labels):
                return None, 0, False, 0

            self.stream_dataset.add_new_data(data=self.initial_data_buffer, labels=self.initial_label_buffer)
            total_embeddings = self._initial_project_model()
            self.data_processor.init_stream_dataset(self.stream_dataset)

            self.data_processor.daemon = True
            self.data_processor.start()
            self._embedding_data_queue.put_res([total_embeddings, None, 0, 0])
            out_num = 0
        else:
            if data is None:
                self._embedding_data_queue.put([data, None, labels, end])
                return None, 0, False, 0

            data_embeddings = self.nn_embedder.embed(data)

            total_embeddings, newest_model, add_data_time, out_num = self._embedding_data_queue.get_res()

            total_embeddings = np.concatenate([total_embeddings, data_embeddings], axis=0)

            if newest_model is not None:
                self.nn_embedder.update_model(newest_model)
            self._embedding_data_queue.put([data, data_embeddings, labels, end])

        return total_embeddings, 0, True, out_num
