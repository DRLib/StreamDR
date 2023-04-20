import os
import queue
import time
from copy import copy
from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist

from model.scdr.scdr_parallel import StreamDRBase, StreamDRParallel
from utils.queue_set import ModelUpdateQueueSet, DataProcessorQueue
from model.scdr.dependencies.experiment import position_vis
from utils.common_utils import time_stamp_to_date_time_adjoin
from utils.logger import InfoLogger
from utils.nn_utils import compute_knn_graph, get_pairwise_distance


plt.rcParams['animation.ffmpeg_path'] = r'D:\softwares\ffmpeg\bin\ffmpeg.exe'


def check_path_exist(t_path):
    if not os.path.exists(t_path):
        os.makedirs(t_path)


class StreamingEx:
    def __init__(self, cfg, seq_indices, result_save_dir, data_generator, log_path="log_streaming_con.txt",
                 do_eval=True):
        self.cfg = cfg
        self.seq_indices = seq_indices
        self._data_generator = data_generator
        self.dataset_name = cfg.exp_params.dataset
        self.streaming_mock = None
        self.vis_iter = cfg.exp_params.vis_iter
        self.save_embedding_iter = cfg.exp_params.save_iter

        self.log_path = log_path
        self.result_save_dir = result_save_dir
        self.model = None
        self.n_components = cfg.exp_params.latent_dim
        self.cur_time_step = 0
        self.metric_tool = None
        self.eval_k = 10
        self.vc_k = self.eval_k

        self._save_embeddings_for_eval = True
        self._key_time = 0
        self.pre_embedding = None
        self.cur_embedding = None
        self.embedding_dir = None
        self.img_dir = None

        self.initial_cached = False
        self.initial_data_num = 0
        self.initial_data = None
        self.initial_labels = None
        self.history_data = None
        self.history_label = None
        self._not_show_num = 0

        self.cls2idx = {}
        self.debug = True
        self.save_img = True
        self._make_animation = cfg.exp_params.make_animation
        self.save_embedding_npy = False
        self._pc_list = []
        self._embeddings_histories = []

        self._x_min = 1e7
        self._x_max = 1e-7
        self._y_min = 1e7
        self._y_max = 1e-7

    def _train_begin(self):
        check_path_exist(self.result_save_dir)
        self.log = open(os.path.join(self.result_save_dir, self.log_path), 'a')

    def stream_fitting(self):
        self._train_begin()
        self.processing()
        return self.train_end()

    def _cache_initial(self, stream_data, stream_labels):
        if len(stream_data.shape) == 1:
            stream_data = np.reshape(stream_data, (-1, len(stream_labels)))
        self.history_data = stream_data if self.history_data is None else np.concatenate(
            [self.history_data, stream_data], axis=0)
        if stream_labels is not None:
            self.history_label = stream_labels if self.history_label is None else np.concatenate(
                [self.history_label, stream_labels])

        if self.initial_cached:
            return True, stream_data, stream_labels
        else:
            self.initial_data = stream_data
            self.initial_labels = stream_labels
            self.initial_cached = True
            return True, stream_data, stream_labels

    def processing(self):
        self.embedding_dir = os.path.join(self.result_save_dir, "embeddings")
        check_path_exist(self.embedding_dir)
        self.img_dir = os.path.join(self.result_save_dir, "imgs")
        check_path_exist(self.img_dir)
        for i in range(self.streaming_mock.time_step_num):
            self.cur_time_step += 1
            output = "Start processing timestamp {}".format(i + 1)
            InfoLogger.info(output)
            stream_data, stream_labels, _ = self.streaming_mock.next_time_data()

            self._project_pipeline(stream_data, stream_labels)

    def _project_pipeline(self, stream_data, stream_labels):
        pre_labels = self.history_label
        cache_flag, stream_data, stream_labels = self._cache_initial(stream_data, stream_labels)
        if cache_flag:
            self.pre_embedding = self.cur_embedding

            ret_embeddings, key_time, embedding_updated, out_num = self.model.fit_new_data(stream_data, stream_labels)
            if not embedding_updated:
                self._not_show_num += stream_data.shape[0]
            else:
                self._not_show_num = 0

            if ret_embeddings is not None:
                cur_x_min, cur_y_min = np.min(ret_embeddings, axis=0)
                cur_x_max, cur_y_max = np.max(ret_embeddings, axis=0)
                self._x_min = min(self._x_min, cur_x_min)
                self._x_max = max(self._x_max, cur_x_max)
                self._y_min = min(self._y_min, cur_y_min)
                self._y_max = max(self._y_max, cur_y_max)
                self._embeddings_histories.append([len(self.history_label) - ret_embeddings.shape[0], ret_embeddings])

                self.cur_embedding = ret_embeddings
                self.save_embeddings_info(self.cur_embedding)

    def save_embeddings_info(self, cur_embeddings, custom_id=None, train_end=False):
        custom_id = self.cur_time_step if custom_id is None else custom_id
        if self.save_embedding_npy and (self.cur_time_step % self.save_embedding_iter == 0 or train_end):
            np.save(os.path.join(self.embedding_dir, "t_{}.npy".format(custom_id)), self.cur_embedding)

        if self.cur_time_step % self.vis_iter == 0 or train_end:
            img_save_path = os.path.join(self.img_dir, "t_{}.jpg".format(custom_id)) if self.save_img else None
            position_vis(self.history_label[-cur_embeddings.shape[0]:], img_save_path, cur_embeddings,
                         "T_{}".format(len(self._embeddings_histories)))

    def train_end(self):

        if isinstance(self.model, StreamDRBase):
            self.model.save_model()

        self.model.ending()

        if self._make_animation:
            self._make_embedding_video(self.result_save_dir)

        return [], 0, 0, 0

    def _make_embedding_video(self, save_dir):

        def _loose(d_min, d_max, rate=0.05):
            scale = d_max - d_min
            d_max += np.abs(scale * rate)
            d_min -= np.abs(scale * rate)
            return d_min, d_max

        l_x_min, l_x_max = _loose(self._x_min, self._x_max)
        l_y_min, l_y_max = _loose(self._y_min, self._y_max)

        fig, ax = plt.subplots()

        def update(idx):
            if idx % 100 == 0:
                print("frame", idx)

            start_idx, cur_embeddings = self._embeddings_histories[idx]

            ax.cla()
            ax.set(xlim=(l_x_min, l_x_max), ylim=(l_y_min, l_y_max))
            ax.axis('equal')
            ax.scatter(x=cur_embeddings[:, 0], y=cur_embeddings[:, 1],
                       c=list(self._data_generator.seq_color[start_idx:start_idx + cur_embeddings.shape[0]]), s=2)
            ax.set_title("Timestep: {}".format(int(idx)))

        ani = FuncAnimation(fig, update, frames=len(self._embeddings_histories), interval=15, blit=False)
        ani.save(os.path.join(save_dir, "embedding.mp4"), writer='ffmpeg', dpi=300)


class StreamingExProcess(StreamingEx, Process):
    def __init__(self, cfg, seq_indices, result_save_dir, stream_data_queue_set, start_data_queue, data_generator,
                 log_path="log_streaming_process.txt", do_eval=True):
        self.name = "data process processor"
        self.stream_data_queue_set = stream_data_queue_set
        self._start_data_queue = start_data_queue
        self.embedding_data_queue = None
        self.cdr_update_queue_set = None
        self.update_num = 0

        Process.__init__(self, name=self.name)
        StreamingEx.__init__(self, cfg, seq_indices, result_save_dir, data_generator, log_path, do_eval)

    def start_streamDR(self, model_update_queue_set, model_trainer, res_save_dir, ckpt_path):
        self.result_save_dir = res_save_dir
        self.cdr_update_queue_set = model_update_queue_set
        self.embedding_data_queue = DataProcessorQueue()
        self.model = StreamDRParallel(self.embedding_data_queue, self.cfg.method_params.n_neighbors,
                                      self.cfg.method_params.batch_size,
                                      self.cfg.method_params.global_pattern_change_thresh, model_update_queue_set,
                                      ckpt_path=ckpt_path, device=model_trainer.device,
                                      window_size=self.cfg.exp_params.window_size)
        model_trainer.daemon = True
        model_trainer.start()
        return self.stream_fitting()

    def processing(self):
        self.run()

    def _get_stream_data(self, accumulate=False):
        data, labels, is_stop = self.stream_data_queue_set.get()
        self.cur_time_step += 1

        total_data = np.array(data, dtype=float)
        total_labels = None if (len(labels) > 0 and labels[0] is None) is None else np.array(labels, dtype=int)

        output = "Get stream data timestamp {}".format(self.cur_time_step)
        InfoLogger.info(output)

        return is_stop, total_data, total_labels

    def _get_data_accumulate(self):
        total_data = self.stream_data_queue_set.data_queue.get()
        acc_num = 1
        while not self.stream_data_queue_set.data_queue.empty():
            acc_num += 1
            total_data.extend(self.stream_data_queue_set.data_queue.get())
        self.cur_time_step += acc_num
        return total_data

    def _get_data_single(self):
        data = self.stream_data_queue_set.get()
        self.cur_time_step += 1
        return data

    def run(self) -> None:
        self.embedding_dir = os.path.join(self.result_save_dir, "embeddings")
        check_path_exist(self.embedding_dir)
        self.img_dir = os.path.join(self.result_save_dir, "imgs")
        check_path_exist(self.img_dir)
        self._start_data_queue.put(True)
        while True:
            stream_end_flag, stream_data, stream_labels = self._get_stream_data(accumulate=False)
            if stream_end_flag:
                if isinstance(self.model, StreamDRBase):
                    self.model.fit_new_data(None, end=True)
                break

            self._project_pipeline(stream_data, stream_labels)

    def train_end(self):
        self.stream_data_queue_set.close()
        if self.cdr_update_queue_set is not None:
            self.cdr_update_queue_set.flag_queue.put(ModelUpdateQueueSet.STOP)

        return super().train_end()
