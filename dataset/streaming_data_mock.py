import time

import numpy as np

from dataset.datasets import load_local_h5_by_path
import os

from utils.constant_pool import ConfigInfo, ProjectSettings
from utils.common_utils import date_time2timestamp
from multiprocessing import Process, Queue


def resort_label(label_seq):
    unique_cls, show_seq = np.unique(label_seq, return_index=True)
    re_indices = np.argsort(show_seq)
    unique_cls = unique_cls[re_indices]

    new_label = np.ones_like(label_seq)
    for i, item in enumerate(unique_cls):
        indices = np.argwhere(label_seq == item).squeeze()
        new_label[indices] = i

    color_list = [ProjectSettings.LABEL_COLORS[i] for i in new_label]
    return new_label.astype(int), color_list


class SimulatedStreamingData(Process):
    def __init__(self, dataset_name, stream_rate, data_queue, start_data_queue, custom_seq=None, iter_time=0.01):
        self.name = "simulate streaming data process"
        Process.__init__(self, name=self.name)
        self._iter_time = iter_time
        self.data_queue = data_queue
        self._start_data_queue = start_data_queue
        self.dataset_name = dataset_name
        self.stream_rate = stream_rate
        self.data_index = 0

        self.data_file_path = os.path.join(ConfigInfo.DATASET_CACHE_DIR, dataset_name + ".h5")
        self.data, self.targets = load_local_h5_by_path(self.data_file_path, ['x', 'y'])
        self.n_samples = self.data.shape[0]
        initial_num = 0
        if custom_seq is None:
            self.custom_seq = np.arange(0, self.n_samples, 1)
            self.time_step_num = int(np.ceil(len(self.custom_seq) / self.stream_rate))
        else:
            if not isinstance(custom_seq[0], int):
                initial_indices, stream_indices = custom_seq
                self.custom_seq = np.concatenate([initial_indices, stream_indices])
                initial_num = len(initial_indices)
                self.time_step_num = int(np.ceil(len(stream_indices) / self.stream_rate)) + 1
            else:
                self.custom_seq = custom_seq
                self.time_step_num = int(np.ceil(len(self.custom_seq) / self.stream_rate))

        if self.targets is None:
            self.seq_label = None
            self.seq_color = None
            self.seq_stream_label = None
        else:
            self.seq_label, self.seq_color = resort_label(self.targets[self.custom_seq])
            self.seq_stream_label = self.seq_label

        stream_num = self.n_samples - initial_num

        self.data_num_list = np.ones(shape=(stream_num // stream_rate)) * stream_rate
        left = stream_num - np.sum(self.data_num_list)
        if left > 0:
            self.data_num_list = np.append(self.data_num_list, left)

        if initial_num > 0:
            self.data_num_list = np.append(initial_num, self.data_num_list)

        self.data_num_list = self.data_num_list.astype(int)

    def run(self) -> None:
        idx = 0
        self._start_data_queue.get(block=True)
        if self.data_index > 0:
            self.data_index = 0
        stop = False
        stream_data_num = 0
        while True:
            if stop:
                break
            stop = False
            cur_data_num = self.data_num_list[idx]
            cur_data = []
            cur_label = []
            for j in self.custom_seq[self.data_index:self.data_index + cur_data_num]:
                cur_data.append(self.data[j])
                cur_label.append(None if self.seq_stream_label is None else self.seq_stream_label[stream_data_num])
                stream_data_num += 1

            time.sleep(self._iter_time)

            self.data_index += cur_data_num
            idx += 1
            if idx >= len(self.data_num_list) or len(cur_data) == 0:
                stop = True
            self.data_queue.put([cur_data, cur_label, stop])

