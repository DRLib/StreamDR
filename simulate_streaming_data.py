import argparse
import math
import os
import h5py
import numpy as np

from utils.constant_pool import ConfigInfo


def check_path_exists(t_path):
    if not os.path.exists(t_path):
        os.makedirs(t_path)


def no_drift(cls, cls_counts, labels, init_data_rate=0.3):
    init_data_indices = []
    stream_data_indices = []

    for i in range(len(cls)):
        cur_indices = np.where(labels == cls[i])[0]
        np.random.shuffle(cur_indices)
        cur_num = int(cls_counts[i] * init_data_rate)
        init_data_indices.extend(cur_indices[:cur_num])
        stream_data_indices.extend(cur_indices[cur_num:])

    np.random.shuffle(init_data_indices)
    np.random.shuffle(stream_data_indices)
    return init_data_indices, stream_data_indices, len(cls), 0


def partial_drift(cls, cls_counts, labels, init_manifold_rate=0.5, init_data_rate=0.4):

    cls_counts = []
    for jtem in cls:
        cls_counts.append(len(np.where(labels == jtem)[0]))

    init_manifold_num = int(len(cls) * init_manifold_rate)
    init_data_indices, init_left_indices = no_drift(cls[:init_manifold_num], cls_counts[:init_manifold_num], labels,
                                                    init_data_rate=init_data_rate)[:2]
    avg_num = len(init_left_indices) // (len(cls) - init_manifold_num)

    init_left_indices = np.array(init_left_indices, dtype=int)

    init_left_counts = []
    init_left_indices_per_cls = []
    for i in range(init_manifold_num):
        cur_indices = np.where(labels[init_left_indices] == cls[i])[0]
        init_left_counts.append(len(cur_indices))
        init_left_indices_per_cls.append(init_left_indices[cur_indices])

    avg_left_counts = np.array(init_left_counts) / (len(cls) - init_manifold_num)
    avg_left_counts = avg_left_counts.astype(int)

    stream_data_indices = []
    idx = 0

    for i in range(init_manifold_num, len(cls)):
        cur_indices = np.where(labels == cls[i])[0]
        cur_total = list(cur_indices)

        for j in range(init_manifold_num):
            cur_total.extend(init_left_indices_per_cls[j][idx*avg_left_counts[j]:(idx+1)*avg_left_counts[j]])

        np.random.shuffle(cur_total)
        stream_data_indices.extend(cur_total)
        idx += 1

    return init_data_indices, stream_data_indices, init_manifold_num, len(cls) - init_manifold_num


def full_drift(cls, cls_counts, labels, init_manifold_rate=0.3, shuffle_stream=False):
    init_data_indices = []
    stream_data_indices = []
    init_manifold_num = int(math.ceil(len(cls) * init_manifold_rate))
    ttt_indices = np.arange(len(cls))
    np.random.shuffle(ttt_indices)
    cls = cls[ttt_indices]

    for i in range(init_manifold_num):
        cur_indices = np.where(labels == cls[i])[0]
        init_data_indices.extend(cur_indices)

    for i in range(init_manifold_num, len(cls)):
        cur_indices = np.where(labels == cls[i])[0]
        stream_data_indices.extend(cur_indices)

    np.random.shuffle(init_data_indices)
    if shuffle_stream:
        np.random.shuffle(stream_data_indices)
    return init_data_indices, stream_data_indices, init_manifold_num, len(cls) - init_manifold_num


func_dict = {
    "ND": no_drift,
    "PD": partial_drift,
    "FD": full_drift
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="HAR", help="Separating multiple data sets by commas.")
    parser.add_argument("--dataset_dir", type=str, default=ConfigInfo.DATASET_CACHE_DIR)
    parser.add_argument("--indices_save_dir", type=str, default=ConfigInfo.CUSTOM_INDICES_DIR)
    parser.add_argument("--change_modes", type=list, default=["PD", "ND", "FD"])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_list = args.datasets.split(",")
    dataset_dir = args.dataset_dir
    save_dir = args.indices_save_dir
    situation_list = args.change_modes
    check_path_exists(save_dir)

    for item in dataset_list:
        data_name = item.split(".")[0]

        with h5py.File(os.path.join(dataset_dir, "{}.h5".format(item)), "r") as hf:
            x = np.array(hf['x'])
            y = np.array(hf['y'], dtype=int)
            unique_cls, cls_nums = np.unique(y, return_counts=True)

            for situation in situation_list:
                init_indices, stream_indices, init_cls_num, stream_new_cls_num = func_dict[situation](unique_cls,
                                                                                                      cls_nums, y)
                save_path = os.path.join(save_dir, "{}_{}.npy".format(data_name, situation))
                np.save(save_path, [init_indices, stream_indices])
                print("{}_{} -> Init Num: {} Stream Num: {} Init Cls: {} Stream New Cls: {}".format(
                    data_name, situation, len(init_indices), len(stream_indices), init_cls_num, stream_new_cls_num))
