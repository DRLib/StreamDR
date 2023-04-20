import argparse
import time
from multiprocessing import Queue

import numpy as np
import os

from dataset.streaming_data_mock import SimulatedStreamingData
from model.dr_models.CDRs.cdr import IncrementalCDR
from model.scdr.model_trainer import SCDRTrainerProcess
from experiments.streaming_experiment import StreamingExProcess
from utils.constant_pool import ConfigInfo, StreamDR
from utils.common_utils import get_config, time_stamp_to_date_time_adjoin
from utils.queue_set import ModelUpdateQueueSet

device = "cuda:0"
log_path = "logs/logs.txt"


def start(ex, recv_args, config, cfg_path, res_save_dir, device, log_path):
    res_save_dir = os.path.join(res_save_dir, config.exp_params.dataset,
                                time_stamp_to_date_time_adjoin(int(time.time())))
    ex.result_save_dir = res_save_dir
    if recv_args.method == StreamDR:
        assert isinstance(ex, StreamingExProcess)
        cdr_model = IncrementalCDR(config, device=device)

        model_update_queue_set = ModelUpdateQueueSet()

        model_trainer = SCDRTrainerProcess(model_update_queue_set, cdr_model, config.exp_params.dataset,
                                           cfg_path, config, res_save_dir, device=device, log_path=log_path)

        return ex.start_streamDR(model_update_queue_set, model_trainer, res_save_dir,
                                 config.exp_params.check_point_path)
    else:
        raise RuntimeError("Non-supported method! please ensure param 'method' is one of 'StreamDR'!")


def custom_indices_training(configs, custom_indices_path, recv_args, res_save_dir, cfg_path, device, log_path):
    custom_indices = np.load(custom_indices_path, allow_pickle=True)

    stream_data_queue_set = Queue()
    start_data_queue = Queue()
    data_generator = SimulatedStreamingData(configs.exp_params.dataset, configs.exp_params.stream_rate,
                                            stream_data_queue_set, start_data_queue, custom_indices)
    ex = StreamingExProcess(configs, custom_indices, res_save_dir, stream_data_queue_set, start_data_queue,
                            data_generator)

    data_generator.start()

    return start(ex, recv_args, configs, cfg_path, res_save_dir, device, log_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default=StreamDR)
    parser.add_argument("--change_mode", type=str, default="PD", choices=["ND", "PD", "FD"])
    parser.add_argument("--indices_dir", type=str, default=r"data/indices_seq")
    parser.add_argument("--save_dir", type=str, default=r"results/")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg_path = ConfigInfo.MODEL_CONFIG_PATH.format(args.method)
    cfg = get_config()
    cfg.merge_from_file(cfg_path)
    result_save_dir = os.path.join(args.save_dir, args.method)

    custom_indices_path = os.path.join(args.indices_dir, "{}_{}.npy".format(cfg.exp_params.dataset, args.change_mode))
    custom_indices_training(cfg, custom_indices_path, args, result_save_dir, cfg_path, device, log_path)
