#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from multiprocessing import Queue

StreamDR = "StreamDR"
STREAM_METHOD_LIST = [StreamDR]


class ProjectSettings:
    LABEL_COLORS = {0: 'steelblue', 1: 'chocolate', 2: 'yellowgreen', 3: 'indianred', 4: 'slateblue',
                    5: 'darkgoldenrod', 6: 'deeppink', 7: 'greenyellow', 8: 'olive', 9: 'cyan', 10: 'yellow',
                    11: 'purple'}


class ConfigInfo:
    MODEL_CONFIG_PATH = "./configs/{}.yaml"
    RESULT_SAVE_DIR = r"./results/{}/n{}_d{}"
    NEIGHBORS_CACHE_DIR = r"./data/knn/{}_k{}.npy"
    PAIRWISE_DISTANCE_DIR = r"./data/pairwise/{}.npy"
    DATASET_CACHE_DIR = r"./data/data_files/"
    CUSTOM_INDICES_DIR = r"./data/indices_seq"
