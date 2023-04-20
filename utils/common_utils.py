#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import time

import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from yaml import FullLoader

from dataset.datasets import TextDataset
from utils.constant_pool import ProjectSettings
from utils.logger import InfoLogger


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=FullLoader))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read(), Loader=FullLoader))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
DATE_ADJOIN_FORMAT = "%Y%m%d"
DATE_TIME_ADJOIN_FORMAT = "%Y%m%d_%Hh%Mm%Ss"


def time_stamp_to_date_time(time_stamp):
    time_array = time.localtime(time_stamp)
    otherStyleTime = time.strftime(DATE_TIME_FORMAT, time_array)
    return otherStyleTime


def time_stamp_to_date(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_FORMAT, time_array)


def time_stamp_to_date_adjoin(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_ADJOIN_FORMAT, time_array)


def time_stamp_to_date_time_adjoin(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_TIME_ADJOIN_FORMAT, time_array)


def date_time2timestamp(str_time):
    timeArray = time.strptime(str_time, DATE_TIME_FORMAT)
    timestamp = time.mktime(timeArray)
    return timestamp


