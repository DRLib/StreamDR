#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import numpy as np
from fast_soft_sort.pytorch_ops import soft_rank
from utils.umap_utils import find_ab_params, convert_distance_to_probability


MACHINE_EPSILON = np.finfo(np.double).eps

a = None
b = None
pre_min_dist = -1


def embedding_similarity(rep1, rep2, min_dist=0.1):
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    global a, b, pre_min_dist
    if a is None or pre_min_dist != min_dist:
        pre_min_dist = min_dist
        a, b = find_ab_params(1.0, min_dist)
    similarity_matrix = convert_distance_to_probability(pairwise_matrix, a, b)
    return similarity_matrix, pairwise_matrix


def _get_correlated_mask(batch_size):
    diag = np.eye(batch_size)
    l1 = np.eye(batch_size, batch_size, k=int(-batch_size / 2))
    l2 = np.eye(batch_size, batch_size, k=int(batch_size / 2))
    mask = torch.from_numpy((diag + l1 + l2))
    mask = (1 - mask).type(torch.bool)
    return mask


def compute_rank_correlation(att, grad_att):
    """
    Function that measures Spearman’s correlation coefficient between target logits and output logits:
    att: [n, m]
    grad_att: [n, m]
    """
    def _rank_correlation_(pred, target):
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return (pred * target).sum()

    att_rank = soft_rank(att, regularization_strength=0.5)
    grad_att_rank = soft_rank(grad_att, regularization_strength=0.5)

    correlation = _rank_correlation_(att_rank, grad_att_rank)
    return correlation
