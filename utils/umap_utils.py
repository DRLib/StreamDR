#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math
import time

import numba
import numpy
import numpy as np
import torch
import scipy
from numba import jit
from scipy.sparse import coo_matrix, csr_matrix
from scipy.optimize import curve_fit

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


def get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge

    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge

    Returns
    -------
    graph scipy.sparse.csr.csr_matrix
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of verticies in graph
    """
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    # CSR -> COO
    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights of how much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    # weights = np.ones_like(weights) / 2
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def construct_edge_dataset(
        X,
        graph_,
        n_epochs
):
    """
    Construct a tf.data.Dataset of edges, sampled by edge weight.
    """

    # get data from graph
    graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
        graph_, n_epochs
    )

    edges_to_exp, edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )

    # shuffle edges
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask]
    edges_from_exp = edges_from_exp[shuffle_mask]

    embedding_to_from_indices = np.array([edges_to_exp, edges_from_exp])
    embedding_to_from_indices_re = np.repeat(embedding_to_from_indices, 2, 1)
    np.random.shuffle(embedding_to_from_indices_re)
    embedding_to_from_data = X[embedding_to_from_indices[0, :]], X[embedding_to_from_indices[1, :]]

    return embedding_to_from_data, len(edges_to_exp), weight

smooth_t = 0
member_t = 0


def simple_fuzzy(updated_knn_indices, updated_knn_dists, local_connectivity=1.0, return_dists=False):
    n_neighbors = updated_knn_indices.shape[1]

    sigmas, rhos = simplified_smooth_knn_dist(updated_knn_dists, np.mean(updated_knn_dists, axis=1), float(n_neighbors),
                                              local_connectivity=float(local_connectivity))

    rows, cols, vals, dists = compute_membership_strengths(
        updated_knn_indices, updated_knn_dists, sigmas, rhos, return_dists
    )

    return sigmas, rhos,  vals.reshape(updated_knn_indices.shape)


def fuzzy_simplicial_set_partial(all_knn_indices, all_knn_dists, all_raw_knn_weights, update_indices,
                                 local_connectivity=1.0, apply_set_operations=True,
                                 return_dists=False, return_coo_results=True):
    updated_knn_indices = all_knn_indices[update_indices]
    updated_knn_dists = all_knn_dists[update_indices].astype(np.float32)
    n_neighbors = all_knn_indices.shape[1]

    sigmas, rhos = simplified_smooth_knn_dist(updated_knn_dists, np.mean(updated_knn_dists, axis=1), float(n_neighbors),
                                              local_connectivity=float(local_connectivity))

    rows, cols, vals, dists = compute_membership_strengths(
        updated_knn_indices, updated_knn_dists, sigmas, rhos, return_dists
    )

    all_raw_knn_weights[update_indices] = vals.reshape(updated_knn_indices.shape)
    if not return_coo_results:
        return None, sigmas, rhos, all_raw_knn_weights

    total_n_samples = all_knn_indices.shape[0]
    new_rows = np.ravel(np.repeat(np.expand_dims(np.arange(0, total_n_samples, 1), axis=1), axis=1, repeats=n_neighbors))
    new_cols = np.ravel(all_knn_indices)
    new_vals = np.ravel(all_raw_knn_weights)

    result = scipy.sparse.coo_matrix(
        (new_vals, (new_rows, new_cols)), shape=(total_n_samples, total_n_samples)
    )
    result.eliminate_zeros()

    if apply_set_operations:
        result = apply_set(result)

    result.eliminate_zeros()
    return result, sigmas, rhos, all_raw_knn_weights


def apply_set(result):
    transpose = result.transpose()
    result = (result + transpose) / 2
    return result


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    knn_indices=None,
    knn_dists=None,
    local_connectivity=1.0,
    apply_set_operations=True,
    return_dists=None,
):
    if knn_indices is None or knn_dists is None:
        pass

    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = simplified_smooth_knn_dist(
        knn_dists,
        np.mean(knn_dists, axis=1),
        float(n_neighbors),
        local_connectivity=float(local_connectivity),
    )

    rows, cols, vals, dists = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists
    )

    origin_knn_weights = vals.reshape(knn_indices.shape)

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()
        result = (result + transpose) / 2

    if return_dists is None:
        return result, sigmas, rhos, origin_knn_weights
    else:
        if return_dists:
            dmat = coo_matrix(
                (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
            )

            dists = dmat.maximum(dmat.transpose()).todok()
        else:
            dists = None

        return result, sigmas, rhos, origin_knn_weights, dists


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """
    Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        mid = _cal_sigma_origin(distances[i], n_iter, rho[i], target, lo, hi, mid)
        result[i] = mid

        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


@jit
def compute_membership_strengths(
    knn_indices, knn_dists, sigmas, rhos, return_dists=False
):
    """
    Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


def find_ab_params(spread, min_dist):
    """
    Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    """
     convert distance representation into probability,
        as a function of a, b params
    """
    return 1.0 / (1.0 + a * distances ** (2 * b))


def compute_cross_entropy(probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0):
    """
    Compute cross entropy between low and high probability
    """
    # cross entropy
    attraction_term = -probabilities_graph * torch.log(
        torch.clip(probabilities_distance, EPS, 1.0)
    )
    repellent_term = (
            -(1.0 - probabilities_graph)
            * torch.log(torch.clip(1.0 - probabilities_distance, EPS, 1.0))
            * repulsion_strength
    )

    # balance the expected losses between attraction and repel
    CE = attraction_term + repellent_term
    return attraction_term, repellent_term, CE


def compute_local_membership(knn_dist, knn_indices, local_connectivity=1):
    knn_dist = knn_dist.astype(np.float32)

    sigmas, rhos = simplified_smooth_knn_dist(
        knn_dist,
        float(knn_indices.shape[1]),
        local_connectivity=float(local_connectivity),
    )

    rows, cols, vals, dists = compute_membership_strengths(
        knn_indices+1, knn_dist, sigmas, rhos, False
    )
    return vals


@jit(nopython=True)
def simplified_smooth_knn_dist(distances, mean_distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """
    Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.
    """
    target = numpy.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    for i in numba.prange(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        ith_distances = distances[i]
        if ith_distances.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            rho[i] = ith_distances[index - 1]
        elif ith_distances.shape[0] > 0:
            rho[i] = np.max(ith_distances)

        mid = _cal_sigma_numba(distances[i], n_iter, rho[i], target, lo, hi, mid)

        result[i] = mid

        if result[i] < MIN_K_DIST_SCALE * mean_distances[i]:
            result[i] = MIN_K_DIST_SCALE * mean_distances[i]

    return result, rho


def _cal_sigma_origin(distances, n_iter, rho, target, lo, hi, mid):
    for n in range(n_iter):
        psum = 0.0
        for j in range(1, len(distances)):
            d = distances[j] - rho
            if d > 0:
                psum += np.exp(-(d / mid))
            else:
                psum += 1.0

        if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
            break
        if psum > target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == NPY_INFINITY:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0
    return mid


@jit
def _cal_sigma_numba(distances, n_iter, rho, target, lo, hi, mid):
    for n in numba.prange(n_iter):
        psum = 0.0
        for j in numba.prange(1, len(distances)):
            d = distances[j] - rho
            if d > 0:
                psum += math.exp(-(d / mid))
            else:
                psum += 1.0

        if abs(psum - target) < SMOOTH_K_TOLERANCE:
            break

        if psum > target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == NPY_INFINITY:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0
    return mid
