import numpy as np
from scipy.spatial.distance import cdist
from utils.umap_utils import convert_distance_to_probability


def nce_loss_grad(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t=0.15):
    center_embedding = center_embedding[np.newaxis, :]
    neighbor_num = neighbor_embeddings.shape[0]

    # 1*(k+m)
    dist = cdist(center_embedding, np.concatenate([neighbor_embeddings, neg_embeddings], axis=0))
    # 1*(k+m)
    umap_sims = convert_distance_to_probability(dist, a, b)

    pos_dist = dist[:, :neighbor_num]
    neg_dist = dist[:, neighbor_num:]
    pos_umap_sims = umap_sims[:, :neighbor_num]
    neg_umap_sims = umap_sims[:, neighbor_num:]
    # k*(1+m)
    total_umap_sims = np.concatenate([pos_umap_sims.T, np.repeat(neg_umap_sims, neighbor_embeddings.shape[0], 0)],
                                     axis=1) / t
    total_umap_sims = np.exp(total_umap_sims)
    total_nce_sims = total_umap_sims / np.sum(total_umap_sims, axis=1)[:, np.newaxis]
    pos_grads = -np.sum(total_nce_sims[:, 1:], axis=1)[:, np.newaxis] * \
                (sim_grad(pos_dist, a, b).T * norm_grad(center_embedding, neighbor_embeddings, pos_dist.T)) / t
    neg_grads = np.sum((total_nce_sims[:, 1:][:, :, np.newaxis] *
                        (sim_grad(neg_dist, a, b).T * norm_grad(center_embedding, neg_embeddings,
                                                                neg_dist.T))[np.newaxis, :, :]) / t, axis=1)

    return np.mean(pos_grads + neg_grads, axis=0)


def sim_grad(dist, a, b):
    return -2 * a * b * ((1 + a * (dist ** (2 * b))) ** -2) * (dist ** (2 * b - 1))


def norm_grad(pred, gt, dist):
    return (dist ** -1) * (pred - gt)


# @jit
def nce_loss_single(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t=0.15):
    center_embedding = center_embedding[np.newaxis, :]
    neighbor_num = neighbor_embeddings.shape[0]

    # 1*(k+m)
    dist = cdist(center_embedding, np.concatenate([neighbor_embeddings, neg_embeddings], axis=0))
    # 1*(k+m)
    umap_sims = convert_distance_to_probability(dist, a, b)

    pos_umap_sims = umap_sims[:, :neighbor_num]
    neg_umap_sims = umap_sims[:, neighbor_num:]
    # k*(1+m)
    total_umap_sims = np.concatenate([pos_umap_sims.T, np.repeat(neg_umap_sims, neighbor_embeddings.shape[0], 0)],
                                     axis=1) / t
    total_umap_sims = np.exp(total_umap_sims)
    pos_nce_sims = (total_umap_sims[:, 0]) / np.sum(total_umap_sims, axis=1)

    loss = np.mean(-np.log(pos_nce_sims))
    return loss
