from utils.math_utils import compute_rank_correlation
import torch


def cal_rank_relation_loss(rep_embeddings, pre_rep_embeddings):
    cluster_nums = rep_embeddings.shape[0]

    dists = torch.linalg.norm(rep_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) -
                              rep_embeddings.unsqueeze(1).repeat(1, cluster_nums, 1), dim=-1)
    pre_dists = torch.linalg.norm(pre_rep_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) -
                                  pre_rep_embeddings.unsqueeze(1).repeat(1, cluster_nums, 1), dim=-1)

    return -compute_rank_correlation(dists.cpu(), pre_dists.cpu())


def cal_position_relation_loss(cluster_center_embeddings: torch.Tensor,
                               pre_cluster_center_embeddings: torch.Tensor):
    cluster_nums = cluster_center_embeddings.shape[0]
    sims = torch.cosine_similarity(cluster_center_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) -
                                   cluster_center_embeddings.unsqueeze(1).repeat(1, cluster_nums, 1),
                                   pre_cluster_center_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) -
                                   pre_cluster_center_embeddings.unsqueeze(1).repeat(1, cluster_nums, 1), dim=-1)

    loss = -torch.mean(torch.square(sims))
    return loss


def cal_shape_loss(rep_embeddings, rep_neighbors_embeddings, pre_rep_embeddings, pre_rep_neighbors_embeddings,
                   steady_weights=None, neighbor_steady_weights=None):
    sims = torch.cosine_similarity(rep_embeddings - rep_neighbors_embeddings,
                                   pre_rep_embeddings - pre_rep_neighbors_embeddings, dim=-1)

    if steady_weights is not None and neighbor_steady_weights is not None:
        sims *= (steady_weights + neighbor_steady_weights) / 2

    return -torch.mean(torch.square(sims))

