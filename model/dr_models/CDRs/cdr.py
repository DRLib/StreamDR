import math
import random
from model.dr_models.CDRs.NCELoss import NTXent, torch_app_skewnorm_func, MixtureNTXent
from model.dr_models.CDRs.nx_cdr import NxCDRModel
import torch
from model.dr_models.CDRs.stability_loss import cal_rank_relation_loss, cal_position_relation_loss, cal_shape_loss


class CDRModel(NxCDRModel):
    def __init__(self, cfg, device='cuda'):
        NxCDRModel.__init__(self, cfg, device)
        self.ratio = math.exp(1 / self.temperature) / torch.max(torch_app_skewnorm_func(torch.linspace(0, 1, 1000), 1))
        self.cfg = cfg
        self.a = torch.tensor(-40)
        self.loc = torch.tensor(0.11)
        self.lower_thresh = torch.tensor(0.001)
        self.scale = torch.tensor(0.13)
        self.alpha = torch.tensor(6)
        self.separation_begin_ratio = 0.2
        self.steady_begin_ratio = 0.9
        self.separate_epoch = int(self.epoch_num * self.separation_begin_ratio)
        self.steady_epoch = int(self.epoch_num * self.steady_begin_ratio)
        self.pre_epochs = self.epoch_num

    def update_separation_rate(self, sepa_rate, steady_rate):
        self.separation_begin_ratio = sepa_rate
        self.steady_begin_ratio = steady_rate

    def update_separate_period(self, epoch_num):
        self.separate_epoch = int(epoch_num * self.separation_begin_ratio) + self.pre_epochs
        self.steady_epoch = int(epoch_num * self.steady_begin_ratio) + self.pre_epochs
        self.pre_epochs += epoch_num

    def preprocess(self):
        self.build_model()
        self.criterion = NTXent.apply

    def _post_loss(self, logits, x_embeddings, epoch, *args):
        if self.separate_epoch <= epoch <= self.steady_epoch:
            epoch_ratio = torch.tensor((epoch - self.separate_epoch) / (self.steady_epoch - self.separate_epoch))
            cur_lower_thresh = 0.001 + (self.lower_thresh - 0.001) * epoch_ratio
            loss = MixtureNTXent.apply(logits, self.temperature, self.alpha, self.a, self.loc,
                                       cur_lower_thresh, self.scale)
        else:
            loss = self.criterion(logits, self.temperature)

        return loss


class IncrementalCDR(CDRModel):

    def __init__(self, cfg, device='cuda', neg_num=None):
        CDRModel.__init__(self, cfg, device)
        self.neg_num = None
        self.update_neg_num(neg_num)
        self.__nce_old_weight = 0.5
        self.__preserve_weight = cfg.method_params.preserve_weight

        self.preserve_rank = True
        self.preserve_pos = True
        self.preserve_shape = True
        self.rank_weight = 1.0
        self.pos_weight = 2.0
        self.shape_weight = 2.0
        self._rep_cluster_indices = None
        self._rep_exclude_indices = None

    def update_neg_num(self, new_neg_num):
        if self.neg_num is not None:
            self.neg_num = min(new_neg_num, self.batch_size)
            self.correlated_mask = (1 - torch.eye(self.neg_num)).type(torch.bool)

    def update_rep_data_info(self, cluster_indices):
        self._rep_cluster_indices = cluster_indices

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        epoch = args[0]
        is_incremental_learning = args[1]
        novel_logits = self.batch_logits(x_embeddings, x_sim_embeddings, *args)
        nce_loss_new = self._post_loss(novel_logits, None, epoch, None, *args)

        if not is_incremental_learning:
            return nce_loss_new

        rep_embeddings, pre_rep_embeddings = args[2], args[3]
        pre_rep_neighbors_embeddings, rep_neighbor_embeddings = args[4], args[5]
        no_change_indices = args[6]
        steady_weights, neighbor_steady_weights = args[7], args[8]
        batch_idx = args[9]

        old_logits = self.cal_old_logits(x_embeddings, x_sim_embeddings, rep_embeddings, novel_logits)
        nce_loss_old = self._post_loss(old_logits, None, epoch, None, *args)

        sim_learn_loss = nce_loss_new + self.__nce_old_weight * nce_loss_old

        cluster_indices = self._rep_cluster_indices[batch_idx]
        if pre_rep_neighbors_embeddings is None:
            sim_preserve_loss = 0
        else:
            sim_preserve_loss = self._cal_preserve_loss(rep_embeddings[no_change_indices], rep_neighbor_embeddings,
                                                        pre_rep_embeddings[no_change_indices], pre_rep_neighbors_embeddings,
                                                        steady_weights, neighbor_steady_weights)

        quality_constraint_loss = sim_learn_loss + self.__preserve_weight * sim_preserve_loss

        rank_loss = torch.tensor(0)
        position_loss = torch.tensor(0)
        shape_loss = torch.tensor(0)
        cluster_num = cluster_indices.shape[0]
        c_indices = [item[random.randint(0, len(item) - 1)] for item in cluster_indices]

        if self.preserve_rank and cluster_num > 1:
            rank_loss = cal_rank_relation_loss(rep_embeddings[c_indices], pre_rep_embeddings[c_indices])

        if self.preserve_pos and cluster_num > 1:
            position_loss = cal_position_relation_loss(rep_embeddings[c_indices], pre_rep_embeddings[c_indices])

        if self.preserve_shape and len(no_change_indices) > 0:
            shape_loss = cal_shape_loss(rep_embeddings[no_change_indices], rep_neighbor_embeddings,
                                        pre_rep_embeddings[no_change_indices], pre_rep_neighbors_embeddings,
                                        steady_weights, neighbor_steady_weights)

        w_rank_relation_loss = self.rank_weight * rank_loss
        w_position_relation_loss = self.pos_weight * position_loss
        w_shape_loss = self.shape_weight * shape_loss

        stability_constraint_loss = w_rank_relation_loss + w_position_relation_loss + w_shape_loss
        incremental_learning_loss = quality_constraint_loss + stability_constraint_loss

        return incremental_learning_loss

    def cal_old_logits(self, x_embeddings, x_sim_embeddings, rep_old_embeddings, novel_logits):
        pos_similarities = torch.clone(novel_logits[:, 0]).unsqueeze(1)

        rep_old_embeddings_matrix = rep_old_embeddings.unsqueeze(0).repeat(x_embeddings.shape[0] * 2, 1, 1)
        x_and_x_sim_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
        x_embeddings_matrix = x_and_x_sim_embeddings.unsqueeze(1).repeat(1, rep_old_embeddings.shape[0], 1)
        rep_old_negatives = self.similarity_func(rep_old_embeddings_matrix, x_embeddings_matrix, self.min_dist)[0]

        old_logits = torch.cat([pos_similarities, rep_old_negatives], dim=1)
        return old_logits

    def batch_logits(self, x_embeddings, x_sim_embeddings, *args):
        is_incremental_learning = args[1]
        if not is_incremental_learning or self.neg_num is None or self.neg_num == 2 * self.batch_size:
            logits = super().batch_logits(x_embeddings, x_sim_embeddings, *args)
        else:
            cur_batch_size = x_embeddings.shape[0]
            cur_available_neg_num = 2 * cur_batch_size - 2

            if cur_available_neg_num < self.neg_num:
                neg_num = cur_available_neg_num
                if self.partial_corr_mask is None:
                    self.partial_corr_mask = (1 - torch.eye(neg_num)).type(torch.bool)

                corr_mask = self.partial_corr_mask
            else:
                neg_num = self.neg_num
                corr_mask = self.correlated_mask

            queries = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
            queries_matrix = queries[:neg_num + 1].unsqueeze(0).repeat(2 * cur_batch_size, 1, 1)
            positives = torch.cat([x_sim_embeddings, x_embeddings], dim=0)
            negatives_p1 = torch.cat([queries_matrix[:neg_num, :neg_num][corr_mask]
                                     .view(neg_num, neg_num - 1, -1),
                                      queries_matrix[:neg_num, neg_num].unsqueeze(1)], dim=1)

            negatives_p2 = torch.cat([queries_matrix[cur_batch_size:cur_batch_size + neg_num, :neg_num]
                                      [corr_mask].view(neg_num, neg_num - 1, -1),
                                      queries_matrix[cur_batch_size:cur_batch_size + neg_num, neg_num]
                                     .unsqueeze(1)], dim=1)

            negatives = torch.cat([negatives_p1, queries_matrix[neg_num:cur_batch_size, :neg_num],
                                   negatives_p2, queries_matrix[cur_batch_size + neg_num:, :neg_num]], dim=0)

            pos_similarities = self.similarity_func(queries, positives, self.min_dist)[0].unsqueeze(1)
            neg_similarities = self.similarity_func(queries_matrix[:, :neg_num], negatives, self.min_dist)[0]

            logits = torch.cat((pos_similarities, neg_similarities), dim=1)
        return logits

    def _cal_preserve_loss(self, rep_embeddings, rep_neighbors_embeddings, pre_rep_embeddings, pre_rep_neighbors_embeddings,
                           steady_weights=None, neighbor_steady_weights=None):
        sim = self.similarity_func(rep_embeddings, rep_neighbors_embeddings)[0]
        pre_sims = self.similarity_func(pre_rep_embeddings, pre_rep_neighbors_embeddings)[0]
        sim_change = sim - pre_sims
        if steady_weights is not None and neighbor_steady_weights is not None:
            sim_change *= (steady_weights + neighbor_steady_weights) / 2
        return torch.mean(torch.square(sim_change))
