import copy
from torch.nn import Module
from utils.math_utils import _get_correlated_mask, embedding_similarity
from utils.umap_utils import find_ab_params
from model.dr_models.baseline_encoder import *


class NxCDRModel(Module):
    def __init__(self, cfg, device='cuda'):
        Module.__init__(self)
        self.device = device
        self.config = cfg
        self.encoder_name = "FC"
        self.pro_dim = 64

        self.input_dims = cfg.exp_params.input_dims
        self.latent_dim = cfg.exp_params.latent_dim
        self.temperature = torch.tensor(cfg.method_params.temperature)
        self.epoch_num = cfg.method_params.initializing_epochs
        self.batch_size = cfg.method_params.batch_size

        self.batch_num = 0
        self.max_neighbors = 0
        self.encoder = None
        self.pro_head = None
        self.criterion = None
        self.correlated_mask = _get_correlated_mask(2 * self.batch_size)
        self.partial_corr_mask = None

        self.min_dist = 0.1
        self._a, self._b = find_ab_params(1, self.min_dist)
        self.similarity_func = embedding_similarity

        self.cur_dist_matrix = None

        self.reduction = "mean"

    def reset_partial_corr_mask(self):
        self.partial_corr_mask = None

    def update_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        self.correlated_mask = _get_correlated_mask(2 * self.batch_size)

    def copy_network(self):
        c_encoder = copy.deepcopy(self.encoder)
        c_pro_header = copy.deepcopy(self.pro_head)
        return nn.Sequential(c_encoder, c_pro_header)

    def build_model(self):
        encoder, encoder_out_dims = get_encoder(self.encoder_name, self.input_dims)
        self.encoder = encoder
        self.pro_head = nn.Sequential(
            nn.Linear(encoder_out_dims, self.pro_dim),
            nn.ReLU(),
            nn.Linear(self.pro_dim, self.latent_dim)
        )

    def preprocess(self):
        self.build_model()
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)

    def encode(self, x):
        if x is None:
            return None, None
        reps = self.encoder(x)
        reps = reps.squeeze()

        embeddings = self.pro_head(reps)
        return reps, embeddings

    def forward(self, x, x_sim):
        x_reps, x_embeddings = self.encode(x)  # [N,C]
        x_sim_reps, x_sim_embeddings = self.encode(x_sim)  # [N,C]

        return x_reps, x_embeddings, x_sim_reps, x_sim_embeddings

    def acquire_latent_code(self, inputs):
        reps, embeddings = self.encode(inputs)
        return embeddings

    def acquire_representations(self, inputs):
        reps, embeddings = self.encode(inputs)
        return reps

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        epoch = args[0]
        logits = self.batch_logits(x_embeddings, x_sim_embeddings, *args)
        loss = self._post_loss(logits, x_embeddings, epoch, None, *args)
        return loss

    def _post_loss(self, logits, x_embeddings, epoch, item_weights, *args):
        labels = torch.zeros(logits.shape[0]).to(self.device).long()
        loss = self.criterion(logits / self.temperature, labels)

        return loss

    def batch_logits(self, x_embeddings, x_sim_embeddings, *args):
        all_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
        representations = all_embeddings.unsqueeze(0).repeat(all_embeddings.shape[0], 1, 1)
        similarity_matrix, pairwise_dist = self.similarity_func(representations.transpose(0, 1), representations,
                                                                self.min_dist)
        self.cur_dist_matrix = pairwise_dist

        cur_batch_size = x_embeddings.shape[0]
        if cur_batch_size != self.batch_size:
            if self.partial_corr_mask is None:
                self.partial_corr_mask = _get_correlated_mask(2 * cur_batch_size)
            correlated_mask = self.partial_corr_mask
        else:
            correlated_mask = self.correlated_mask

        l_pos = torch.diag(similarity_matrix, cur_batch_size)
        r_pos = torch.diag(similarity_matrix, -cur_batch_size)
        positives = torch.cat([l_pos, r_pos]).view(all_embeddings.shape[0], 1)
        negatives = similarity_matrix[correlated_mask].view(all_embeddings.shape[0], -1)

        logits = torch.cat((positives, negatives), dim=1)
        return logits

