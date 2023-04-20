#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import torch
from dataset.warppers import DataSetWrapper
from model.scdr.dependencies.experiment import Experiment
from model.dr_models.CDRs.cdr import CDRModel
from utils.constant_pool import *


class CDRsExperiments(Experiment):
    def __init__(self, cdr_model, dataset_name, configs, result_save_dir, config_path, shuffle=True, device='cuda',
                 log_path="logs.txt", multi=False):
        Experiment.__init__(self, cdr_model, dataset_name, configs, result_save_dir, config_path, shuffle, device,
                            log_path)
        self.model = cdr_model
        self.similar_num = 1
        self.cdr_dataset = None

        self.resume_epochs = 0
        self.model.to(self.device)
        self.steps = 0
        self.init_epoch = self.resume_epochs if self.resume_epochs > 0 else self.epoch_num
        self.separation_epochs = 0
        self.accelerate = False

        self.warmup_epochs = int(self.epoch_num * cdr_model.separation_begin_ratio)
        self.separation_epochs = int(self.epoch_num * cdr_model.steady_begin_ratio)

    def build_dataset(self, *args):
        # 数据加载器
        knn_cache_path = ConfigInfo.NEIGHBORS_CACHE_DIR.format(self.dataset_name, self.n_neighbors)
        pairwise_cache_path = ConfigInfo.PAIRWISE_DISTANCE_DIR.format(self.dataset_name)

        cdr_dataset = DataSetWrapper(self.similar_num, self.batch_size, self.n_neighbors)

        init_epoch = self.warmup_epochs

        self.train_loader, self.n_samples = cdr_dataset.get_data_loaders(
            init_epoch, self.dataset_name, ConfigInfo.DATASET_CACHE_DIR, self.n_neighbors, knn_cache_path,
            pairwise_cache_path, self.is_image)

        self.batch_num = cdr_dataset.batch_num
        self.model.batch_num = self.batch_num

    def forward(self, x, x_sim):
        return self.model.forward(x, x_sim)

    def acquire_latent_code(self, inputs):
        return self.model.acquire_latent_code(inputs)

    def train(self, launch_time_stamp=None, target_metric_val=-1):
        batch_print_inter, self.vis_inter, ckp_save_inter = self._train_begin(launch_time_stamp)
        embeddings = None
        net = self.model

        net.batch_num = self.batch_num
        training_loss_history = []
        test_loss_history = []

        for epoch in range(self.start_epoch, self.epoch_num):

            train_iterator, training_loss = self._before_epoch(epoch)
            for idx, data in enumerate(train_iterator):
                self.steps += 1

                train_data = self._step_prepare(data, epoch, train_iterator)
                loss = self._train_step(*train_data)
                training_loss.append(loss.detach().cpu().numpy())

            embeddings = self._after_epoch(ckp_save_inter, epoch + 1, training_loss, training_loss_history,
                                           self.vis_inter)

        self._train_end(test_loss_history, training_loss_history, embeddings)
        return embeddings

    def resume_train(self, resume_epoch, *args):
        self.start_epoch = self.epoch_num
        self.epoch_num = self.start_epoch + resume_epoch
        if isinstance(self.model, CDRModel):
            self.model.update_separate_period(resume_epoch)
        self.tmp_log_file = open(self.tmp_log_path, "a")

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader),
                                                                    eta_min=0.00001, last_epoch=-1)
        return self.train()

    def _before_epoch(self, epoch):
        self.model = self.model.to(self.device)
        training_loss = []

        if epoch == self.warmup_epochs:
            self.train_loader.dataset.transform.build_neighbor_repo(self.separation_epochs - self.warmup_epochs,
                                                                    self.n_neighbors)
        elif epoch == self.separation_epochs:
            self.train_loader.dataset.transform.build_neighbor_repo(self.epoch_num - self.separation_epochs,
                                                                    self.n_neighbors)

        train_iterator = iter(self.train_loader)

        return train_iterator, training_loss

    def _after_epoch(self, ckp_save_inter, epoch, training_loss, training_loss_history, val_inter):
        ret_val = super()._after_epoch(ckp_save_inter, epoch, training_loss, training_loss_history, val_inter)
        return ret_val

    def _step_prepare(self, *args):
        data, epoch, train_iterator = args
        x, x_sim, indices, sim_indices = data[0]

        x = x.to(self.device, non_blocking=True)
        x_sim = x_sim.to(self.device, non_blocking=True)
        return x, x_sim, epoch, indices, sim_indices

    def _train_step(self, *args):
        x, x_sim, epoch, indices, sim_indices = args

        self.optimizer.zero_grad()

        with torch.cuda.device(self.device_id):
            _, x_embeddings, _, x_sim_embeddings = self.forward(x, x_sim)

        x_and_x_sim = torch.cat([x, x_sim], dim=0)

        net = self.model

        train_loss = net.compute_loss(x_embeddings, x_sim_embeddings, epoch, x_and_x_sim, indices, sim_indices,
                                      self.train_loader.dataset.targets, self.result_save_dir, self.steps)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
        self.optimizer.step()
        return train_loss

    def model_prepare(self):
        self.model.preprocess()


