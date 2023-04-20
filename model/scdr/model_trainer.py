#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import shutil
from multiprocessing import Process
import torch
import os
import time

from model.scdr.dependencies.cdr_experiment import CDRsExperiments
from dataset.warppers import StreamingDatasetWrapper
from model.scdr.dependencies.scdr_utils import ClusterRepDataSampler
from utils.constant_pool import *
from utils.nn_utils import compute_knn_graph
from utils.queue_set import ModelUpdateQueueSet


class SCDRTrainer(CDRsExperiments):
    def __init__(self, cdr_model, dataset_name, configs, result_save_dir,
                 config_path, device='cuda', log_path="logs.txt"):
        CDRsExperiments.__init__(self, cdr_model, dataset_name, configs, result_save_dir, config_path, True, device,
                                 log_path)
        self.stream_dataset = None
        self.initial_train_epoch = configs.method_params.initializing_epochs
        self.finetune_epoch = configs.method_params.updating_epochs
        self.pre_torch_embeddings = None
        self._rep_old_data_indices = None
        self._is_incremental_learning = False
        self.incremental_steps = 0
        self._fixed_batch_num = 3
        self._initial_fixed_batch_num = 8

        self.rep_batch_nums = None
        self.__steady_weights = None

        self._rep_data_list = []
        self._rep_neighbor_data_list = []
        self._pre_rep_embedding_list = []
        self._pre_rep_neighbors_embedding_list = []
        self._steady_weights_list = []
        self._neighbor_steady_weights_list = []
        self._neighbor_nochange_list = []

    def update_batch_size(self, data_num):
        self.batch_size = min(data_num // self._fixed_batch_num, 1024)
        self.stream_dataset.batch_size = self.batch_size
        self.model.update_batch_size(int(self.batch_size))
        self.model.reset_partial_corr_mask()

    def initialize_streaming_dataset(self, dataset):
        self.stream_dataset = dataset

    def first_train(self, dataset: StreamingDatasetWrapper, ckpt_path=None):
        self.preprocess(load_data=False)
        self.initialize_streaming_dataset(dataset)
        self.batch_size = self.configs.method_params.batch_size
        tmp_batch_num = self._fixed_batch_num
        self._fixed_batch_num = self._initial_fixed_batch_num
        self.update_batch_size(dataset.get_n_samples())
        self._fixed_batch_num = tmp_batch_num
        self.update_dataloader(self.initial_train_epoch)

        self.result_save_dir_modified = False
        self.do_vis = False
        self.save_model = True
        self.save_final_embeddings = False
        self.draw_loss = False
        self.print_time_info = False
        self.result_save_dir = os.path.join(self.result_save_dir, "initial")
        if ckpt_path is None:
            self.epoch_num = self.initial_train_epoch
            launch_time_stamp = int(time.time())
            self.pre_embeddings = self.train(launch_time_stamp)
        else:
            self.load_checkpoint(ckpt_path)
            self.pre_embeddings = self.visualize(None, device=self.device)[0]
            self._train_begin(int(time.time()))

        if self.config_path is not None:
            if not os.path.exists(self.result_save_dir):
                os.makedirs(self.result_save_dir)
            shutil.copyfile(self.config_path, os.path.join(self.result_save_dir, "config.yaml"))

        self.active_incremental_learning()
        return self.pre_embeddings

    def prepare_resume(self, fitted_num, train_num, resume_epoch, sample_indices=None):
        self.update_batch_size(train_num)
        self.update_neg_num(train_num / 10)
        if sample_indices is None:
            sample_indices = np.arange(fitted_num, fitted_num + train_num, 1)
        self.update_dataloader(resume_epoch, sample_indices)

    def train(self, launch_time_stamp=None, target_metric_val=-1):
        embeddings = super().train(launch_time_stamp, target_metric_val)
        return embeddings

    def resume_train(self, resume_epoch, *args):
        rep_args = args[0]
        self._update_rep_data_info(*rep_args)

        self.pre_embeddings = super().resume_train(resume_epoch)
        return self.pre_embeddings

    def _train_step(self, *args):
        sta = time.time()
        x, x_sim, epoch, indices, sim_indices = args
        self.optimizer.zero_grad()

        with torch.cuda.device(self.device_id):
            _, x_embeddings, _, x_sim_embeddings = self.forward(x, x_sim)

        if self._is_incremental_learning:
            idx = self.incremental_steps % self.rep_batch_nums
            neighbor_idx = self.incremental_steps % self.n_neighbors

            cur_rep_data = self._rep_data_list[idx]
            pre_rep_embeddings = self._pre_rep_embedding_list[idx]
            pre_rep_neighbors_embeddings = self._pre_rep_neighbors_embedding_list[idx][neighbor_idx]
            rep_neighbors_data = self._rep_neighbor_data_list[idx][neighbor_idx]
            steady_weights = self._steady_weights_list[idx][neighbor_idx]
            neighbor_steady_weights = self._neighbor_steady_weights_list[idx][neighbor_idx]
            no_change_indices = self._neighbor_nochange_list[idx][neighbor_idx]

            with torch.cuda.device(self.device_id):
                rep_embeddings = self.model.acquire_latent_code(cur_rep_data)
                rep_neighbors_embeddings = self.model.acquire_latent_code(rep_neighbors_data)

            train_loss = self.model.compute_loss(x_embeddings, x_sim_embeddings, epoch,
                                                 self._is_incremental_learning, rep_embeddings, pre_rep_embeddings,
                                                 pre_rep_neighbors_embeddings, rep_neighbors_embeddings,
                                                 no_change_indices, steady_weights, neighbor_steady_weights, idx)
            self.incremental_steps += 1
        else:
            train_loss = self.model.compute_loss(x_embeddings, x_sim_embeddings, epoch, self._is_incremental_learning)

        train_loss.backward()
        self.optimizer.step()

        return train_loss

    def build_dataset(self, *args):
        new_data, new_labels = args[0], args[1]
        self.stream_dataset = StreamingDatasetWrapper(self.batch_size, self.n_neighbors)
        self.stream_dataset.add_new_data(data=new_data, labels=new_labels)

    def _reset_rep_data_info(self):
        self.incremental_steps = 0
        self._rep_data_list = []
        self._rep_neighbor_data_list = []
        self._pre_rep_embedding_list = []
        self._pre_rep_neighbors_embedding_list = []
        self._steady_weights_list = []
        self._neighbor_steady_weights_list = []
        self._neighbor_nochange_list = []

    def _update_rep_data_info(self, rep_batch_nums, rep_data_indices, cluster_indices, fitted_data_num,
                              steady_weights=None):
        self._reset_rep_data_info()
        self.rep_batch_nums = rep_batch_nums
        rep_data_indices = np.array(rep_data_indices, dtype=int)
        with torch.no_grad():
            pre_torch_embeddings = self.model.acquire_latent_code(torch.tensor(self.stream_dataset.get_total_data(),
                                                                               dtype=torch.float).to(self.device))
        if steady_weights is not None and not isinstance(steady_weights, torch.Tensor):
            steady_weights = torch.tensor(steady_weights, dtype=torch.float).to(self.device)

        for i, item in enumerate(rep_data_indices):
            cur_rep_data = self.stream_dataset.get_total_data()[item]
            cur_rep_data_torch = torch.tensor(cur_rep_data, dtype=torch.float).to(self.device)
            self._rep_data_list.append(cur_rep_data_torch)
            cur_pre_rep_embeddings = pre_torch_embeddings[item]
            self._pre_rep_embedding_list.append(cur_pre_rep_embeddings)

            tmp_pre_rep_neighbor_e_list = []
            tmp_rep_neighbor_data_list = []
            tmp_steady_weights = []
            tmp_neighbor_steady_weights = []
            tmp_neighbor_nochange_list = []

            for j in range(self.n_neighbors):
                cur_neighbor_indices = self.stream_dataset.get_knn_indices()[item, j]
                no_change_indices = np.where(cur_neighbor_indices < fitted_data_num)[0]
                if len(no_change_indices) > 0:
                    cur_neighbor_indices = cur_neighbor_indices[no_change_indices]

                    pre_rep_neighbor_e = pre_torch_embeddings[cur_neighbor_indices]
                    rep_neighbors_data = self.stream_dataset.get_total_data()[cur_neighbor_indices]
                    rep_steady_weights = steady_weights[item[no_change_indices]] if steady_weights is not None else None
                    neighbor_rep_steady_weights = steady_weights[
                        cur_neighbor_indices] if steady_weights is not None else None

                else:
                    pre_rep_neighbor_e = None
                    rep_neighbors_data = None
                    rep_steady_weights = None
                    neighbor_rep_steady_weights = None

                tmp_pre_rep_neighbor_e_list.append(pre_rep_neighbor_e)
                tmp_rep_neighbor_data_list.append(torch.tensor(rep_neighbors_data, dtype=torch.float).to(self.device))
                tmp_steady_weights.append(rep_steady_weights)
                tmp_neighbor_steady_weights.append(neighbor_rep_steady_weights)
                tmp_neighbor_nochange_list.append(no_change_indices)

            self._pre_rep_neighbors_embedding_list.append(tmp_pre_rep_neighbor_e_list)
            self._rep_neighbor_data_list.append(tmp_rep_neighbor_data_list)
            self._steady_weights_list.append(tmp_steady_weights)
            self._neighbor_steady_weights_list.append(tmp_neighbor_steady_weights)
            self._neighbor_nochange_list.append(tmp_neighbor_nochange_list)

        self.model.update_rep_data_info(np.array(cluster_indices))

    def update_dataloader(self, epochs, sampled_indices=None):
        if self.train_loader is None:
            self.train_loader, self.n_samples = self.stream_dataset.get_data_loaders(
                epochs, self.dataset_name, ConfigInfo.DATASET_CACHE_DIR, self.n_neighbors, is_image=self.is_image)
        else:
            if sampled_indices is None:
                sampled_indices = np.arange(0, self.stream_dataset.get_total_data().shape[0], 1)
            self.train_loader, self.n_samples = self.stream_dataset.update_data_loaders(epochs, sampled_indices)

    def active_incremental_learning(self):
        self._is_incremental_learning = True

    def update_neg_num(self, neg_num=None):
        neg_num = self.model.neg_num if neg_num is None else neg_num
        self.model.update_neg_num(neg_num)

    def update_train_loader(self, train_indices):
        self.train_loader, self.n_samples = \
            self.cdr_dataset.get_train_validation_data_loaders(self.cdr_dataset.train_dataset, None,
                                                               train_indices, [], False, False)

    def quantitative_test_all(self, epoch, embedding_data=None, mid_embeddings=None, device='cuda', val=False):
        self.metric_tool = None
        embedding_data, k, = self.quantitative_test_preprocess(embedding_data, device)[:2]
        self.model_update_queue_set.eval_data_queue.put([epoch, k, embedding_data, (epoch == self.epoch_num), False])


class SCDRTrainerProcess(SCDRTrainer, Process):
    def __init__(self, model_update_queue_set, model, dataset_name, config_path,
                 configs, result_save_dir, device='cuda:0', log_path="log_streaming.txt"):
        SCDRTrainer.__init__(self, model, dataset_name, configs, result_save_dir, config_path, device, log_path)
        Process.__init__(self, name="model update process")
        self.model_update_queue_set = model_update_queue_set
        self.update_count = 0
        self.rep_data_sample_rate = 0.07
        self.rep_data_minimum_num = 50
        self.cover_all = True

        self.cluster_rep_data_sampler = ClusterRepDataSampler(self.rep_data_sample_rate, self.rep_data_minimum_num,
                                                              self.cover_all)
        self._last_fit_data_idx = 0
        self.pre_rep_data_info = None

    def run(self) -> None:
        while True:

            flag = self.model_update_queue_set.flag_queue.get()
            if flag == ModelUpdateQueueSet.SAVE:
                self.save_weights(self.epoch_num)
                continue
            elif flag == ModelUpdateQueueSet.STOP:
                self.ending()
                break

            training_info = self.model_update_queue_set.training_data_queue.get()
            if self.update_count == 0:
                stream_dataset, _, ckpt_path = training_info
                embeddings = self.first_train(stream_dataset, ckpt_path)
                self._last_fit_data_idx = stream_dataset.get_n_samples()
                total_data_idx = embeddings.shape[0]
            else:
                stream_dataset, _, fitted_data_num, cur_data_num, total_data_idx, out_num = training_info

                pre_symm_nn_indices = self.stream_dataset.symmetric_nn_indices[out_num:] - out_num
                pre_symm_nn_weights = self.stream_dataset.symmetric_nn_weights[out_num:]

                for i in range(len(pre_symm_nn_indices)):
                    indices = np.where(pre_symm_nn_indices[i] >= 0)
                    pre_symm_nn_indices[i] = pre_symm_nn_indices[i][indices]
                    pre_symm_nn_weights[i] = pre_symm_nn_weights[i][indices]

                self.stream_dataset = stream_dataset
                self.stream_dataset.symmetric_nn_indices = np.concatenate([pre_symm_nn_indices, np.ones(cur_data_num)])
                self.stream_dataset.symmetric_nn_weights = np.concatenate([pre_symm_nn_weights, np.ones(cur_data_num)])

                knn_indices, knn_dists = compute_knn_graph(stream_dataset.get_total_data(), None, self.n_neighbors, None)
                self.stream_dataset._knn_manager.update_knn_graph(knn_indices, knn_dists)

                self.stream_dataset.update_cached_neighbor_similarities()

                sample_indices = np.arange(fitted_data_num, fitted_data_num + cur_data_num)

                self.prepare_resume(fitted_data_num, len(sample_indices), self.finetune_epoch, sample_indices)
                steady_constraints = self.stream_dataset.cal_old2new_relationship(old_n_samples=fitted_data_num)

                cluster_indices, rep_batch_nums, rep_data_indices = \
                    self._sample_rep_data(stream_dataset.get_total_embeddings(), fitted_data_num)
                self.pre_rep_data_info = [rep_batch_nums, rep_data_indices, cluster_indices, fitted_data_num,
                                          steady_constraints]

                embeddings = self.resume_train(self.finetune_epoch, self.pre_rep_data_info)
                self.model_update_queue_set.WAITING_UPDATED_DATA.value = 1

            ret = [embeddings, self.model.copy_network().cpu(), self.stream_dataset, total_data_idx]
            self.model_update_queue_set.embedding_queue.put(ret)
            self.model_update_queue_set.MODEL_UPDATING.value = 0
            self.update_count += 1

    def _sample_rep_data(self, total_embeddings, fitted_num):
        embeddings = total_embeddings[:fitted_num]
        ravel_1 = np.reshape(np.repeat(total_embeddings[:, np.newaxis, :], self.n_neighbors // 2, 1), (-1, 2))
        ravel_2 = total_embeddings[np.ravel(self.stream_dataset.get_knn_indices()[:, :self.n_neighbors // 2])]

        embedding_nn_dist = np.mean(np.linalg.norm(ravel_1 - ravel_2, axis=-1))
        rep_batch_nums, rep_data_indices, cluster_indices, _, _ = \
            self.cluster_rep_data_sampler.sample(embeddings, eps=embedding_nn_dist, min_samples=self.n_neighbors,
                                                 labels=self.stream_dataset.get_total_label()[:fitted_num])

        return cluster_indices, rep_batch_nums, rep_data_indices

    def ending(self):
        pass
