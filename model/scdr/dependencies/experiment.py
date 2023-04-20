#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import math
from utils.common_utils import time_stamp_to_date_time_adjoin
from utils.math_utils import *
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import shutil
from multiprocessing import Queue
from utils.logger import InfoLogger
import seaborn as sns


def draw_loss(training_loss, test_loss, idx, save_path=None):
    plt.figure()
    plt.plot(idx, training_loss, color="blue", label="training loss")
    if len(test_loss) > 0:
        plt.plot(idx, test_loss, color="red", label="test loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def position_vis(c, vis_save_path, z, title=None):
    x = z[:, 0]
    y = z[:, 1]

    plt.figure(figsize=(8, 8))
    if c is None:
        sns.scatterplot(x=x, y=y, s=8, legend=False, alpha=1.0)
    else:
        c = np.array(c, dtype=int)
        classes = np.unique(c)
        num_classes = classes.shape[0]
        palette = "tab10" if num_classes <= 10 else "tab20"
        sns.scatterplot(x=x, y=y, hue=c, s=8, palette=palette, legend=False, alpha=1.0)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    if vis_save_path is not None:
        plt.savefig(vis_save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def draw_projections(z, c, vis_save_path):
    position_vis(c, vis_save_path, z)


class Experiment:
    def __init__(self, model, dataset_name, configs, result_save_dir, config_path, shuffle, device='cuda',
                 log_path="logs.txt"):
        self.model = model
        self.config_path = config_path
        self.configs = configs
        self.device = device
        self.result_save_dir_modified = False
        self.result_save_dir = result_save_dir
        self.dataset_name = dataset_name
        self.train_loader = None
        self.n_neighbors = configs.method_params.n_neighbors
        self.epoch_num = configs.method_params.initializing_epochs
        self.batch_size = configs.method_params.batch_size
        self.lr = configs.method_params.LR

        self.n_samples = 0
        self.batch_num = 0
        self.vis_inter = 0
        self.start_epoch = 0
        self.ckp_save_dir = None

        self.PRINT_ITER = 10
        self.launch_date_time = None
        self.optimizer = None
        self.scheduler = None

        self.shuffle = shuffle
        self.is_image = False
        self.device_id = int(self.device.split(":")[1])

        self.tmp_log_path = log_path
        self.tmp_log_file = None
        self.log_process = None
        self.log_path = None
        self.message_queue = Queue()

        self.pre_embeddings = None

        self.do_vis = True
        self.save_model = True
        self.save_final_embeddings = True
        self.draw_loss = True
        self.print_time_info = True

        self.FIXED_K = 10

    def _train_begin(self, launch_time_stamp=None):
        self.model = self.model.to(self.device)
        self.sta_time = time.time() if launch_time_stamp is None else launch_time_stamp
        if self.device_id == 0:
            InfoLogger.info("Start Training for {} Epochs".format(self.epoch_num))

        self.initial_result_save_dir(launch_time_stamp)
        self.log_path = os.path.join(self.result_save_dir, "logs.txt")
        self.ckp_save_dir = self.result_save_dir

        if self.optimizer is None:
            self.init_optimizer()
            self.init_scheduler(cur_epochs=self.epoch_num)

        batch_print_inter = 0
        vis_inter = math.ceil(self.epoch_num * self.configs.exp_params.vis_iter)
        ckp_save_inter = math.ceil(self.epoch_num * self.configs.exp_params.save_iter)

        return batch_print_inter, vis_inter, ckp_save_inter

    def initial_result_save_dir(self, launch_time_stamp):
        if self.launch_date_time is None:
            if launch_time_stamp is None:
                launch_time_stamp = int(time.time())
            self.launch_date_time = time_stamp_to_date_time_adjoin(launch_time_stamp)
        if not self.result_save_dir_modified:
            self.result_save_dir = os.path.join(self.result_save_dir,
                                                "{}_{}".format(self.dataset_name, self.launch_date_time))
            self.result_save_dir_modified = True

    def init_optimizer(self):
        self.configs.method_params.optimizer = "adam"
        if self.configs.method_params.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        elif self.configs.method_params.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9,
                                             weight_decay=0.0001)
        else:
            raise RuntimeError("Unsupported optimizer!")

    def init_scheduler(self, cur_epochs, base=0, gamma=0.1, milestones=None):
        self.configs.method_params.scheduler = "multi_step"
        if milestones is None:
            milestones = [0.8, 0.9]
        if self.configs.method_params.scheduler == "multi_step":
            self.scheduler = MultiStepLR(self.optimizer, milestones=[int(base + p * cur_epochs) for p in milestones],
                                         gamma=gamma, last_epoch=-1)
        elif self.configs.method_params.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader),
                                                                        eta_min=0.00001, last_epoch=-1)
        else:
            raise RuntimeError("Unsupported learning schedule!")

    def _before_epoch(self, epoch):
        self.model = self.model.to(self.device)
        training_loss = 0
        train_iterator = iter(self.train_loader)

        return train_iterator, training_loss

    def _step_prepare(self, *args):
        pass

    def _train_step(self, *args):
        return None

    def _after_epoch(self, ckp_save_inter, epoch, training_loss, training_loss_history, val_inter):

        self.scheduler.step()
        train_loss = np.mean(training_loss)
        if epoch % self.PRINT_ITER == 0:
            epoch_template = 'Epoch %d/%d, Train Loss: %.5f, '
            epoch_output = epoch_template % (epoch, self.epoch_num, train_loss)
            InfoLogger.info(epoch_output)
            self.message_queue.put(epoch_output)

        training_loss_history.append(float(train_loss))
        embeddings, h = self.post_epoch(ckp_save_inter, epoch, val_inter)

        return embeddings

    def _train_end(self, test_loss_history, training_loss_history, embeddings):
        if self.save_final_embeddings:
            np.save(os.path.join(self.result_save_dir, "embeddings_{}.npy".format(self.epoch_num)), embeddings)

        if self.print_time_info:
            end_time = time.time()
            self.sta_time = end_time
        self.message_queue.put("end")

        if self.save_model:
            self.save_weights(self.epoch_num)

        if self.draw_loss:
            x_idx = np.linspace(self.start_epoch, self.epoch_num, self.epoch_num - self.start_epoch)
            save_path = os.path.join(self.result_save_dir,
                                     "{}_loss_{}.jpg".format(self.configs.method_params.method, self.epoch_num))
            draw_loss(training_loss_history, test_loss_history, x_idx, save_path)

        if self.tmp_log_file is not None and self.tmp_log_path != self.log_path:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)

        InfoLogger.info("Training process logging to {}".format(self.log_path))

    def train(self, launch_time_stamp=None, target_metric_val=-1):
        batch_print_inter, self.vis_inter, ckp_save_inter = self._train_begin(launch_time_stamp)
        embeddings = None
        param_dict = None
        training_loss_history = []
        test_loss_history = []

        for epoch in range(self.start_epoch, self.epoch_num):
            train_iterator, training_loss = self._before_epoch(epoch)
            for idx, data in enumerate(train_iterator):
                train_data = self._step_prepare(data, param_dict, train_iterator)
                loss = self._train_step(*train_data)
                training_loss += loss

            metrics, val_metrics, embeddings = self._after_epoch(ckp_save_inter, epoch + 1, training_loss,
                                                                 training_loss_history, self.vis_inter)

        self._train_end(test_loss_history, training_loss_history, embeddings)
        return embeddings

    def quantitative_test_preprocess(self, embedding_data=None, device='cuda'):

        data = torch.tensor(self.train_loader.dataset.get_all_data()).to(device).float()

        if self.is_image:
            data = data / 255.
        if embedding_data is None:
            self.model.to(device)
            embedding_data, _ = self.acquire_latent_code_allin(data, device)

        return embedding_data, self.FIXED_K, data

    def save_weights(self, epoch, prefix_name=None):
        if prefix_name is None:
            prefix_name = epoch
        if not os.path.exists(self.ckp_save_dir):
            os.makedirs(self.ckp_save_dir)
        weight_save_path = os.path.join(self.ckp_save_dir, "{}.pth.tar".
                                        format(prefix_name))
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr': self.lr, 'launch_time': self.launch_date_time}, weight_save_path)
        InfoLogger.info("weights successfully save to {}".format(weight_save_path))

    def load_checkpoint(self, checkpoint_path):
        model_ckpt = torch.load(checkpoint_path)
        self.model.load_state_dict(model_ckpt['state_dict'])
        InfoLogger.info('loading checkpoint success!')
        if self.optimizer is None:
            self.init_optimizer()
        self.optimizer.load_state_dict(model_ckpt['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def train_for_visualize(self):
        if self.device_id == 0:
            InfoLogger.info("Start train for Visualize")
        launch_time_stamp = int(time.time())
        self.preprocess()
        self.pre_embeddings = self.train(launch_time_stamp)
        return self.pre_embeddings

    def cal_lower_embeddings(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float).to(self.device)
        if self.is_image:
            data = data / 255.
        embeddings = self.acquire_latent_code_allin(data, self.device)
        return embeddings

    def visualize(self, vis_save_path=None, device="cuda"):
        self.model.to(device)

        data = torch.tensor(self.train_loader.dataset.get_all_data()).to(device).float()
        z = self.cal_lower_embeddings(data)
        val_z = None

        if self.configs.exp_params.latent_dim <= 2:
            draw_projections(z, self.train_loader.dataset.targets, vis_save_path)

        return z, val_z

    def acquire_latent_code_allin(self, data, device):
        with torch.no_grad():
            self.model.eval()
            data = data.to(device)
            z = self.model.acquire_latent_code(data)
            self.model.train()

            z = z.cpu().numpy()
        return z

    def preprocess(self, train=True, load_data=True):
        if load_data:
            self.build_dataset()

        self.model_prepare()

    def model_prepare(self):
        pass

    def build_dataset(self):
        pass

    def post_epoch(self, ckp_save_inter, epoch, vis_inter):

        embeddings = None
        h = None

        vis_save_path = os.path.join(self.result_save_dir, '{}_vis_{}.jpg'.format(self.dataset_name, epoch))
        final = epoch == self.epoch_num

        if (epoch % vis_inter == 0 and self.do_vis) or epoch == self.epoch_num:
            if not os.path.exists(self.result_save_dir):
                os.makedirs(self.result_save_dir)
                if self.config_path is not None:
                    shutil.copyfile(self.config_path, os.path.join(self.result_save_dir, "config.yaml"))

            embeddings, val_embeddings = self.visualize(vis_save_path, device=self.device)

        if epoch % ckp_save_inter == 0:
            if not os.path.exists(self.ckp_save_dir):
                os.mkdir(self.ckp_save_dir)
            self.save_weights(epoch)

        return embeddings, h
