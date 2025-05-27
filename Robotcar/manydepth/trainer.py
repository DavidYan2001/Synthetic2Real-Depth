# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import numpy as np
import time
import random

import torch
from torch import nn
import torch.distributed as dist  
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision
import json
from .utils import readlines, sec_to_hm_str, flip_lr, post_process_inv_depth, inv2depth
from .layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors
from collections import OrderedDict
from Robotcar.manydepth import datasets, networks
from Robotcar.manydepth.datasets.nuscenes_dataset import NuScenesDataset, Subset, NuScenesDataSubset
from Robotcar.robotcar.robotcar_dataset import RobotCarDataset, Subset, RobotCarDataSubset
import matplotlib.pyplot as plt
from Robotcar.manydepth.utils import compute_depth_metrics
import copy
from .visualize import Visualizer


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def read_txt_to_tensor(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Convert each line into a list of floats
            data = [list(map(float, line.strip().split())) for line in lines]

            # Convert the list of lists to a tensor
            tensor_data = torch.tensor(data)
            return tensor_data

class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose
        if self.train_teacher_and_pose:
            print('using adaptive depth binning!')
            self.min_depth_tracker = self.opt.min_depth_tracker
            # self.max_depth_tracker = 10.0
            self.max_depth_tracker = self.opt.max_depth_tracker
        else:
            self.min_depth_tracker = self.opt.min_depth_tracker
            self.max_depth_tracker = self.opt.max_depth_tracker
            print('fixing pose network and monocular network!')

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))


        #daytime model set up
        self.models["daytime_encoder"] = \
            networks.ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=False, min_depth_bin=self.opt.min_depth_tracker, max_depth_bin=self.opt.max_depth_tracker,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins,normalizing_mode='Dataset')
        self.models["daytime_encoder"].to(self.device)

        self.models["daytime_depth"] = \
            networks.DepthDecoder(self.models["daytime_encoder"].num_ch_enc, self.opt.scales)
        self.models["daytime_depth"].to(self.device)

        self.models["daytime_pose_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                   num_input_images=self.num_pose_frames, normalize_mode='Dataset')
        self.models["daytime_pose_encoder"].to(self.device)

        self.models["daytime_pose"] = \
            networks.PoseDecoder(self.models["daytime_pose_encoder"].num_ch_enc,
                                 num_input_features=1,
                                 num_frames_to_predict_for=2)
        self.models["daytime_pose"].to(self.device)


        self.models["teacher_encoder"] = \
            networks.ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=False, min_depth_bin=self.opt.min_depth_tracker, max_depth_bin=self.opt.max_depth_tracker,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins,normalizing_mode='Dataset')
        self.models["teacher_encoder"].to(self.device)

        self.models["teacher_depth"] = \
            networks.DepthDecoder(self.models["teacher_encoder"].num_ch_enc, self.opt.scales)
        self.models["teacher_depth"].to(self.device)

        self.models["teacher_pose_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                   num_input_images=self.num_pose_frames, normalize_mode='Dataset')
        self.models["teacher_pose_encoder"].to(self.device)

        self.models["teacher_pose"] = \
            networks.PoseDecoder(self.models["teacher_pose_encoder"].num_ch_enc,
                                 num_input_features=1,
                                 num_frames_to_predict_for=2)
        self.models["teacher_pose"].to(self.device)

        #student model set up
        self.models["encoder"] = networks.ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=True, min_depth_bin=self.opt.min_depth_tracker, max_depth_bin=self.opt.max_depth_tracker,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins,normalizing_mode='Daytime')
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        self.models["pose_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                   num_input_images=self.num_pose_frames, normalizing_mode='Daytime')
        self.models["pose_encoder"].to(self.device)

        self.models["pose"] = \
            networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                 num_input_features=1,
                                 num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.parameters_to_train += list(self.models["pose"].parameters())
        

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, self.opt.scheduler_w)

        # if self.opt.load_weights_folder is not None:
        #     self.load_model()

        file_path = "/mnt/codemnt/weilong/syn2real_depth/Robotcar/depth_stat_day_clear_diff_origin.txt"
        self.distribution_daytime = read_txt_to_tensor(file_path).to(self.device).view(101)
        self.distribution_daytime = self.distribution_daytime/self.distribution_daytime.sum()
       



        if self.opt.load_daytime_weights_folder is not None:
            self.load_daytime_model()

        if self.opt.load_teacher_weights_folder is not None:
            self.load_teacher_model()

        if self.opt.load_student_weights_folder is not None:
            self.load_student_model()



        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        #nuscenes dataset
        resize = datasets.transforms.Resize(size=(self.opt.resized_height, self.opt.resized_width),
                                        interpolation=torchvision.transforms.InterpolationMode.LANCZOS,
                                        antialias=True, sample_keys=['color', 'color_aug'],
                                        temp_context=self.opt.temp_context,
                                        original_size=(self.opt.original_height, self.opt.original_width))

        color_jitter = datasets.transforms.RandomColorJitter(brightness=self.opt.brightness,
                                                   contrast=self.opt.contrast,
                                                   saturation=self.opt.saturation,
                                                   hue=self.opt.hue,
                                                   p=self.opt.color_jitter_prob,
                                                   sample_keys=self.opt.color_jitter_keys, temp_context=self.opt.temp_context,
                                                    scales=[0])
        gaussian_noise = datasets.transforms.RandomGaussianNoise(min_rand=self.opt.gaussian_noise_rand_min,
                                                             max_rand=self.opt.gaussian_noise_rand_max,
                                                             p=self.opt.gaussian_noise_prob,
                                                             sample_size=(3, self.opt.resized_height, self.opt.resized_width),
                                                             sample_keys=self.opt.gaussian_noise_keys,
                                                             temp_context=self.opt.temp_context,
                                                             scales=[0])
        gaussian_blur = datasets.transforms.RandomGaussianBlur(kernel_size=self.opt.gaussian_blur_kernel_size,
                                                     sigma=self.opt.gaussian_blur_sigma,
                                                     p=self.opt.gaussian_blur_prob,
                                                     sample_keys=self.opt.gaussian_blur_keys, temp_context=self.opt.temp_context, scales=[0])
        to_tensor = datasets.transforms.ToTensor(sample_keys=['color', 'color_aug', 'color_aug_pose'],
                                             temp_context=self.opt.temp_context, scales=self.opt.scales)
        to_tensor_test = datasets.transforms.ToTensor_test(sample_keys=['color', 'color_aug', 'color_aug_pose'],
                                             temp_context=[0], scales=self.opt.scales)

        # train_transform_non_deterministic = [resize, to_tensor]
        train_transform_non_deterministic = [to_tensor]
        if self.opt.gaussian_blur_prob > 0.0:
            train_transform_non_deterministic.insert(0, gaussian_blur)
        if self.opt.gaussian_noise_prob > 0.0:
            train_transform_non_deterministic.insert(0, gaussian_noise)
        if self.opt.color_jitter_prob > 0.0:
            train_transform_non_deterministic.insert(0, color_jitter)
        if self.opt.day_night_translation_prob > 0.0:
            day_night_translation_train = datasets.transforms.DaytimeTranslation(
                path=self.opt.day_night_translation_path,
                direction=self.opt.day_night_translation_direction,
                p=self.opt.day_night_translation_prob,
                sample_keys=self.opt.day_night_translation_keys,
                scales = [0],
                temp_context=[0] if self.opt.day_night_translation_key_frame_only else self.opt.temp_context,
                resized_height=self.opt.resized_height, 
                resized_width=self.opt.resized_width)
            train_transform_non_deterministic.insert(0, day_night_translation_train)
        if self.opt.day_clear_day_rain_translation_prob > 0.0:
            day_clear_day_rain_translation_train = datasets.transforms.DaytimeTranslation(
                path=self.opt.day_clear_day_rain_translation_path,
                direction=self.opt.day_clear_day_rain_translation_direction,
                p=self.opt.day_clear_day_rain_translation_prob,
                sample_keys=self.opt.day_clear_day_rain_translation_keys,
                temp_context=[0] if self.opt.day_clear_day_rain_translation_key_frame_only else self.opt.temp_context,
                scales = [0],
                resized_width=self.opt.resized_width, 
                resized_height=self.opt.resized_height)
            train_transform_non_deterministic.insert(0, day_clear_day_rain_translation_train)

        train_transform = train_transform_non_deterministic if not self.opt.deterministic else [to_tensor]
        val_transform = [to_tensor]
        eval_test_transform = [to_tensor_test]

        if self.opt.evaluation_day_night_translation_enabled:
            day_night_translation_eval = datasets.transforms.DaytimeTranslation(
                path=self.opt.day_night_translation_path,
                direction=self.opt.day_night_translation_direction,
                p=1.0,
                sample_keys=self.opt.day_night_translation_keys,
                temp_context=self.opt.temp_context)
            eval_test_transform.insert(0, day_night_translation_eval)
        if self.opt.evaluation_day_clear_day_rain_translation_enabled:
            day_clear_day_rain_translation_eval = datasets.transforms.DaytimeTranslation(
                path=self.opt.day_clear_day_rain_translation_path,
                direction=self.opt.day_clear_day_rain_translation_direction,
                p=1.0,
                sample_keys=self.opt.day_clear_day_rain_translation_keys,
                temp_context=self.opt.temp_context)
            eval_test_transform.insert(0, day_clear_day_rain_translation_eval)

        transform = {'train': datasets.transforms.CustomCompose(train_transform),
                     'val': datasets.transforms.CustomCompose(val_transform),
                     'test': datasets.transforms.CustomCompose(eval_test_transform)}
        

        self.visualizer = Visualizer(self.opt.frame_ids, False,
                                     self.opt.evaluation_qualitative_res_path, self.opt.evaluation_visualization_set,
                                     self.opt.save_rgb, self.opt.save_depth_pred,
                                     self.opt.save_depth_gt, 
                                     self.opt.save_lowest_cost,
                                     self.opt.min_depth,
                                     self.opt.max_depth)
            
            
        robotcar_dataset = RobotCarDataset(self.opt, transform)

        robotcar_dataset_train = RobotCarDataSubset(robotcar_dataset=robotcar_dataset, mode='train')
        if self.opt.train_repeat > 1:
            self.train_dataset = Subset(robotcar_dataset_train, [train_idx for train_idcs in
                                                           [robotcar_dataset_train.indices for _ in
                                                            range(self.opt.train_repeat)]
                                                           for train_idx in train_idcs])
        else:
            self.train_dataset = Subset(robotcar_dataset_train, robotcar_dataset_train.indices)

        self.train_loader = DataLoader(
                self.train_dataset, self.opt.batch_size, True, 
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=seed_worker)

        robotcar_subset_val = RobotCarDataSubset(robotcar_dataset=robotcar_dataset, mode='test')
        self.val_dataset = Subset(robotcar_subset_val, robotcar_subset_val.indices)
        self.val_loader = DataLoader(
                self.val_dataset, self.opt.batch_size, True, 
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        

        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val", "epoch_val", "epoch_val_teacher"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.resized_height // (2 ** scale)
            w = self.opt.resized_width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.weather_conditions_names =[
            'day-clear', 'day-rain', 'night-clear', 'night-rain'
        ]

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        self.depth_ms_metric_names = [
            "de_ms/abs_rel", "de_ms/sq_rel", "de_ms/rms", "de_ms/log_rms", "da_ms/a1", "da_ms/a2", "da_ms/a3"]

        # print("Using split:\n  ", self.opt.split)
        if self.opt.running_mode == "train":
            print("There are {:d} training items and {:d} validation items\n".format(
                len(self.train_dataset), len(self.val_dataset)))
        elif self.opt.running_mode == "val":
            print("There are {:d} validation items\n".format(
                len(self.val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """

        for k, m in self.models.items():
            if self.train_teacher_and_pose:
                m.train()
            else:
                # if teacher + pose is frozen, then only use training batch norm stats for
                # multi components
                if k in ['depth', 'encoder']:
                    m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def set_distillate(self):
        """convert teacher depth and teacher pose to eval
            convert student depth and pose to train
        """
        for m in self.models.values():
            m.eval()

        for k, m in self.models.items():
            
            if k in ['depth', 'encoder', 'pose_encoder', 'pose']:
                m.train()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        if self.opt.running_mode == "train":
            
            self.start_time = time.time()
            self.epoch_val(val_multi=True)
            for self.epoch in range(self.opt.num_epochs):
                if self.epoch == self.opt.freeze_pose_epoch:
                    self.freeze_pose()

                self.run_epoch()
                
                print("start evaluate multi!")
                self.epoch_val(val_multi=True)

                if (self.epoch+1) % self.opt.eval_teacher_interval ==0:
                # and (self.epoch+1) != self.opt.num_epochs:
                    print("start evaluate teacher!")
                    # self.epoch_val(val_teacher=True)

                if (self.epoch + 1) % self.opt.save_frequency == 0:
                    self.save_model()

        elif self.opt.running_mode == "val":

            self.epoch_val(val_multi=True)

        else:
            assert NotImplementedError

    def freeze_pose(self):
        if self.train_teacher_and_pose:
            self.train_teacher_and_pose = False
            print('freezing pose networks!')

            # here we reinitialise our optimizer to ensure there are no updates to the
            # teacher and pose networks
            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, self.opt.scheduler_w)

            # set eval so that teacher + pose batch norm is running average
            self.set_eval()
            # set train so that multi batch norm is in train mode
            self.set_train()

    def set_freeze_teacher(self):
        """Convert all models to testing/evaluation mode
        """
        self.models["daytime_encoder"].eval()
        self.models["daytime_depth"].eval()
        self.models["daytime_pose_encoder"].eval()
        self.models["daytime_pose"].eval()

        self.models["teacher_encoder"].eval()
        self.models["teacher_depth"].eval()
        self.models["teacher_pose_encoder"].eval()
        self.models["teacher_pose"].eval()

        # self.models["encoder"].eval()
        # self.models["depth"].eval()
        # self.models["pose_encoder"].eval()
        # self.models["pose"].eval()
        self.models["encoder"].train()
        self.models["depth"].train()
        self.models["pose_encoder"].train()
        self.models["pose"].train()

        self.models["mono_teacher"].eval()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")

        # self.epoch_val()
        # self.set_train()
        self.set_freeze_teacher()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            print("consistency_loss={}".format(losses["consistency_loss/0"]))
            print("translation_loss={}".format(losses["trans_loss"]))
            print("rotation_loss={}".format(losses["rot_loss"]))
            print("depth_cv_loss={}".format(losses["depth_cv_loss"]))
            print("flow_cv_loss={}".format(losses["flow_cv_loss"]))
            print("align_loss={}".format(losses["align_loss"]))
            print("total_loss={}".format(losses["loss"]))
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                # self.val()

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)

            # if self.step == self.opt.freeze_teacher_step:
            #     self.freeze_teacher()

            self.step += 1
        
        self.model_lr_scheduler.step()

    def differentiable_histogram(self, img, num_bins, min_depth, max_depth):
        """
        Compute a differentiable histogram for an image tensor using two sigmoids.
        
        Args:
            img (torch.Tensor): The input image tensor of size (N, H, W).
            num_bins (int): The number of bins for the histogram.
            min_depth (float): The minimum depth value.
            max_depth (float): The maximum depth value.
            width (float): The width W that controls the smoothness of the sigmoid.
        
        Returns:
            hist (torch.Tensor): A tensor of shape (N, num_bins) representing the histogram.
        """
        N,_, H, W = img.shape  # Assume input is (N, H, W)
        L = (max_depth - min_depth) / num_bins  # Bin width
        width = L/10
        bin_centers = torch.linspace(min_depth + L / 2, max_depth - L / 2, num_bins).to(img.device)  # bin centers
        bin_centers = torch.linspace(min_depth, max_depth, num_bins).to(img.device)

        img_flat = img.view(N*H*W)  # Flatten to (N, H*W)

        # Calculate the bin probabilities using two sigmoids
        hist = torch.zeros((num_bins), device=img.device)  # Initialize histogram

        for i, mu_k in enumerate(bin_centers):
            upper_sigmoid = torch.sigmoid((img_flat - (mu_k - L / 2)) / width) #N*hw
            lower_sigmoid = torch.sigmoid((img_flat - (mu_k + L / 2)) / width) #N*hw
            hist[i] = torch.sum(upper_sigmoid - lower_sigmoid)  # Sum over pixels

        # Normalize the histogram by the total number of pixels
        hist = (hist+1e-7) / (N * H * W)
        # print(hist)
        # hist = hist + 1e-7

        hist = hist/torch.sum(hist)

        

        return hist
    

    def epoch_val(self, val_teacher=False, val_multi=False):
        # 重置验证集迭代器
        self.set_eval()

        self.epoch_val_iter = iter(self.val_loader)

        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')

        self.metric_conditions = ('all-conditions', 'day', 'night', 'clear', 'rain', 'day-clear', 'day-rain',
                                  'night-clear', 'night-rain')

        # Dictionary for metrics in different conditions
        self.metrics = OrderedDict({condition: {mode: {metric: 0.0 for metric in self.metrics_keys} for mode in self.metrics_modes} for condition in self.metric_conditions})
        for condition in self.metrics.keys():
            self.metrics[condition]['count'] = 0
        
        num_bins = 101
        L = (80 - 3.5) / (num_bins-1)  # Bin width
        width = L/20
        # bin_centers = torch.linspace(3.5 + L / 2, 80 - L / 2, num_bins).to(self.device)  # bin centers
        bin_centers = torch.linspace(3.5, 80, num_bins).to(self.device)
        hist_day_clear = torch.zeros((num_bins), device=self.device) 
        hist_day_rain = torch.zeros((num_bins), device=self.device) 
        hist_night = torch.zeros((num_bins), device=self.device) 

        # 遍历整个验证集
        count = 0
        while True:
            try:
                inputs = next(self.epoch_val_iter)
            except StopIteration:
                break

            with torch.no_grad():

                outputs = self.process_batch_val_single(inputs, val_multi=val_multi, val_teacher=val_teacher)
                depth = outputs["depth"]
                depth_pp = outputs["depth_pp"]

                self.visualizer.visualize_test(inputs, outputs)
                




                for i, weather in enumerate(inputs["weather"]):
                    gt_i = inputs["depth_gt"][i]
                    valid = (gt_i > self.opt.min_depth) & (gt_i < self.opt.max_depth)
                    if valid.sum() == 0:
                        continue
                    for metric_condition in self.metric_conditions:
                        if metric_condition in weather:
                            self.metrics[metric_condition]['count'] += 1
                    self.metrics['all-conditions']['count'] += 1

              


                if self.opt.running_mode == "val":
                    print(count)

                    # self.visualizer.visualize_test(inputs, outputs)
                count = count+1

                # Calculate predicted metrics
                for mode in self.metrics_modes:
                    compute_depth_metrics(
                        gt=inputs["depth_gt"], pred=depth_pp if 'pp' in mode else depth, weather=inputs["weather"],
                        metrics=self.metrics, mode=mode, min_depth=self.opt.min_depth,
                        max_depth=self.opt.max_depth, use_gt_scale='gt' in mode
                    )

        results = self.compute_average_metrics()
        export_dict = OrderedDict({condition: {'everything': {}} for condition in self.metric_conditions})
        for condition in self.metric_conditions:
            for mode in self.metrics_modes:
                for metric in self.metrics_keys:
                    export_dict[condition]['everything'][metric + mode] = results[condition][mode][metric]
            export_dict[condition]['everything']['count'] = float(results[condition]['count'])
        
        if val_teacher:
            writer = self.writers["epoch_val_teacher"]
        elif val_multi:
            writer = self.writers["epoch_val"]
        for condition in export_dict.keys():
            for obj_class in export_dict[condition].keys():
                for metric_name, metric_value in export_dict[condition][obj_class].items():
                    # self.log(f"metrics_{condition}/{obj_class}/{metric_name}", metric_value)
                    writer.add_scalar(f"metrics_{condition}/{obj_class}/{metric_name}", metric_value, self.step)

        del export_dict, results, self.metrics

    def compute_average_metrics(self):
        intermediate_res = copy.deepcopy(self.metrics)
        for condition in self.metric_conditions:
            for mode in self.metrics_modes:
                for metric in self.metrics_keys:
                    intermediate_res[condition][mode][metric] = self.metrics[condition][mode][metric] / self.metrics[condition]['count'] if self.metrics[condition]['count'] != 0 else 0.0

        return intermediate_res
    
    

                

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if torch.is_tensor(ipt):
                inputs[key] = ipt.to(self.device)


        teacher_outputs = {}
        outputs = {}

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        #teacher model (pose+depth)
        with torch.no_grad():

            daytime_pose_pred = self.predict_poses_daytime_model(inputs, None)
            daytime_lookup_frames = [inputs[('color', idx, 0)] for idx in self.matching_ids[1:]]
            daytime_lookup_frames = torch.stack(daytime_lookup_frames, 1)

            daytime_relative_poses = [daytime_pose_pred[('relative_pose', idx)] for idx in self.matching_ids[1:]]
            daytime_relative_poses = torch.stack(daytime_relative_poses, 1)

            daytime_features, daytime_lowest_cost, daytime_confidence_mask, daytime_cost_volume, daytime_full_cv, daytime_flow_cv  = self.models["daytime_encoder"](inputs["color", 0, 0],
                                                                            daytime_lookup_frames,
                                                                            inputs['weather'],
                                                                            daytime_relative_poses,
                                                                            inputs[('K', 2)],
                                                                            inputs[('inv_K', 2)],
                                                                            min_depth_bin=min_depth_bin,
                                                                            max_depth_bin=max_depth_bin,
                                                                            inv_poses=torch.inverse(daytime_relative_poses), 
                                                                            high_level_K=inputs[('K', 3)], 
                                                                            high_lvel_invK=inputs[('inv_K', 3)])
                                                                            
            daytime_model_multi_disps = self.models["daytime_depth"](daytime_features)

            teacher_pose_pred = self.predict_poses_teacher(inputs, None)
            teacher_lookup_frames = [inputs[('color', idx, 0)] for idx in self.matching_ids[1:]]
            teacher_lookup_frames = torch.stack(teacher_lookup_frames, 1)

            teacher_relative_poses = [teacher_pose_pred[('relative_pose', idx)] for idx in self.matching_ids[1:]]
            teacher_relative_poses = torch.stack(teacher_relative_poses, 1)


            teacher_features, teacher_lowest_cost, teacher_confidence_mask, teacher_cost_volume, teacher_full_cv, teacher_flow_cv = self.models["teacher_encoder"](inputs["color", 0, 0],
                                                                            teacher_lookup_frames,
                                                                            inputs['weather'],
                                                                            teacher_relative_poses,
                                                                            inputs[('K', 2)],
                                                                            inputs[('inv_K', 2)],
                                                                            min_depth_bin=min_depth_bin,
                                                                            max_depth_bin=max_depth_bin,
                                                                            inv_poses=torch.inverse(teacher_relative_poses), 
                                                                            high_level_K=inputs[('K', 3)], 
                                                                            high_lvel_invK=inputs[('inv_K', 3)])
                                                                            
            teacher_multi_disps = self.models["teacher_depth"](teacher_features)



        teacher_outputs.update(teacher_pose_pred)
        
        teacher_outputs.update(teacher_multi_disps)

        for key in list(teacher_outputs.keys()):
            _key = list(key)
            if _key[0] in ['relative_pose', 'disp']:
                _key[0] = 'teacher_' + key[0]
                _key = tuple(_key)
                outputs[_key] = teacher_outputs[key]

        #student model (pose+depth)
        pose_pred = self.predict_poses_student(inputs, None)
        outputs.update(pose_pred)

        relative_poses = [pose_pred[('relative_pose', idx)].clone().detach() for idx in self.matching_ids[1:]]
        # relative_poses = [pose_pred[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)
        relative_poses_clone= relative_poses.clone()

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        
        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        if is_train and not self.opt.no_matching_augmentation:
            for batch_idx in range(batch_size):
                
                # if 'day-clear' in inputs['weather_depth'][batch_idx]:
                    rand_num = random.random()
                    # # static camera augmentation -> overwrite lookup frames with current frame
                    if rand_num < 0.25:
                   
                        replace_frames = \
                            [inputs[('color_aug', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                        replace_frames = torch.stack(replace_frames, 0)
                        lookup_frames[batch_idx] = replace_frames
                        augmentation_mask[batch_idx] += 1
                    # missing cost volume augmentation -> set all poses to 0, the cost volume will
                    # skip these frames
                    elif rand_num < 0.5:
                   
                        relative_poses[batch_idx] *= 0
                        augmentation_mask[batch_idx] += 1
        


        outputs['augmentation_mask'] = augmentation_mask
        
        # multi frame path
        rand_num_2 = random.random()
        if rand_num_2<0.5:
            batch_size, num_frames, chns, height, width = lookup_frames.shape
            lookup_frames = lookup_frames.reshape(batch_size * num_frames, chns, height, width)
            lookup_frames = flip_lr(lookup_frames)
            lookup_frames = lookup_frames.reshape(batch_size, num_frames, chns, height, width)
            inputs[("color_aug", 0, 0)]=flip_lr(inputs[("color_aug", 0, 0)])

            inputs[('K', 2)][:,0,2]=width//4-1-inputs[('K', 2)][:,0,2]
            inputs[('inv_K', 2)] = torch.inverse(inputs[('K', 2)])

            inputs[('K', 3)][:,0,2]=width//8-1-inputs[('K', 3)][:,0,2]
            inputs[('inv_K', 3)] = torch.inverse(inputs[('K', 3)])

        
        features, lowest_cost, confidence_mask, cost_volume, full_cv , flow_cv= self.models["encoder"](inputs["color_aug", 0, 0],
                                                                        lookup_frames,
                                                                        inputs['weather_depth'],
                                                                        relative_poses,
                                                                        inputs[('K', 2)],
                                                                        inputs[('inv_K', 2)],
                                                                        min_depth_bin=min_depth_bin,
                                                                        max_depth_bin=max_depth_bin,
                                                                        inv_poses=torch.inverse(relative_poses_clone), 
                                                                        high_level_K=inputs[('K', 3)], 
                                                                        high_lvel_invK=inputs[('inv_K', 3)])
                                                                        
        multi_disps=self.models["depth"](features)

        lowest_cost = lowest_cost.unsqueeze(1)
        confidence_mask = confidence_mask.unsqueeze(1)
        teacher_confidence_mask =teacher_confidence_mask.unsqueeze(1)
        if rand_num_2<0.5:   
            for i in range(5):
                features[i]=flip_lr(features[i])         
            lowest_cost = flip_lr(lowest_cost)
            cost_volume = flip_lr(cost_volume)
            full_cv = flip_lr(full_cv)
            flow_cv = flip_lr(flow_cv)
            confidence_mask = flip_lr(confidence_mask)
            inputs[("color_aug", 0, 0)]=flip_lr(inputs[("color_aug", 0, 0)])
            multi_disps = {scale: flip_lr(disp) for scale, disp in multi_disps.items()}

            inputs[('K', 2)][:,0,2]=width//4-1-inputs[('K', 2)][:,0,2]
            inputs[('inv_K', 2)] = torch.inverse(inputs[('K', 2)])

            inputs[('K', 3)][:,0,2]=width//8-1-inputs[('K', 3)][:,0,2]
            inputs[('inv_K', 3)] = torch.inverse(inputs[('K', 3)])
        outputs.update(multi_disps)


        outputs["lowest_cost"] = F.interpolate(lowest_cost,
                                               [self.opt.resized_height, self.opt.resized_width],
                                               mode="nearest")[:, 0]
        
        outputs["teacher_lowest_cost"] = F.interpolate(teacher_lowest_cost.unsqueeze(1),
                                               [self.opt.resized_height, self.opt.resized_width],
                                               mode="nearest")[:, 0]
        
        outputs["consistency_mask"] = F.interpolate(confidence_mask,
                                                    [self.opt.resized_height, self.opt.resized_width],
                                                    mode="nearest")[:, 0]
        
        outputs["confidence_mask"] = F.interpolate(confidence_mask,
                                                    [self.opt.resized_height, self.opt.resized_width],
                                                    mode="nearest")[:, 0]
        outputs["teacher_confidence_mask"] = F.interpolate(teacher_confidence_mask,
                                                    [self.opt.resized_height, self.opt.resized_width],
                                                    mode="nearest")[:, 0]


        losses = self.compute_TS_losses(outputs, teacher_outputs,daytime_model_multi_disps, cost_volume, teacher_cost_volume, augmentation_mask, confidence_mask, teacher_confidence_mask,
                                         full_cv, teacher_full_cv, flow_cv, teacher_flow_cv )

        _, depth = disp_to_depth(multi_disps[('disp', 0)], self.opt.min_depth, self.opt.max_depth)
        outputs[("depth", 0, 0)] = depth

        distribution_student = self.differentiable_histogram(depth, num_bins=101, min_depth=3.5, max_depth=80)
        kl_div = 0.01*F.kl_div(distribution_student.log(), self.distribution_daytime, reduction='mean')
        align_loss = kl_div
        
        losses['align_loss'] = align_loss
        losses['loss'] += align_loss

        return outputs, losses
    
    def process_batch_val(self, inputs, is_train=False, val_multi=False, val_teacher=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if torch.is_tensor(ipt):
                inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        inputs[("color_aug_flip", 0, 0)]=flip_lr(inputs[("color_aug", 0, 0)])

        # single frame path
        if val_teacher:
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
            mono_disps = self.models['mono_depth'](feats)
            mono_disp, _ = disp_to_depth(mono_disps[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
              
            feats = self.models["mono_encoder"](inputs["color_aug_flip", 0, 0])
            mono_disps_flip = self.models['mono_depth'](feats)
            mono_disp_flip, _ = disp_to_depth(mono_disps_flip[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            mono_disp_pp = post_process_inv_depth(mono_disp, mono_disp_flip, method='mean')

            mono_depth = inv2depth(mono_disp)
            mono_depth_pp = inv2depth(mono_disp_pp)
            outputs["depth"]=mono_depth
            outputs["depth_pp"] = mono_depth_pp

        # multi frame path
        elif val_multi:
             # predict poses for all frames
            pose_pred = self.predict_poses(inputs, None)
            outputs.update(pose_pred)
            mono_outputs.update(pose_pred)

            # grab poses + frames and stack for input to the multi frame network
            relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
            relative_poses = torch.stack(relative_poses, 1)
            # relative_poses *= 0
            lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
            lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w
            
            # apply static frame and zero cost volume augmentation
            batch_size = len(lookup_frames)
            augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()

            outputs['augmentation_mask'] = augmentation_mask

            min_depth_bin = self.min_depth_tracker
            max_depth_bin = self.max_depth_tracker

            features, lowest_cost, confidence_mask, costvolume, full_cv, flow_cv_teacher = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                            lookup_frames,
                                                                            inputs['weather_depth'],
                                                                            relative_poses,
                                                                            inputs[('K', 2)],
                                                                            inputs[('inv_K', 2)],
                                                                            min_depth_bin=min_depth_bin,
                                                                            max_depth_bin=max_depth_bin,
                                                                            inv_poses=torch.inverse(relative_poses), 
                                                                            high_level_K=inputs[('K', 3)], 
                                                                            high_lvel_invK=inputs[('inv_K', 3)])
                                                                           
            multi_disps=self.models["depth"](features)
            multi_disp, _ = disp_to_depth(multi_disps[("disp", 0)], self.opt.min_depth, self.opt.max_depth)


            batch_size, num_frames, chns, height, width = lookup_frames.shape
            lookup_frames = lookup_frames.reshape(batch_size * num_frames, chns, height, width)
            lookup_frames = flip_lr(lookup_frames)
            lookup_frames = lookup_frames.reshape(batch_size, num_frames, chns, height, width)
            inputs[("color_aug", 0, 0)]=flip_lr(inputs[("color_aug", 0, 0)])

            inputs[('K', 2)][:,0,2]=width//4-1-inputs[('K', 2)][:,0,2]
            inputs[('inv_K', 2)] = torch.inverse(inputs[('K', 2)])

            features, lowest_cost_gun, confidence_mask, cost_volume,full_cv, flow_cv = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                            lookup_frames,
                                                                            inputs['weather_depth'],
                                                                            relative_poses,
                                                                            inputs[('K', 2)],
                                                                            inputs[('inv_K', 2)],
                                                                            min_depth_bin=min_depth_bin,
                                                                            max_depth_bin=max_depth_bin,
                                                                            inv_poses=torch.inverse(relative_poses), 
                                                                            high_level_K=inputs[('K', 3)], 
                                                                            high_lvel_invK=inputs[('inv_K', 3)])
            
            multi_disps_flip=self.models["depth"](features)
            lowest_cost = flip_lr(lowest_cost.unsqueeze(1))
            cost_volume = flip_lr(cost_volume)
            flow_cv = flip_lr(flow_cv)
            confidence_mask = flip_lr(confidence_mask.unsqueeze(1))
            inputs[("color_aug", 0, 0)]=flip_lr(inputs[("color_aug", 0, 0)])
            
            multi_disp_flip, _ = disp_to_depth(multi_disps_flip[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            multi_disp_pp = post_process_inv_depth(multi_disp, multi_disp_flip, method='mean')
            
            multi_depth = inv2depth(multi_disp)
            multi_depth_pp = inv2depth(multi_disp_pp)
            outputs[('disp', 0, 0)] = multi_disp
            outputs["depth"]=multi_depth
            outputs["depth_pp"] = multi_depth_pp
            outputs["lowest_cost"] = F.interpolate(lowest_cost,
                                               [self.opt.resized_height, self.opt.resized_width],
                                               mode="nearest")[:, 0]
            

          

        return outputs

    def process_batch_val_single(self, inputs, is_train=False, val_multi=False, val_teacher=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if torch.is_tensor(ipt):
                inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        inputs[("color_aug_flip", 0, 0)]=flip_lr(inputs[("color_aug", 0, 0)])

        # single frame path
        if val_teacher:
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
            mono_disps = self.models['mono_depth'](feats)
            mono_disp, _ = disp_to_depth(mono_disps[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
              
            feats = self.models["mono_encoder"](inputs["color_aug_flip", 0, 0])
            mono_disps_flip = self.models['mono_depth'](feats)
            mono_disp_flip, _ = disp_to_depth(mono_disps_flip[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            mono_disp_pp = post_process_inv_depth(mono_disp, mono_disp_flip, method='mean')

            mono_depth = inv2depth(mono_disp)
            mono_depth_pp = inv2depth(mono_disp_pp)
            outputs["depth"]=mono_depth
            outputs["depth_pp"] = mono_depth_pp

        # multi frame path
        elif val_multi:
             # predict poses for all frames
            pose_pred = self.predict_poses_single(inputs, None)
            outputs.update(pose_pred)
            mono_outputs.update(pose_pred)

            # grab poses + frames and stack for input to the multi frame network
            relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
            relative_poses = torch.stack(relative_poses, 1)
            # relative_poses *= 0
            lookup_frames = [inputs[('color_aug', 0, 0)]]
            lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w
            
            # apply static frame and zero cost volume augmentation
            batch_size = len(lookup_frames)
            augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()

            outputs['augmentation_mask'] = augmentation_mask

            min_depth_bin = self.min_depth_tracker
            max_depth_bin = self.max_depth_tracker

            features, lowest_cost, confidence_mask, costvolume, full_cv, flow_cv = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                            lookup_frames,
                                                                            inputs['weather_depth'],
                                                                            relative_poses,
                                                                            inputs[('K', 2)],
                                                                            inputs[('inv_K', 2)],
                                                                            min_depth_bin=min_depth_bin,
                                                                            max_depth_bin=max_depth_bin,
                                                                            inv_poses=torch.inverse(relative_poses), 
                                                                            high_level_K=inputs[('K', 3)], 
                                                                            high_lvel_invK=inputs[('inv_K', 3)])
                                                                           
            multi_disps=self.models["depth"](features)
            multi_disp, _ = disp_to_depth(multi_disps[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

        
            

            batch_size, num_frames, chns, height, width = lookup_frames.shape
            lookup_frames = lookup_frames.reshape(batch_size * num_frames, chns, height, width)
            lookup_frames = flip_lr(lookup_frames)
            lookup_frames = lookup_frames.reshape(batch_size, num_frames, chns, height, width)
            inputs[("color_aug", 0, 0)]=flip_lr(inputs[("color_aug", 0, 0)])

            inputs[('K', 2)][:,0,2]=width//4-1-inputs[('K', 2)][:,0,2]
            inputs[('inv_K', 2)] = torch.inverse(inputs[('K', 2)])

            features, lowest_cost_gun, confidence_mask, cost_volume,full_cv, flow_cv = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                            lookup_frames,
                                                                            inputs['weather_depth'],
                                                                            relative_poses,
                                                                            inputs[('K', 2)],
                                                                            inputs[('inv_K', 2)],
                                                                            min_depth_bin=min_depth_bin,
                                                                            max_depth_bin=max_depth_bin,
                                                                            inv_poses=torch.inverse(relative_poses), 
                                                                            high_level_K=inputs[('K', 3)], 
                                                                            high_lvel_invK=inputs[('inv_K', 3)])
            
            multi_disps_flip=self.models["depth"](features)
            lowest_cost = flip_lr(lowest_cost.unsqueeze(1))
            cost_volume = flip_lr(cost_volume)
            confidence_mask = flip_lr(confidence_mask.unsqueeze(1))
            inputs[("color_aug", 0, 0)]=flip_lr(inputs[("color_aug", 0, 0)])
            
            multi_disp_flip, _ = disp_to_depth(multi_disps_flip[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            multi_disp_pp = post_process_inv_depth(multi_disp, multi_disp_flip, method='mean')
            
            multi_depth = inv2depth(multi_disp)
            multi_depth_pp = inv2depth(multi_disp_pp)
            outputs[('disp', 0, 0)] = multi_disp
            outputs["depth"]=multi_depth
            outputs["depth_pp"] = multi_depth_pp
            outputs["lowest_cost"] = F.interpolate(lowest_cost,
                                               [self.opt.resized_height, self.opt.resized_width],
                                               mode="nearest")[:, 0]
          

        return outputs
    
    
    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            
            pose_feats = {f_i: inputs["color_aug_pose", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        if self.opt.maintain_temp:
                            pose_inputs = [pose_feats[f_i], pose_feats[0]]
                        else:
                            pose_inputs = [pose_feats[0], pose_feats[f_i]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs['weather_pose'])]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    if self.opt.maintain_temp:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    else:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

            # now we need poses for matching - compute without gradients
            pose_feats = {f_i: inputs["color_aug_pose", f_i, 0] for f_i in self.matching_ids}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        if self.opt.maintain_temp:
                            pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        else:
                            pose_inputs = [pose_feats[fi + 1], pose_feats[fi]]
                        
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs['weather_pose'])]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=self.opt.maintain_temp)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs['weather_pose'])]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0
                  
                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs

    def predict_poses_single(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            # pose_feats = {f_i: inputs["color_aug_pose", f_i, 0] for f_i in self.opt.frame_ids}
            pose_feats = {0: inputs["color_aug_pose", 0, 0], -1:inputs["color_aug_pose", 0, 0]}
            for f_i in self.opt.frame_ids[1:2]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        if self.opt.maintain_temp:
                            pose_inputs = [pose_feats[f_i], pose_feats[0]]
                        else:
                            pose_inputs = [pose_feats[0], pose_feats[f_i]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs['weather_pose'])]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    if self.opt.maintain_temp:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    else:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

            # now we need poses for matching - compute without gradients
            pose_feats = {0: inputs["color_aug_pose", 0, 0], -1:inputs["color_aug_pose", 0, 0]}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        if self.opt.maintain_temp:
                            pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        else:
                            pose_inputs = [pose_feats[fi + 1], pose_feats[fi]]
                        # print(pose_inputs)
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs['weather_pose'])]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=self.opt.maintain_temp)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs['weather_pose'])]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0
                 
                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs
    
    def predict_poses_daytime_model(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        
            # now we need poses for matching - compute without gradients
        pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.matching_ids}
        with torch.no_grad():
            # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
            for fi in self.matching_ids[1:]:
                if fi < 0:
                    if self.opt.maintain_temp:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                    else:
                        pose_inputs = [pose_feats[fi + 1], pose_feats[fi]]
                    pose_inputs = [self.models["daytime_pose_encoder"](torch.cat(pose_inputs, 1), inputs['weather'])]
                    axisangle, translation = self.models["daytime_pose"](pose_inputs)
                    
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=self.opt.maintain_temp)

                 

                else:
                    pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                    pose_inputs = [self.models["daytime_pose_encoder"](torch.cat(pose_inputs, 1), inputs['weather'])]
                    axisangle, translation = self.models["daytime_pose"](pose_inputs)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)

                   

                # set missing images to 0 pose
                for batch_idx, feat in enumerate(pose_feats[fi]):
                    if feat.sum() == 0:
                        pose[batch_idx] *= 0

                outputs[('relative_pose', fi)] = pose
                outputs[('axisangle', fi)] =axisangle
                outputs[('translation', fi)] = translation
        
        return outputs
    
    def predict_poses_teacher(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        
            # now we need poses for matching - compute without gradients
        pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.matching_ids}
        with torch.no_grad():
            # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
            for fi in self.matching_ids[1:]:
                if fi < 0:
                    if self.opt.maintain_temp:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                    else:
                        pose_inputs = [pose_feats[fi + 1], pose_feats[fi]]
                    pose_inputs = [self.models["teacher_pose_encoder"](torch.cat(pose_inputs, 1), inputs['weather'])]
                    axisangle, translation = self.models["teacher_pose"](pose_inputs)
                    
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=self.opt.maintain_temp)

               

                else:
                    pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                    pose_inputs = [self.models["teacher_pose_encoder"](torch.cat(pose_inputs, 1), inputs['weather'])]
                    axisangle, translation = self.models["teacher_pose"](pose_inputs)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)


                # set missing images to 0 pose
                for batch_idx, feat in enumerate(pose_feats[fi]):
                    if feat.sum() == 0:
                        pose[batch_idx] *= 0

                outputs[('relative_pose', fi)] = pose
                outputs[('axisangle', fi)] =axisangle
                outputs[('translation', fi)] = translation
        
        return outputs
    
    def predict_poses_student(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        
            # now we need poses for matching - compute without gradients
        pose_feats = {f_i: inputs["color_aug_pose", f_i, 0] for f_i in self.matching_ids}
        # with torch.no_grad():
        # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
        for fi in self.matching_ids[1:]:
            if fi < 0:
                if self.opt.maintain_temp:
                    pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                else:
                    pose_inputs = [pose_feats[fi + 1], pose_feats[fi]]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs['weather_pose'])]
                axisangle, translation = self.models["pose"](pose_inputs)
                
                pose = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=self.opt.maintain_temp)


            else:
                pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs['weather_pose'])]
                axisangle, translation = self.models["pose"](pose_inputs)
                pose = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=False)

            # set missing images to 0 pose
            for batch_idx, feat in enumerate(pose_feats[fi]):
                if feat.sum() == 0:
                    pose[batch_idx] *= 0

            outputs[('relative_pose', fi)] = pose
            outputs[('axisangle', fi)] =axisangle
            outputs[('translation', fi)] = translation
        
        return outputs
    
    


    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.resized_height, self.opt.resized_width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]
    
    def compute_TS_losses(self, outputs, teacher_outputs, daytime_model_disps,cost_volume, teacher_cost_volume, augmentation_mask, confidence_mask, teacher_confidence_mask, 
                          student_full_cv, teacher_full_cv,flow_cv, teacher_flow_cv):


        # print(teacher_confidence_mask.shape)
        losses = {}
        total_loss = 0

        for fi in self.matching_ids[1:]:

            pose = outputs[('relative_pose', fi)]
            # print("pose={}".format(pose[0]))
            axisangle = outputs[('axisangle', fi)]
            translation = outputs[('translation', fi)]

            teacher_pose = teacher_outputs[('relative_pose', fi)]
            # print("teacher_pose={}".format(teacher_pose[0]))
            teacher_axisangle = teacher_outputs[('axisangle', fi)].detach()
            teacher_translation = teacher_outputs[('translation', fi)].detach()

            trans_loss = F.mse_loss(translation, teacher_translation)
            rot_loss = F.mse_loss(axisangle, teacher_axisangle)

            total_loss=total_loss + trans_loss
            total_loss=total_loss + rot_loss

            losses['trans_loss'] = trans_loss
            losses['rot_loss'] = rot_loss

        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            teacher_disp = teacher_outputs[("disp", scale)]
            daytime_model_disp = daytime_model_disps[("disp", scale)]
        

            disp, _ = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            teacher_disp, teacher_depth = disp_to_depth(teacher_disp, self.opt.min_depth, self.opt.max_depth)
            daytime_model_disp, daytime_model_depth = disp_to_depth(daytime_model_disp, self.opt.min_depth, self.opt.max_depth)

            confidence_map = torch.exp(-torch.abs((teacher_disp-daytime_model_disp)/teacher_disp))+1.0
            consistency_loss = (confidence_map*torch.abs((disp - teacher_disp)/disp)).mean()
            losses['consistency_loss/{}'.format(scale)] = consistency_loss
            total_loss = total_loss + consistency_loss/len(self.opt.scales)

         

        depth_cv_loss = self.compute_L1loss_with_mask(cost_volume,teacher_cost_volume, confidence_mask, augmentation_mask,teacher_confidence_mask)
        flow_cv_loss = 0.
     
        losses['depth_cv_loss'] = depth_cv_loss
        losses['flow_cv_loss'] = flow_cv_loss
        total_loss = total_loss + depth_cv_loss + flow_cv_loss
      
        losses['loss'] = total_loss
        
        return losses
    
    def compute_L1loss_with_mask(self, cost_volume, teacher_cost_volume, mask, augmentation_mask, teacher_mask):

        cv_diff = torch.abs(cost_volume-teacher_cost_volume)
        aug_mask = torch.ones_like(cv_diff)
        cv_diff = cv_diff*aug_mask
        L1loss = cv_diff.sum()/(aug_mask.sum())

        return L1loss


    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            
            for frame_id in self.opt.frame_ids[1:]:
        
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                if not self.opt.disable_motion_masking:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              outputs['consistency_mask'].unsqueeze(1))
                if not self.opt.no_matching_augmentation:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              (1 - outputs['augmentation_mask']))
                consistency_mask = (1 - reprojection_loss_mask).float()

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                if self.opt.consistency_loss == "L1":
                    consistency_loss = torch.abs((multi_depth - mono_depth)) * consistency_mask
                elif self.opt.consistency_loss == "absrel":
                    consistency_loss = torch.abs((multi_depth - mono_depth)/mono_depth) * consistency_mask
                consistency_loss = consistency_loss.mean()

                # save for logging to tensorboard
                consistency_target = (mono_depth.detach() * consistency_mask +
                                      multi_depth.detach() * (1 - consistency_mask))
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)] = consistency_target
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss + self.opt.consistency_weight*consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        losses["velocity_loss"] = 0
        if self.train_teacher_and_pose and self.opt.use_velocity:
            for frame_id in self.opt.frame_ids[1:]:
                    pred_trans = outputs[("cam_T_cam", 0, frame_id)][:,:3,3].norm(dim=-1)
                    gt_trans = inputs[('pose_gt', frame_id)][:,:3,3].norm(dim=-1)
                    velocity_loss = torch.mean(torch.abs(pred_trans-gt_trans))
                    losses["velocity_loss"] += velocity_loss

        total_loss += self.opt.velocity_weight * losses["velocity_loss"]

        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [900, 1600], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        _, _, height, width = depth_gt.shape
        # print(depth_gt.shape)
        # print(0)
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)


        if self.opt.save_absrel:
            
            abs_rel_map = torch.abs((depth_pred-depth_gt)/depth_gt)
            abs_rel_map = torch.where(torch.isnan(abs_rel_map), 0.0, abs_rel_map)
            outputs["abs_rel_map"] = abs_rel_map

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred_ms = depth_pred*torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        depth_pred_ms = torch.clamp(depth_pred_ms, min=1e-3, max=80)
        # print(depth_gt.shape)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        depth_errors_ms = compute_depth_errors(depth_gt, depth_pred_ms)

        
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())
        for i, metric in enumerate(self.depth_ms_metric_names):
            losses[metric] = np.array(depth_errors_ms[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                writer.add_image(
                    "color_aug_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color_aug", frame_id, s)][j].data, self.step)
                writer.add_image(
                    "color_aug_pose_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color_aug_pose", frame_id, s)][j].data, self.step)
         

            disp = colormap(outputs[("disp", s)][j, 0])
            writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)

            disp = colormap(outputs[('teacher_disp', s)][j, 0])
            writer.add_image(
                "teacher_disp/{}".format(j),
                disp, self.step)
            
          
            if outputs.get("lowest_cost") is not None:
                lowest_cost = outputs["lowest_cost"][j]

            
                
                confidence_mask = \
                    outputs['confidence_mask'][j].cpu().detach().unsqueeze(0).numpy()
                
                teacher_confidence_mask = \
                    outputs['teacher_confidence_mask'][j].cpu().detach().unsqueeze(0).numpy()


                min_val = np.percentile(lowest_cost.numpy(), 10)
                max_val = np.percentile(lowest_cost.numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)

                writer.add_image(
                    "lowest_cost/{}".format(j),
                    lowest_cost, self.step)
                
            if outputs.get("teacher_lowest_cost") is not None:
                lowest_cost = outputs["teacher_lowest_cost"][j]

                min_val = np.percentile(lowest_cost.numpy(), 10)
                max_val = np.percentile(lowest_cost.numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)

                writer.add_image(
                    "teacher_lowest_cost/{}".format(j),
                    lowest_cost, self.step)
             
                writer.add_image(
                    "confidence_mask/{}".format(j),
                    confidence_mask, self.step)
                writer.add_image(
                    "teacher_confidence_mask/{}".format(j),
                    teacher_confidence_mask, self.step)


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if save_step:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                       self.step))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.resized_height
                to_save['width'] = self.opt.resized_width
                # save estimates of depth bins
                to_save['min_depth_bin'] = self.min_depth_tracker
                to_save['max_depth_bin'] = self.max_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_mono_model(self):

        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
    
    def load_daytime_model(self):
        """Load model(s) from disk
        """
        self.opt.load_daytime_weights_folder = os.path.expanduser(self.opt.load_daytime_weights_folder)

        assert os.path.isdir(self.opt.load_daytime_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_daytime_weights_folder)
        print("loading model from folder {}".format(self.opt.load_daytime_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_daytime_weights_folder, "{}.pth".format(n))
            model_dict = self.models['daytime_'+ n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['daytime_encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models['daytime_'+n].load_state_dict(model_dict)

    def load_teacher_model(self):
        """Load model(s) from disk
        """
        self.opt.load_teacher_weights_folder = os.path.expanduser(self.opt.load_teacher_weights_folder)

        assert os.path.isdir(self.opt.load_teacher_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_teacher_weights_folder)
        print("loading model from folder {}".format(self.opt.load_teacher_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_teacher_weights_folder, "{}.pth".format(n))
            model_dict = self.models['teacher_'+n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['teacher_encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models['teacher_'+n].load_state_dict(model_dict)

    def load_student_model(self):
        """Load model(s) from disk
        """
        self.opt.load_student_weights_folder = os.path.expanduser(self.opt.load_student_weights_folder)

        assert os.path.isdir(self.opt.load_student_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_student_weights_folder)
        print("loading model from folder {}".format(self.opt.load_student_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_student_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['teacher_encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)



def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
