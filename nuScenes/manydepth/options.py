# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ManyDepth options")
        # DDP
        self.parser.add_argument("--local_rank", 
                                 default=0, 
                                 type=int)  
        self.parser.add_argument('--world_size', 
                                 type=int, 
                                 default=2, 
                                 help='number of GPUs to use')
        self.parser.add_argument('--running_mode',
                                 help='train/val',
                                 default="train")
        # dataset
        self.parser.add_argument("--train_repeat",
                                 default=1)
        self.parser.add_argument("--deterministic",
                                 default=False)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=320)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=576)
        self.parser.add_argument("--resized_height",
                                 type=int,
                                 help="resized height of imgs",
                                 default=320)
        self.parser.add_argument("--resized_width",
                                 type=int,
                                 help="resized width of imgs",
                                 default=576)
        self.parser.add_argument("--temp_context",
                                 default=[0, -1, 1])
        self.parser.add_argument("--original_height",
                                 type = int,
                                 default=900)
        self.parser.add_argument("--original_width",
                                 type = int,
                                 default=1600)
        self.parser.add_argument("--brightness",
                                 default=[0.8, 1.2])
        self.parser.add_argument("--contrast",
                                 default=[0.8, 1.2])
        self.parser.add_argument("--saturation",
                                 default=[0.8, 1.2])
        self.parser.add_argument("--hue",
                                 default=[-0.05, 0.05])
        self.parser.add_argument("--color_jitter_prob",
                                 default=1.0)
        self.parser.add_argument("--color_jitter_keys",
                                 default=['color_aug', 'color_aug_pose'])
        self.parser.add_argument("--gaussian_noise_rand_min",
                                 default=0.005)
        self.parser.add_argument("--gaussian_noise_rand_max",
                                 default=0.05)
        self.parser.add_argument("--gaussian_noise_prob",
                                 default=0.5)
        self.parser.add_argument("--gaussian_noise_keys",
                                 default=['color_aug', 'color_aug_pose'])
        self.parser.add_argument("--gaussian_blur_kernel_size",
                                 default=9)
        self.parser.add_argument("--gaussian_blur_sigma",
                                 default=2)
        self.parser.add_argument("--gaussian_blur_prob",
                                 default=0.0)
        self.parser.add_argument("--gaussian_blur_keys",
                                 default=['color_aug', 'color_aug_pose'])
        self.parser.add_argument("--day_night_translation_prob",
                                default=0.)
        self.parser.add_argument("--day_night_translation_path",
                                 default='/mnt/weilong/datasets/nuscenes/trainval/translated/night')
        self.parser.add_argument("--day_night_translation_direction",
                                 default='day->night')
        self.parser.add_argument("--day_night_translation_keys",
                                 default=['color_aug', 'color_aug_pose'])
        self.parser.add_argument("--day_night_translation_key_frame_only",
                                 default=False)
        self.parser.add_argument("--day_clear_day_rain_translation_prob",
                                 default=0.)
        self.parser.add_argument("--day_clear_day_rain_translation_path",
                                 default='/mnt/weilong/datasets/nuscenes/trainval/translated/rain')
        self.parser.add_argument("--day_clear_day_rain_translation_direction",
                                 default='day-clear->day-rain')
        self.parser.add_argument("--day_clear_day_rain_translation_keys",
                                 default=['color_aug', 'color_aug_pose'])
        self.parser.add_argument("--day_clear_day_rain_translation_key_frame_only",
                                 default=False)
        self.parser.add_argument("--evaluation_day_night_translation_enabled",
                                 default=False)
        self.parser.add_argument("--evaluation_day_clear_day_rain_translation_enabled",
                                 default=False)
        self.parser.add_argument("--use_flip_teacher",
                                 type=str2bool,
                                 default=True)
        self.parser.add_argument("--flip_p",
                                 default=0.5)
        self.parser.add_argument("--use_pp_teacher",
                         type=str2bool,
                         default=False)
        self.parser.add_argument("--use_flip_multi",
                         type=str2bool,
                         default=False)

        
        
        self.parser.add_argument("--drop_static",
                                default=True)
        self.parser.add_argument("--orientations",
                                 default=['front'])
     #    self.parser.add_argument("--scales",
     #                             default=[0, 1, 2, 3])
        self.parser.add_argument("--weather_train",
                              #    default=['night-clear','night-rain'])
                                 default=['day-clear', 'day-rain', 'night-clear', 'night-rain'])
        self.parser.add_argument("--weather_train_real",
                                 default=['day-clear'])
                                   # default=['day-rain', 'night-clear', 'night-rain'])
                              # default=['night-clear'])
        self.parser.add_argument("--weather_val",
                              #    default=['night-clear'])
                                 default=['day-clear', 'day-rain', 'night-clear', 'night-rain'])
        self.parser.add_argument("--weather_test",
                                 default=['day-clear', 'day-rain', 'night-clear', 'night-rain'])
        self.parser.add_argument("--load_gt_depth",
                                 default=True)
        self.parser.add_argument("--load_gt_pose",
                                 default=True)
        self.parser.add_argument("--color_full_size",
                                 default=False)
        self.parser.add_argument("--evaluation_qualitative_res_path",
                              #    default=False)
                              default='')
        self.parser.add_argument("--evaluation_visualization_set",
                                 default=False)
        self.parser.add_argument("--save_rgb",
                                 default=False)
        self.parser.add_argument("--save_depth_pred",
                                 default=True)
        self.parser.add_argument("--save_depth_gt",
                                 default=False)
        self.parser.add_argument("--save_lowest_cost",
                                 default=False)
   

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark",
                                          "cityscapes_preprocessed"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--depth_binning", 
                                 help="defines how the depth bins are constructed for the cost"
                                      "volume. 'linear' is uniformly sampled in depth space,"
                                      "'inverse' is uniformly sampled in inverse depth space",
                                 type=str,
                                 choices=['linear', 'inverse','sid'],
                                 default='sid'),
        self.parser.add_argument("--num_depth_bins",
                                 type=int,
                                 default=128)
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
                                          "cityscapes_preprocessed"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=80.0)
        

        self.parser.add_argument("--update_bins", 
                                 type=str2bool, 
                                 default=False, 
                    help="Update bins flag (default: True)")
        self.parser.add_argument("--min_depth_tracker",
                                 type=float,
                                 help="min depth tracker",
                                 default=3.5)
        self.parser.add_argument("--max_depth_tracker",
                                 type=float,
                                 help="maximum depth trakcer",
                                 default=80.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--maintain_temp",
                                 type=str2bool,
                                 default=True
                                 )

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--use_velocity",
                                 help="use velocity to be scale-aware",
                                 default=False)
        self.parser.add_argument("--velocity_weight",
                                 type=float,
                                 help="velocity loss weight",
                                 default=0.02)
        self.parser.add_argument("--consistency_weight",
                                 type=float,
                                 help="batch size",
                                 default=0.1)
        self.parser.add_argument("--consistency_loss",
                                 type=str,
                                 help="L1/absrel",
                                 default="absrel")
        self.parser.add_argument("--learning_rate",
                                   type=float,
                                   help="learning rate",
                                   default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=5)
        self.parser.add_argument("--scheduler_w",
                                   type = float,
                                   help="step size of the scheduler",
                                   default=0.5)
        self.parser.add_argument("--freeze_teacher_and_pose",
                                 action="store_true",
                                 help="If set, freeze the weights of the single frame teacher"
                                      " network and pose network.")
        self.parser.add_argument("--freeze_teacher_epoch",
                                 type=int,
                                 default=20,
                                 help="Sets the epoch number at which to freeze the teacher"
                                      "network and the pose network.")
        self.parser.add_argument("--freeze_pose_epoch",
                                 type=int,
                                 default=20,
                                 help="Sets the epoch number at which to freeze the teacher"
                                      "network and the pose network.")
        self.parser.add_argument("--freeze_teacher_step",
                                 type=int,
                                 default=-1,
                                 help="Sets the step number at which to freeze the teacher"
                                      "network and the pose network. By default is -1 and so"
                                      "will not be used.")
        self.parser.add_argument("--pytorch_random_seed",
                                 default=1,
                                 type=int)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument('--use_future_frame',
                                 action='store_true',
                                 help='If set, will also use a future frame in time for matching.')
        self.parser.add_argument('--num_matching_frames',
                                 help='Sets how many previous frames to load to build the cost'
                                      'volume',
                                 type=int,
                                 default=1)
        self.parser.add_argument("--disable_motion_masking",
                                 help="If set, will not apply consistency loss in regions where"
                                      "the cost volume is deemed untrustworthy",
                                 action="store_true")
        self.parser.add_argument("--no_matching_augmentation",
                                 action='store_true',
                                 help="If set, will not apply static camera augmentation or "
                                      "zero cost volume augmentation during training")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
    
        self.parser.add_argument("--load_daytime_weights_folder",
                                 type=str,
                                 help="name of teacher model to load",)
                         
        self.parser.add_argument("--load_teacher_weights_folder",
                                 type=str,
                                 help="name of teacher model to load",)
                             
        self.parser.add_argument("--load_student_weights_folder",
                                 type=str,
                                 help="name of student model to load",
                                 default="/mnt/codemnt/weilong/syn2real_depth/checkpoints/nuScenes")
        self.parser.add_argument("--mono_weights_folder",
                                 type=str)
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--eval_teacher_interval",
                                 type=int,
                                 default=20)
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--save_intermediate_models",
                                 help="if set, save the model each time we log to tensorboard",
                                 action='store_true')
        self.parser.add_argument("--save_absrel",
                                 type=str2bool,
                                 default=False)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=["eigen", "eigen_benchmark", "benchmark", "odom_9",
                                          "odom_10", "cityscapes"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--zero_cost_volume",
                                 action="store_true",
                                 help="If set, during evaluation all poses will be set to 0, and "
                                      "so we will evaluate the model in single frame mode")
        self.parser.add_argument('--static_camera',
                                 action='store_true',
                                 help='If set, during evaluation the current frame will also be'
                                      'used as the lookup frame, to simulate a static camera')
        self.parser.add_argument('--eval_teacher',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')


         #set up for marigold DDPM
        self.parser.add_argument("--config",
                                   type=str,
                                   default="/mnt/weilong/manydepth4all_marigold_regularize/manydepth/config/train_marigold.yaml",
                                   help="Path to config file.",
                              )
        self.parser.add_argument(
               "--base_ckpt_dir",
               type=str,
               default="/mnt/weilong/Marigold/",
               help="directory of pretrained checkpoint",
          )
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')