# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from manydepth.datasets.nuscenes_dataset import NuScenesDataset, Subset, NuScenesDataSubset
from .utils import readlines, inv2depth
from .options import MonodepthOptions
from manydepth import datasets, networks
from .layers import transformation_from_parameters, disp_to_depth
import tqdm

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # Setup dataloaders
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    if opt.eval_teacher:
        encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
        encoder_class = networks.ResnetEncoder

    else:
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_class = networks.ResnetEncoderMatching

    encoder_dict = torch.load(encoder_path)
    try:
        HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
    except KeyError:
        print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                'using command line values!')
        HEIGHT, WIDTH = opt.height, opt.width


    to_tensor = datasets.transforms.ToTensor(sample_keys=['color', 'color_aug', 'color_aug_pose'],
                                            temp_context=opt.temp_context, scales=opt.scales)
    val_transform = [to_tensor]
    transform = {
                    'val': datasets.transforms.CustomCompose(val_transform)
                    }
    nuscenes_dataset = NuScenesDataset(opt, transform, 'trainval')
    nuscenes_subset_val = NuScenesDataSubset(nuscenes_dataset=nuscenes_dataset, mode='val')
    val_dataset = Subset(nuscenes_subset_val, nuscenes_subset_val.indices)
    dataloader = DataLoader(
        val_dataset, opt.batch_size, True, 
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    # setup models
    if opt.eval_teacher:
        encoder_opts = dict(num_layers=opt.num_layers,
                            pretrained=False)
    else:
        encoder_opts = dict(num_layers=opt.num_layers,
                            pretrained=False,
                            input_width=encoder_dict['width'],
                            input_height=encoder_dict['height'],
                            adaptive_bins=True,
                            min_depth_bin=0.1, max_depth_bin=20.0,
                            depth_binning=opt.depth_binning,
                            num_depth_bins=opt.num_depth_bins)
        pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
        pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))

        pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)

        pose_enc.load_state_dict(pose_enc_dict, strict=True)
        pose_dec.load_state_dict(pose_dec_dict, strict=True)

        min_depth_bin = encoder_dict.get('min_depth_bin')
        max_depth_bin = encoder_dict.get('max_depth_bin')

        pose_enc.eval()
        pose_dec.eval()

        if torch.cuda.is_available():
            pose_enc.cuda()
            pose_dec.cuda()

    encoder = encoder_class(**encoder_opts)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.eval()
    depth_decoder.eval()

    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()



    metrics_name = 'depth'
    metrics_keys = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
    # metrics_modes = ('', '_pp', '_gt', '_pp_gt')
    metrics_modes = ('',  '_gt', )

    metric_conditions = ('all-conditions', 'day', 'night', 'clear', 'rain', 'day-clear', 'day-rain',
                                'night-clear', 'night-rain')

    # Dictionary for metrics in different conditions
    metrics = OrderedDict({condition: {mode: {metric: 0.0 for metric in metrics_keys} for mode in self.metrics_modes} for condition in self.metric_conditions})
    for condition in metrics.keys():
        metrics[condition]['count'] = 0


    print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

    # do inference
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            input_color = data[('color_aug', 0, 0)]
            if torch.cuda.is_available():
                input_color = input_color.cuda()

            outputs = {}


            if opt.static_camera:
                for f_i in frames_to_load:
                    data["color", f_i, 0] = data[('color', 0, 0)]

            # predict poses
            pose_feats = {f_i: data["color_aug_pose", f_i, 0] for f_i in frames_to_load}
            if torch.cuda.is_available():
                pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
            # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
            for fi in frames_to_load[1:]:
                if fi < 0:
                    if opt.maintain_temp:
                        pose_inputs = [pose_feats[fi+1], pose_feats[fi]]
                    else:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                    pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                    axisangle, translation = pose_dec(pose_inputs)
                    if opt.maintain_temp:
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)
                    else:
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                    # now find 0->fi pose
                    if fi != -1:
                        pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                else:
                    pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                    pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                    axisangle, translation = pose_dec(pose_inputs)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)

                    # now find 0->fi pose
                    if fi != 1:
                        pose = torch.matmul(pose, data[('relative_pose', fi - 1)])

                data[('relative_pose', fi)] = pose

            lookup_frames = [data[('color_aug', idx, 0)] for idx in frames_to_load[1:]]
            lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

            relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
            relative_poses = torch.stack(relative_poses, 1)

            K = data[('K', 2)]  # quarter resolution for matching
            invK = data[('inv_K', 2)]

            if torch.cuda.is_available():
                lookup_frames = lookup_frames.cuda()
                relative_poses = relative_poses.cuda()
                K = K.cuda()
                invK = invK.cuda()

            if opt.zero_cost_volume:
                relative_poses *= 0


            output, lowest_cost, costvol = encoder(input_color, lookup_frames,
                                                    relative_poses,
                                                    K,
                                                    invK,
                                                    min_depth_bin, max_depth_bin,
                                                    use_flip=False,is_train=False,do_flip=False)
            output = depth_decoder(output)

           
            multi_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            multi_depth = inv2depth(multi_disp)
            outputs["depth"]=multi_depth

    depth = outputs["depth"]

    for i, weather in enumerate(inputs["weather"]):
                    gt_i = inputs["depth_gt"][i]
                    valid = (gt_i > self.opt.min_depth) & (gt_i < self.opt.max_depth)
                    if valid.sum() == 0:
                        continue
                    for metric_condition in self.metric_conditions:
                        if metric_condition in weather:
                            self.metrics[metric_condition]['count'] += 1
                    self.metrics['all-conditions']['count'] += 1

                # Calculate predicted metrics
                for mode in self.metrics_modes:
                    compute_depth_metrics(
                        gt=inputs["depth_gt"], pred=depth_pp if 'pp' in mode else depth, weather=inputs["weather"],
                        metrics=self.metrics, mode=mode, min_depth=self.opt.min_depth,
                        max_depth=self.opt.max_depth, use_gt_scale='gt' in mode
                    )

    # pred_disps = np.concatenate(pred_disps)

    print('finished predicting!')


    

    
    


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
