import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torch
from .utils import viz_inv_depth, viz_disps, viz_lowest_cost
from .utils import interpolate_image

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
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


class Visualizer:
    def __init__(self, temp_context, vis_color_pred, evaluation_qualitative_res_path, evaluation_visualization_set, vis_rgb, vis_pred_depth, vis_gt_depth,vis_lowest_cost, min_depth, max_depth):
        self.temp_context = temp_context
        self.vis_color_pred = vis_color_pred
        self.evaluation_qualitative_res_path = evaluation_qualitative_res_path
        self.evaluation_visualization_set = evaluation_visualization_set
        self.vis_rgb = vis_rgb
        self.vis_pred_depth = vis_pred_depth
        self.vis_gt_depth = vis_gt_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.vis_lowest_cost = vis_lowest_cost

    def visualize_rgb(self, img_rgb, filename):
        save_image(img_rgb, os.path.join(self.evaluation_qualitative_res_path, f"{filename}.jpg"))

    def visualize_depth(self, inv_depth, filename):
        inv_depth = cv2.cvtColor(viz_inv_depth(inv_depth, percentile=95).astype(np.float32) * 255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.evaluation_qualitative_res_path, f"{filename}_inv_depth.jpg"), inv_depth)

    def visualize_lowest_cost(self, lowest_cost, filename):
        lowest_cost = cv2.cvtColor(viz_lowest_cost(lowest_cost, percentile_max=90, percentile_min=10).astype(np.float32) * 255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.evaluation_qualitative_res_path, f"{filename}_lowest_cost_dino_s14_pre_layer.jpg"), lowest_cost)

    def visualize_depth_gt(self, img_rgb, img_lidar, filename):
        z = img_lidar.cpu().numpy().flatten()
        valid_mask = (1.0 <= z) & (z <= self.max_depth)

        _, height, width = img_lidar.shape
        indices = np.indices((height, width))
        self.indices_x = indices[1, :, :].flatten()
        self.indices_y = indices[0, :, :].flatten()
        indices_x = self.indices_x[valid_mask]
        indices_y = self.indices_y[valid_mask]
        z = z[valid_mask]

        # Create the figure and the axes
        fig, ax = plt.subplots()

        # Show image
        plt.imshow(np.transpose(0.5*img_rgb.numpy(), (1, 2, 0)))

        # Scatter LiDAR points
        plt.scatter(indices_x, indices_y, s=5, c=z, edgecolors='none', cmap="plasma_r", vmin=1.0, vmax=self.max_depth)

        # Axis limiting
        plt.xlim(0, width)
        plt.ylim(height, 0)
        plt.xticks([])
        plt.yticks([])

        # Remove boundaries
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')

        # Save the figure
        plt.savefig(os.path.join(self.evaluation_qualitative_res_path, filename), bbox_inches='tight', pad_inches=0)
        plt.close()

    def visualize_test(self, batch, outputs):
        depth_gt = batch['depth_gt']
        _, _, height, width = depth_gt.shape
        imgs = interpolate_image(batch[('color',0, 0)], [height, width]).cpu()
        #imgs = batch[('color',0, 0)].cpu() if ('color',0, 0) in batch else interpolate_image(batch[('color',0, 0)], [height, width]).cpu()
        inv_depths = outputs[('disp', 0, 0)]
        # lowest_cost = outputs['lowest_cost']

        for j, (img, inv_depth, depth_gt_) in enumerate(zip(imgs, inv_depths, depth_gt)):
            filename = os.path.basename(batch[('filename', 0)][j]).split('.')[0]
            # if not self.evaluation_visualization_set or filename in self.evaluation_visualization_set:
            if self.vis_rgb:
                self.visualize_rgb(img, filename)
            if self.vis_pred_depth:
                self.visualize_depth(inv_depth, filename)
            if self.vis_gt_depth:
                self.visualize_depth_gt(img, depth_gt_, f"{filename}_depth_gt.jpg")
                if 'depth_gt_raw' in batch:
                    depth_gt_raw = batch['depth_gt_raw']
                    self.visualize_depth_gt(img, depth_gt_raw[j], f"{filename}_depth_gt_raw.jpg")
            # if self.vis_lowest_cost:
            #     self.visualize_lowest_cost(lowest_cost_, filename)
                # min_val = np.percentile(lowest_cost_.numpy(), 10)
                # max_val = np.percentile(lowest_cost_.numpy(), 90)
                # lowest_cost_ = torch.clamp(lowest_cost_, min_val, max_val)
                # lowest_cost_ = colormap(lowest_cost)
                # cv2.imwrite(os.path.join(self.evaluation_qualitative_res_path, f"{filename}_lowest_cost.jpg"), lowest_cost_)


    def visualize_predict(self, batch, outputs):
        inv_depths = outputs[('disp', 0, 0)]
        for j, inv_depth in enumerate(inv_depths):
            filename = os.path.basename(batch[('filename', 0)][j]).split('.')[0]
            self.visualize_depth(inv_depth, filename)

    def tensorboard_add_images(self, images, logger, global_step):
        for k, imgs in images.items():
            logger.experiment.add_images(k, imgs, global_step)

    def tensorboard_add_text(self, tag, text, logger, global_step):
        logger.experiment.add_text(tag, text, global_step)

    def visualize_train(self, batch, outputs, logger, global_step):
        color_0 = {"color_0": batch[("color", 0)].clone().detach().cpu()}
        color_ref = {f"color_{frame_id}": batch[("color", frame_id)].clone().detach().cpu() for frame_id in self.temp_context[1:]}
        color = {**color_0, **color_ref}
        self.tensorboard_add_images(color, logger, global_step)

        color_aug_0 = {"color_aug_0": batch[("color_aug", 0)].clone().detach().cpu()}
        color_aug_ref = {f"color_aug_{frame_id}": batch[("color_aug", frame_id)].clone().detach().cpu() for frame_id in self.temp_context[1:]}
        color_aug = {**color_aug_0, **color_aug_ref}
        self.tensorboard_add_images(color_aug, logger, global_step)

        color_aug_pose_0 = {"color_aug_pose_0": batch[("color_aug_pose", 0)].clone().detach().cpu()}
        color_aug_pose_ref = {f"color_aug_pose_{frame_id}": batch[("color_aug_pose", frame_id)].clone().detach().cpu() for frame_id in self.temp_context[1:]}
        color_aug_pose = {**color_aug_pose_0, **color_aug_pose_ref}
        self.tensorboard_add_images(color_aug_pose, logger, global_step)

        if self.vis_color_pred:
            color_pred = {f"color_pred_{frame_id}": outputs[("color", frame_id, 0)].clone().detach().cpu() for frame_id in self.temp_context[1:]}
            self.tensorboard_add_images(color_pred, logger, global_step)

        disparity = {"disparity": viz_disps(outputs[("disp", 0, 0)], batch[("color", 0)].shape)}
        self.tensorboard_add_images(disparity, logger, global_step)

        self.tensorboard_add_text('weather', " ".join(batch['weather']), logger, global_step)
        self.tensorboard_add_text('weather_depth', " ".join(batch['weather_depth']), logger, global_step)
        self.tensorboard_add_text('weather_pose', " ".join(batch['weather_pose']), logger, global_step)
