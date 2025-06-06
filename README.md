# [CVPR 2025] Synthetic-to-Real Self-supervised Robust Depth Estimation via Learning with Motion and Structure Priors

## Paper Links

[arXiv]([https://arxiv.org/abs/2503.20211]

![image](https://github.com/DavidYan2001/Synthetic2Real-Depth/blob/main/Figures/comparison_qualitative.png)


## Dataset 
Follow the process of [md4all](https://github.com/md4all/md4all).


## Dependencies
```
conda env create -f environment.yml -n syn2realdepth
```
## Checkpoint Files
You can download the checkpoints from https://drive.google.com/drive/folders/1Z95Q8m5z4g_9JdYxrOw-VPg2umGWgv1J?usp=sharing to /checkpoints/nuScenes or /checkpoints/Robotcar.



## Evaluation
### Eval on nuScenes:
```
python -m nuScenes.manydepth.train --data_path /datasets/nuscenes
--log_dir /syn2real_depth/log --model_name=test_nuScenes --running_mode='val'
```

### Eval on Robotcar:
```
python -m Robotcar.manydepth.train --data_path /datasets/Robotcar
--log_dir /syn2real_depth/log --model_name=test_Robotcar --running_mode='val'
```

Then use the script for Tensorboard to see the performance.
```
tensorboard --logdir= --port=9999
```
## Contact
Feel free to contact Weilong Yan: 1092443660ywl@gmail.com.


## Citation
If you find our paper and/or code helpful, please consider citing :
```
@InProceedings{Yan_2025_CVPR,
    author    = {Yan, Weilong and Li, Ming and Li, Haipeng and Shao, Shuwei and Tan, Robby T.},
    title     = {Synthetic-to-Real Self-supervised Robust Depth Estimation via Learning with Motion and Structure Priors},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {21880-21890}
}
```
