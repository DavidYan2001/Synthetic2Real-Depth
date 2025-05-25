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
@misc{yan2025synthetictorealselfsupervisedrobustdepth,
      title={Synthetic-to-Real Self-supervised Robust Depth Estimation via Learning with Motion and Structure Priors}, 
      author={Weilong Yan and Ming Li and Haipeng Li and Shuwei Shao and Robby T. Tan},
      year={2025},
      eprint={2503.20211},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.20211}, 
}
```
