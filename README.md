# [CVPR 2025] Synthetic-to-Real Self-supervised Robust Depth Estimation via Learning with Motion and Structure Priors

## Paper Links

[arXiv]([https://arxiv.org/abs/2503.20211]



## Dataset 
Follow the process of [UPT](https://github.com/md4all/md4all).


## Dependencies
```
conda env create -f environment.yml -n syn2realdepth
```
## Checkpoint Files
You can download the checkpoints from to /Checkpoints/nuScenes or /Checkpoints/Robotcar



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
