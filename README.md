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



3. run the python file to obtain the pre-extracted CLIP image features
```
python CLIP_hicodet_extract.py
```
Remember to make sure the correct path for annotation files and datasets.



## HICO-DET
### Train on HICO-DET:
```
bash scripts/hico_train_vitB_zs.sh
```

### Test on HICO-DET:
```
bash scripts/hico_test_vitB_zs.sh
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
