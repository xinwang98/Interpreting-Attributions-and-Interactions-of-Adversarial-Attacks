This repository is a pytorch implementation of Interpreting Attributions and Interactions of Adversarial Attacks [arXiv](https://arxiv.org/abs/2108.06895).

## Requirement

* python >= 3.6
* pytorch >= 1.1.0
* torchvision >= 0.4.0
* numpy >= 1.16.0
* matplotlib >= 3.1.0
* argparse >= 1.1

## Program and Parameters

### Tools

- **util:** useful short functions
- **dataset.py:** provide datasets for datasets
- **main_attribution.py:** set parameters and call function to compute regional attribution
- **generate_info.py:** for interaction task to judge whether a component is in foreground or background
- **component_func.py:** for interaction task to compute function of components in attack(increase target score or decrease label score)
- **gaussian_plot.ipynb:** for interaction task plot to perturbation components
- **iou.ipynb:** for attribution task to compute iou between regional attribution and perturbation magnitudes
- **cal_shap.py:** for attribution task to get final regional attribution

### Core codes
**attribution** regional attribution computation
- **to_attack.py:** to conduct attack images with sampling for shapley value computation
- hyper-parameters:
  - --gpu_id: GPU id
  - --model_type: model used
  - --data_type: dataset used
  - --num_labels: total labels in dataset, voc-default=20, cub=200, sdd=120, celeba=80
  - --partition: the number for image division, default=8
  - --img_size: the size of image
  - --padding_size: the size for image padding
  - --begin: smallest region number which allowed to be attacked, default=0
  - --low: when region number in (begin,low), do sequential sampling, default=10
  - --high: when region number in (high,end), do sequential sampling, default=54
  - --end: largest region number which allowed to be attacked, default=64
  - --sample_num: sample times for each region number, default=64
  - --batch_size: batch size for attack, default=64
  - --tlab: target label to be attacked to, voc-default=2, cub-default=199, sdd-default=119
- **cw_attack.py:** l2-attack for images
- hyper-parameters:
  - --bss: binary search steps, default=5
  - --mir: max iterations in each search step, default=200
  - --lr: learning rate for attack, default=25e-3
  - --confidence: confidence for loss1 in cw-attack, default=0
- **li_fgsm.py:** li-attack for images
- hyper-parameters:
  - --alpha: step size for adversarial update, default=1/256
  - --iterations: max iterations for each epsilon, default=80
- **mask.py** generate region indices and corresponding masks for attack


**interaction** to cumpute interactions and extract perturbation components via hierarchical clustering
- **run.py:** prepare to compute interactions and do clustering
- hyper-parameters:
  - --gpu_id: GPU id
  - --model_type: model used
  - --data_type: dataset used
  - --num_labels: total labels in dataset, voc-default=20, tiny_cub=10, sdd=120, celeba=80
  - --partition: the number for image division, default=8
  - --img_size: the size of image, default=224
  - --padding_size: the size for image padding, default=0
  - --component_size: components number to be merged into a larger one, default=4
  - --super_size: pixels number to be seen as a whole, default=4
  - --tlab: target label to be attacked to, voc-default=2, cub-default=9, sdd-default=119
- **player.py:** to represent for the component in the game
- **shapley_compute.py:** implementation of shapley computation with taylar approximation
- **patch_shapley.py:** prepare to compute interactions in a patch
- **global_shapley.py:** prepare to compute interactions in global
- **combination_global.py:** generate component candidates in global
- **combination_patch.py:** generate component candidates in a patch
- **cluster.py:** do clustering according to component interactions

## Examples
**Here gives several example commands to run the experiments**
- to generate components of 10th to 15th images in VOC 2012(voc) on resnet18
```cmd
python main_interaction.py --data_type voc --model_type res18 --img_start 10 --img_run 5
```

- to generate regional attribution of images in CUB200-2011(cub) on vgg16 with cw_attack
```cmd
python main_attribution.py --data_type cub --model_type vgg16 --attack_type l2
python cal_shap.py
```

- to generate regional attribution of images in Stanford Dog Dataset(sdd) on alexnet with bim
```cmd
python main_attribution.py --data_type sdd --model_type alexnet --attack_type li
python cal_shap.py
```



## - Citation

Please cite the following paper, if you use this code.

```
@InProceedings{Wang_2021_ICCV,
    author    = {Wang, Xin and Lin, Shuyun and Zhang, Hao and Zhu, Yufei and Zhang, Quanshi},
    title     = {Interpreting Attributions and Interactions of Adversarial Attacks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {1095-1104}
}
```
