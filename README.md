# 🌋 SELECT-1.2M: A Large-Scale Benchmark of Data Curation Strategies
This is the repository for the SELECT-1.2M benchmark 

---

We introduce SELECT-1.2M, a formal strategy to benchmark labeled datasets and orient data engineers to direct their. Utilizing SELCET-1.2M, we evaluate ImageNet++ as the largest and most diverse superset of the ImageNet-1k training set to date. ImageNet++ extends beyond the scope of traditional datasets by incorporating 7 distinct training shifts, each employing a unique modality of data (two natural and one synthetic) and utilizing distinct selection techniques. The constituent datasets of ImageNet++ are enumerated as follows:
1. OpenImages-1000 (OI1000): A subset of the OpenImages dataset constructed via schema mapping.
2. LAION-1000 (LA1000): A subset of the unlabeled LAION dataset, selected through CLIP retrieval nearest neighbors search against the ImageNet-1k training set.
3. Stable Diffusion-1000 (SD1000): A dataset generated from the ImageNet-1k dataset using an image-to-image Stable Diffusion pipeline

![ImageNet++ comparison](images/imagenetpp2.png)

## Table of Contents

1. [Dataset](#dataset)
2. [Installation](#installation)
3. [Create ImageNet++](#imagenetpp)
4. [Train models on ImageNet++](#Train)
5. [Evaluation Benchmark](#eval)
6. [Paper, website, and docs](#paper)
7. [Citation](#citation)

## Dataset 
The complete dataset is accessible via Huggingface.

Download from [Huggingface](https://huggingface.co): `git lfs clone https://huggingface.co`.

```
├── ImageNet++
    ├── OpenImages-1000(OI-1000)
    ├── LAION-1000(LA-1000)
        ├── LA-1000(img2img)
        ├── LA-1000(txt2img)
        └── LA-1000(substring matching)
    └── StableDiffusion-1000
        ├── SD-1000(txt2img)
        └── SD-1000(img2img)
``` 
We show some image examples from ImageNet++ as above. 

## Installation
Firstly, download and set up the repo:
```
git clone https://github.com/jimmyxu123/SELECT-1.2M.git
cd 
```
## Create ImageNet++  
### OI-1000

### LA-1000
#### img2img
#### txt2img
Inspired by [LAIONNet](https://github.com/alishiraliGit/eval-on-laion), we augmented the original LAIONNet dataset to 1.2M firstly filtering examples where the corresponding caption contains one and only one of the synsets of ImageNet from LAION-400M. Then we only retain examples where the similarity between the ImageNet synset definition and the caption exceeds a threshold of 0.5 with CLIP.
#### substring matching

### SD-1000
#### img2img
#### txt2img

## Train on ImageNet++
We use [timm](https://github.com/huggingface/pytorch-image-models) scripts to train models on our dataset ImageNet++. For instance, to launch training on OI1000 with N GPUs on one node:
```
torchrun --nnodes=1 --nproc_per_node=N train.py -c config/oi1000-rn50-tv-wds.yaml
```
You can modify the configuration file to decide your favorite hyperparameters.

## Evaluation Benchmark


## Paper, Website, and Docs
<h2 id="paper"></h2>

## Acknowledgement


## Citation

If you find our work useful, please consider citing as follows.

