# 🌋 SELECT: A Large-Scale Benchmark of Data Curation Strategies for Image Recognition

This is the repository for the SELECT benchmark of data curation strategies and
the ImageNet++ dataset. 

Please check out our [website](https://nyu-dice-lab.github.io/SELECT/) for updates on this project.

The repository contains links to and descriptions of
the ImageNet++ dataset, a superset of ImageNet-1k. It also contains the code we
used to train our data curation benchmark models, and all of the necessary
scripts for evaluating those models in a manner similar to that described in
our paper.

<p align="center">
  <img src="images/select.jpg" />
</p>

## Table of Contents

1. [Installation](#installation)
2. [ImageNet++](#ImageNet++)
3. [Training](#Training)
4. [SELECT](#SELECT)
5. [Paper, website, and docs](#Paper)
6. [Citation](#Citation)

## Installation

Firstly, download and set up the repo:

```
git clone https://github.com/jimmyxu123/SELECT.git
cd SELECT/subrepo/you/want/to/use
```

You can install dependencies using requirements.txt.

## ImageNet++

ImageNet++ is the largest, most diverse set of distribution shifts for
ImageNet-1k to date.
For more details on its contents and the curation strategies used to construct
it, please refer to our paper.

![ImageNet++ comparison](images/imagenetpp.jpg)

### How to download ImageNet++

Our training dataset (baseline and shifts) can be downloaded
from [Huggingface](https://huggingface.co)

```
IN1000: https://huggingface.co/datasets/ILSVRC/imagenet-1k
OI1000: https://huggingface.co/datasets/nyu-dice-lab/imagenetpp-openimages
SD1000(i2i): https://huggingface.co/datasets/nyu-dice-lab/imagenetpp-gen-sd-i2i
SD1000(t2i): https://huggingface.co/datasets/ek826/imagenet-gen-sd1.5
LA1000(i2i): https://huggingface.co/datasets/nyu-dice-lab/imagenetpp-laion-i2i
LA1000(t2i): https://huggingface.co/datasets/nyu-dice-lab/imagenetpp-laion-t2i
```

## Training

We train our models with a modified version of the
popular [timm](https://github.com/huggingface/pytorch-image-models) library.
For convenience, we condense the necessary code into a single training script.

### Example Training Command

For instance, to launch training on OI1000 with N GPUs on one node:

```
torchrun --nnodes=1 --nproc_per_node=N train.py -c config/oi1000-rn50-tv-wds.yaml
```

You can modify the configuration file to decide your favorite hyperparameters.

## SELECT

SELECT is a diverse benchmark for data curation methods for computer vision.
Our holistic benchmark evaluates datasets and the models trained using them
according to several utility metrics. In this README, we address FAQs
and describe how to use our benchmark. For more
details on the SELECT benchmark, including results from our model runs, please
refer to our paper.

### What is data curation?

We model a data curation strategy as a rational series of choices made by
humans with the aim of maximizing the utility of a dataset of a given size (
also referred to as a shift).

### What metrics does SELECT include?

The first metric we report is **base accuracy** on holdout data drawn from the
same distribution as the data of the baseline strategy (in this case, ImageNet
validation accuracy). In order to assess **out-of-distribution robustness**, we
report several metrics, both synthetic and natural. Inspired by
the [VTAB-1k benchmark for TensorFlow models](https://google-research.github.io/task_adaptation/),
we assemble 11 downstream tasks and fine-tune our pretrained model checkpoints
to assess **transfer learning**. Finally, we assess the utility of our datasets
for **self-supervised learning** by evaluating
a [DINO](https://github.com/facebookresearch/dino) model pretrained on
ImageNet-train using varying-sized subsets of our shifts.

### How to use SELECT

There are two ways to use our benchmark; first, it is possible to run each
evaluation type independently as its own standalone module. Second, we provide
a simple script to evaluate a model on all of the modules in sequence.

### Pretrained SELECT Baselines

All of our pretrained ImageNet++ baselines can be retrieved from the
associated [HuggingFace repository](https://huggingface.co/collections/nyu-dice-lab/select-baselines-666e8963b955b0e655b62d13):

```
IN1000     : https://huggingface.co/nyu-dice-lab/imagenetpp-baseline-imagenet-1k
OI1000     : https://huggingface.co/nyu-dice-lab/imagenetpp-baseline-open-images
SD1000(i2i): https://huggingface.co/nyu-dice-lab/imagenetpp-baseline-sd-i2i
SD1000(t2i): https://huggingface.co/nyu-dice-lab/imagenetpp-baseline-sd-t2i
LA1000(i2i): https://huggingface.co/nyu-dice-lab/imagenetpp-baseline-laion-i2i
LA1000(t2i): https://huggingface.co/nyu-dice-lab/imagenetpp-baseline-laion-t2i
```

### Base Accuracy and OOD Robustness

The following instructions describe how to evaluate a trained model on the OOD
robustness and base accuracy benchmarks in our paper. You will need a
pretrained timm model checkpoint (you can also use the ones we provide).

#### First Run

Prior to evaluating your model, please download and prepare the benchmark
datasets by following the instructions provided on
the [EasyRobust repository.](https://github.com/alibaba/easyrobust)

#### How to Run

Here is an example command to evaluate the timm ResNet-50 checkpoint on
ImageNet-val:

```
python base_ood_eval.py --imagenet_val "/imagenet/val/"
```

You can also pass a command to the model weights to evaluate a particular
checkpoint.

```
python base_ood_eval.py --imagenet_val "/imagenet/val/" --model "PATH/TO/CKPT"
```

#### Datasets Supported

```
ImageNet-V2: --imagenet_v2 
ImageNet-Sketch: --imagenet_s
ImageNet-A: --imagenet_a
ImageNet-R: --imagenet_r
ObjectNet: --objectnet
ImageNet-C: --imagenet_c
Stylized-ImageNet: --stylized_imagenet
```

### Transfer Learning

Steps to reproduce our transfer learning (VTAB) benchmark:

1. Create a folder called "vtab_weights". Store the [model checkpoints](https://huggingface.co/collections/nyu-dice-lab/imagenet-666e885314f1c262fec84ef8) in this
   folder. In order to reproduce the results in our paper, your folder should
   include the following:

```
vtab_weights/imagenet-1k.pth.tar,
vtab_weights/open-images.pth.tar,
vtab_weights/sd1000-i2i.pth.tar,
vtab_weights/sd1000-t2i.pth.tar,
vtab_weights/laion-i2i.pth.tar,
vtab_weights/laion-t2i.pth.tar
```

2. Run "testAllModelsDatasets.py". Results will be found in the "results"
   folder, or you can choose to evaluate a single pre-trained model on all of
   the VTAB datasets:

```
# run a standard pretrained timm ResNet-50 model
python TestAllDatasets.py

# run a ImageNet++ pretrained ResNet-50 model
python TestAllDatasets.py --model "your/preferred/model/pth"
```

### Self-supervised learning

You can run a self-supervised learning method DINO KNN evaluation on
ImageNet++. For people running in a SLURM environment, run `unset SLURM_PROCID`
first to avoid triggering an error with the distributed code.
For running on a webdataset format dataset, please follow the below code.

```
# Example eval command on WDS SD1000(img2img)
python ssl/eval_knn.py \
	--train_data_path "/your/dataset/path/to/sd-imagenet-wds/{00000..01197}.tar" \
	--train_data_type wds \
	--val_data_path /imagenet/val \
	--pretrained_weights resnet50 \
	--arch resnet50 \
	--dump_train_features /your/path/to/dino/logs/sdimg2img-5spc-train-wds \
	--load_val_features /scratch/bf996/dino/logs/imagenet-val \
	--batch_size_per_gpu 1000 \
	--dataset_n_samples 5000
```

For running on a huggingface dataset, please follow the below code.

```
# HFDS SD1000(txt2img)
python eval_knn.py \
	--train_data_path "/your/dataset/path/to/ek826___imagenet-gen-sd1.5" \
	--train_data_type hfds \
	--val_data_path /imagenet/val \
	--pretrained_weights resnet50 \
	--arch resnet50 \
	--dump_train_features /scratch/bf996/dino/logs/sdtxt2img-5spc-train-wds \
	--load_val_features /scratch/bf996/dino/logs/imagenet-val \
	--batch_size_per_gpu 1000 \
	--dataset_n_samples 5000 > /scratch/bf996/dino/logs/sdtxt2img-5spc-val.txt;
```

For imbalanced datasets, you can use oversampling to increase the number of
real labeled samples per class by adding another argument `--dataset_n_search`.
Here is an example:

```
# Example eval command on WDS OI1000
python eval_knn.py \
 --train_data_path "/your/dataset/path/to/oi1k-imagenet/{00000..01227}.tar" \
 --train_data_type wds \
 --val_data_path /imagenet/val \
 --pretrained_weights resnet50 \
 --arch resnet50 \
 --dump_train_features /scratch/bf996/dino/logs/openimages-10spc-train-wds-oversample \
 --load_val_features /scratch/bf996/dino/logs/imagenet-val \
 --batch_size_per_gpu 1000 \
 --dataset_n_samples 10000 \
 --dataset_n_search 100000
```

### Run All Evals

We also provide a script to run all the evaluations in sequence. For instance,
you can execute the following command to run the whole evaluation:

```
python run_select.py \
	--base_ood \
	--imagenet_val "/imagenet/val/" \
	--vtab \
	--ssl "\
	--train_data_path "/your/dataset/path/to/sd-imagenet-wds/{00000..01197}.tar" \
	--train_data_type wds \
	--val_data_path /imagenet/val \
	--pretrained_weights resnet50 \
	--arch resnet50 \
	--dump_train_features /your/path/to/dino/logs/sdimg2img-5spc-train-wds \
	--load_val_features /scratch/bf996/dino/logs/imagenet-val \
	--batch_size_per_gpu 1000 \
	--dataset_n_samples 5000"

```

## Citation

Coming Soon


