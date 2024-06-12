import torch
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from getDataLoader import get_data_loader, get_default_device, to_device
import numpy as np
from tqdm import tqdm
from vtab_finetune import finetune_eval
import copy

random_seed = 1234
torch.manual_seed(random_seed)
import sys
import os

os.makedirs("results", exist_ok=True)

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Load a pretrained model.")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the pretrained model file."
    )
    args = parser.parse_args()
    return args


DATASET_LIST = [
    "caltech256",
    "svhn",
    "dtd",
    "eurosat",
    "flowers102",
    "country211",
    "fgvcaircraft",
    "gtsrb",
    "renderedsst2",
    "lfwpeople",
    "sun397",
]


def test_all_datasets(
    model,
    model_name,
    finetune_layer="fc",
    LR=0.01,
    MAX_STEPS=1000,
    DECAY_STEPS=300,
    DECAY_GAMMA=0.1,
    MOMENTUM=0.9,
    BATCH_SIZE=64,
    INPUT_SIZE=64,
):
    for dataset in DATASET_LIST:
        f = open("results/" + model_name + ".txt", "a")
        copy_model = copy.deepcopy(model)
        acc, CI = finetune_eval(
            copy_model,
            finetune_layer=finetune_layer,
            LR=LR,
            MAX_STEPS=MAX_STEPS,
            DECAY_STEPS=DECAY_STEPS,
            DECAY_GAMMA=DECAY_GAMMA,
            MOMENTUM=MOMENTUM,
            BATCH_SIZE=BATCH_SIZE,
            INPUT_SIZE=INPUT_SIZE,
            DATASET=dataset,
        )
        print(dataset + " Accuracy: " + str(acc) + " \u00B1 " + str(CI), file=f)
        f.close()


def test():
    if args.model:
        pretrained_cfg = {"url": "", "file": args.model}
        model = timm.create_model(
            "resnet50", pretrained=True, pretrained_cfg=pretrained_cfg
        )
    else:
        # Load the default pretrained model if no path is provided
        model = timm.create_model("resnet50", pretrained=True)
    test_all_datasets(
        model,
        "Timm Resnet50",
        finetune_layer="fc",
        LR=0.01,
        MAX_STEPS=1000,
        DECAY_STEPS=300,
        DECAY_GAMMA=0.1,
        MOMENTUM=0.9,
        BATCH_SIZE=64,
        INPUT_SIZE=64,
    )


if __name__ == "__main__":
    test()
