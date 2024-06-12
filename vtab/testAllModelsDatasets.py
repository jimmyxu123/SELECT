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

os.makedirs("results", exist_ok=True)

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

MODEL_LIST = [
    "vtab_weights/in1000.pth.tar",
    "vtab_weights/la1000.pth.tar",
    "vtab_weights/oi1000.pth.tar",
    "vtab_weights/sd1000-i2i.tar",
    "vtab_weights/sd1000-t2i.tar",
    "vtab_weights/laionnet.pth.tar",
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


def test_all_models():
    for model_name in MODEL_LIST:
        model = timm.create_model(
            "resnet50", pretrained=True, pretrained_cfg={"file": model_name}
        )
        test_all_datasets(
            model,
            model_name[13:],
            finetune_layer="fc",
            LR=0.01,
            MAX_STEPS=1000,
            DECAY_STEPS=300,
            DECAY_GAMMA=0.1,
            MOMENTUM=0.9,
            BATCH_SIZE=64,
            INPUT_SIZE=64,
        )
    original_resnet_50 = timm.create_model("resnet50", pretrained=True)
    test_all_datasets(
        original_resnet_50,
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
    test_all_models()
