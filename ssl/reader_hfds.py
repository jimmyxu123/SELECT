""" Dataset reader that wraps Hugging Face datasets

Hacked together by / Copyright 2022 Ross Wightman, modified 2024 by Benjamin Feuer
"""
import io
import math
import os
import pickle
from abc import abstractmethod

import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms as pth_transforms


class Reader:
    def __init__(self):
        pass

    @abstractmethod
    def _filename(self, index, basename=False, absolute=False):
        pass

    def filename(self, index, basename=False, absolute=False):
        return self._filename(index, basename=basename, absolute=absolute)

    def filenames(self, basename=False, absolute=False):
        return [
            self._filename(index, basename=basename, absolute=absolute)
            for index in range(len(self))
        ]


try:
    import datasets
except ImportError as e:
    print("Please install Hugging Face datasets package `pip install datasets`.")
    exit(1)


def load_class_map(map_or_filename, root=""):
    if isinstance(map_or_filename, dict):
        assert dict, "class_map dict must be non-empty"
        return map_or_filename
    class_map_path = map_or_filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, class_map_path)
        assert os.path.exists(class_map_path), (
            "Cannot locate specified class map file (%s)" % map_or_filename
        )
    class_map_ext = os.path.splitext(map_or_filename)[-1].lower()
    if class_map_ext == ".txt":
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    elif class_map_ext == ".pkl":
        with open(class_map_path, "rb") as f:
            class_to_idx = pickle.load(f)
    else:
        assert False, f"Unsupported class map file extension ({class_map_ext})."
    return class_to_idx


def get_class_labels(info, label_key="label"):
    if "label" not in info.features:
        return {}
    class_label = info.features[label_key]
    class_to_idx = {n: class_label.str2int(n) for n in class_label.names}
    return class_to_idx


class ReaderHfds(Reader):
    def __init__(
        self,
        root,
        name,
        split="train",
        class_map=None,
        label_key="label",
        download=False,
    ):
        """ """
        super().__init__()
        self.root = root
        self.split = split
        self.dataset = datasets.load_dataset(
            name,  # 'name' maps to path arg in hf datasets
            split=split,
            cache_dir=self.root,  # timm doesn't expect hidden cache dir for datasets, specify a path
        )
        self.dataset = self.dataset.shuffle(seed=42)
        self.transform = pth_transforms.Compose(
            [
                pth_transforms.Resize(256, interpolation=3),
                pth_transforms.CenterCrop(224),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.img_key = "image"
        # leave decode for caller, plus we want easy access to original path names...
        try:
            self.dataset = self.dataset.cast_column(
                self.img_key, datasets.Image(decode=False)
            )
        except:
            try:
                self.dataset = self.dataset.cast_column(
                    "jpg", datasets.Image(decode=False)
                )
                self.img_key = "jpg"
            except:
                raise ValueError("Could not cast image column to datasets.Image")
        self.label_key = label_key
        self.remap_class = False
        if class_map:
            self.class_to_idx = load_class_map(class_map)
            self.remap_class = True
        else:
            self.class_to_idx = get_class_labels(self.dataset.info, self.label_key)
        self.split_info = self.dataset.info.splits[split]
        self.num_samples = self.split_info.num_examples

    def __getitem__(self, index):
        item = self.dataset[index]
        if "__key__" in item:
            item[self.label_key] = int(item["__key__"].split("/")[0])
        image = item[self.img_key]
        if "bytes" in image and image["bytes"]:
            image = io.BytesIO(image["bytes"])
        else:
            assert "path" in image and image["path"]
            image = open(image["path"], "rb")
        label = item[self.label_key]
        if self.remap_class:
            label = self.class_to_idx[label]
        image = Image.open(image).convert("RGB")
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _filename(self, index, basename=False, absolute=False):
        item = self.dataset[index]
        return item["image"]["path"]
