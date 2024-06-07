# Original copyright (c) Facebook, Inc. and its affiliates, modified 06-2024 by Benjamin Feuer
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

try:
    import webdataset as wds
except ImportError as e:
    print("Please install webdataset package `pip install webdataset`.")
    wds = None

try:
    import datasets as hf_datasets
except ImportError as e:
    print("Please install Hugging Face datasets package `pip install datasets`.")
    hf_datasets = None

from reader_hfds import ReaderHfds
import utils

def extract_feature_pipeline(args, do_train=False, do_val=False):
    train_features = test_features = train_labels = test_labels = None
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if do_train:
        if args.train_data_type == "imagefolder":
            dataset_train = ReturnIndexDataset(args.train_data_path, transform=transform)
            sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                sampler=sampler,
                batch_size=args.batch_size_per_gpu,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            if args.dataset_n_samples == 0:
                args.dataset_n_samples = len(data_loader_train.dataset)
        elif args.train_data_type == "wds":
            def preprocess(t):
                img, json = t
                int_idx = int(json['idx'])
                if int_idx > 999 or int_idx < 0:
                    return None
                return transform(img), torch.tensor([int_idx])
            dataset_train = wds.WebDataset(args.train_data_path, resampled=False, shardshuffle=True).decode('pil', handler=utils.log_and_continue).to_tuple("jpg", "json", handler=utils.log_and_continue).map(preprocess, handler=utils.log_and_continue)
            breakpoint()
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size_per_gpu,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            assert args.dataset_n_samples is not None, "Please provide the number of samples in the dataset"
        elif args.train_data_type == "hfds":
            dataset_train  = ReaderHfds(root=args.dump_train_features if args.dump_train_features is not None else args.load_train_features, name=args.train_data_path, split='train', download=False)
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size_per_gpu,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            if args.dataset_n_samples == 0:
                args.dataset_n_samples = len(data_loader_train.dataset)
        else:
            raise NotImplementedError
        if args.dataset_n_search == 0:
            args.dataset_n_search = args.dataset_n_samples
        args.dl_size = args.dataset_n_search // args.batch_size_per_gpu
    if do_val:
        dataset_val = ReturnIndexDataset(args.val_data_path, transform=transform)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    # print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    if do_train:
        print("Extracting features for train set...")
        train_features, train_labels = extract_features(model, data_loader_train, args.use_cuda, dl_size=args.dl_size, ds_size=args.dataset_n_search, subsample_size=args.dataset_n_samples)
    if do_val:
        print("Extracting features for val set...")
        test_features, _ = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        if do_train:
            train_features = nn.functional.normalize(train_features, dim=1, p=2)
        if do_val:
            test_features = nn.functional.normalize(test_features, dim=1, p=2)
    if do_train and args.train_data_type not in ["wds", "hfds"]:
        train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    if do_val:
        test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if train_features is not None and dist.get_rank() == 0:
        os.makedirs(args.dump_train_features, mode = 0o777, exist_ok = True)
        torch.save(train_features.cpu(), os.path.join(args.dump_train_features, "trainfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_train_features, "trainlabels.pth"))
    if test_features is not None and dist.get_rank() == 0:
        os.makedirs(args.dump_val_features, mode = 0o777, exist_ok = True)
        torch.save(test_features.cpu(), os.path.join(args.dump_val_features, "testfeat.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_val_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False, dl_size=None, ds_size=None, subsample_size=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    all_labels = []
    if ds_size is None:
        ds_size = len(data_loader.dataset)
    batch_count = 0
    for samples, label in metric_logger.log_every(data_loader, 10, iter_len=dl_size):
        #NOTE: this is not distributed safe
        index = torch.arange(batch_count * samples.shape[0], (batch_count * samples.shape[0]) + samples.shape[0]).cuda(non_blocking=True)
        batch_count += 1

        if args.save_images and batch_count % 10 == 0:
            torchvision.utils.save_image(
                samples,
                os.path.join(output_dir, 'train-batch-%d.jpg' % batch_count),
                padding=0,
                normalize=True
            )
            target_list = torch.argmax(label, dim=1).detach().cpu().tolist()
            with open(os.path.join(output_dir, 'train-batch-%d.txt' % batch_count), 'w') as f:
                f.write(str(target_list))

        samples = samples.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(ds_size, feats.shape[-1])
            if len(label.shape) == 1:
                ret_labels = torch.zeros(ds_size, dtype=label.dtype)
            elif len(label.shape) == 2:
                ret_labels = torch.zeros((ds_size, 1), dtype=label.dtype)
            else:
                raise ValueError("Label shape not supported")
            if use_cuda:
                features = features.cuda(non_blocking=True)
                ret_labels = ret_labels.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)
        all_labels.append(label)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
                ret_labels.index_copy_(0, index_all, label)
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
                ret_labels.index_copy_(0, index_all.cpu(), label.cpu())
    # ret_labels = torch.cat(all_labels).long().flatten()[:features.size(0)]
    assert features.size(0) == ret_labels.size(0), "Features and labels have different sizes, {} vs {}".format(features.size(0), ret_labels.size(0))
    features, ret_labels = utils.dedup_features(features, ret_labels)
    #flatten ret_labels
    ret_labels = ret_labels.long().flatten()
    #subsample the features
    if subsample_size is not None:
        features, ret_labels = utils.resample_features(features, ret_labels, subsample_size)

    #count frequency of each label
    label_freq = torch.bincount(ret_labels)
    # print(f"Label frequency before balancing: {label_freq}")
    #Balance the features
    if subsample_size is not None:
        ret_labels, features = utils.regularize_label_count(ret_labels, features, label_freq, thresh=subsample_size // len(label_freq))
    else:
        ret_labels, features = utils.regularize_label_count(ret_labels, features, label_freq, thresh=ds_size // len(label_freq))
    label_freq = torch.bincount(ret_labels)
    assert torch.unique(label_freq).size(0) < 3, "Balancing failed"
    print("Dataset size after balancing: ", len(ret_labels))
    # assert len(features) - torch.unique(features, dim=0).size(0) == 0, "Duplicate features found"

    return features, ret_labels


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5

#just a wrapper for ImageFolder
class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_train_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--dump_val_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_train_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--load_val_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--train_data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--train_data_type', default="imagefolder", type=str)
    parser.add_argument('--val_data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--dataset_n_samples', default=0, type=int)
    parser.add_argument('--dataset_n_search', default=0, type=int)
    parser.add_argument('--save_images', action='store_true', help='Save images in the output folder')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_train_features is not None:
        train_features = torch.load(os.path.join(args.load_train_features, "trainfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_train_features, "trainlabels.pth"))
    else:
        train_features, _, train_labels, _ = extract_feature_pipeline(args, do_train=True)
    if args.load_val_features is not None:
        test_features = torch.load(os.path.join(args.load_val_features, "testfeat.pth"))
        test_labels = torch.load(os.path.join(args.load_val_features, "testlabels.pth"))
    else:
        # need to extract features !
        _, test_features, _, test_labels = extract_feature_pipeline(args, do_val=True)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top5 = knn_classifier(train_features, train_labels,
                test_features, test_labels, k, args.temperature)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
    dist.barrier()
