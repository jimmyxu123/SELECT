import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torchvision
import timm

try:
    from easyrobust.benchmarks import *
    er = True
except ImportError:
    print('easyrobust not available')
    er = False

def easyrobust_eval(model, args):
    if not er:
        print('easyrobust not available')
        return

    # ood
    top1_val = top1_v2 = top1_a = top1_r = top1_sk = top1_si = top1_c = top1_obj = None
    if args.imagenet_val is not None:
        top1_val = evaluate_imagenet_val(model, args.imagenet_val)
    if args.imagenet_v2 is not None:
        top1_v2 = evaluate_imagenet_v2(model, args.imagenet_v2)
    if args.imagenet_a is not None:
        top1_a = evaluate_imagenet_a(model, args.imagenet_a)
    if args.imagenet_r is not None:
        top1_r = evaluate_imagenet_r(model, args.imagenet_r)
    if args.imagenet_s is not None:
        top1_sk = evaluate_imagenet_sketch(model, args.imagenet_s)
    if args.stylized_imagenet is not None:
        top1_si = evaluate_stylized_imagenet(model, args.stylized_imagenet)
    if args.imagenet_c is not None:
        top1_c, _ = evaluate_imagenet_c(model, args.imagenet_c)
    if args.objectnet is not None:
        top1_obj = evaluate_objectnet(model, args.objectnet)
    return {
        'imagenet_val_top1': top1_val,
        'imagenet_a_top1': top1_a,
        'imagenet_r_top1': top1_r,
        'imagenet_sketch_top1': top1_sk,
        'stylized_imagenet_top1': top1_si,
        'imagenet_c_top1': top1_c,
        'objectnet_top1': top1_obj,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--imagenet_val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet_v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--objectnet",
        type=str,
        default=None,
        help="Path to objectnet for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet_c",
        type=str,
        default=None,
        help="Path to imagenet_c for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--stylized_imagenet",
        type=str,
        default=None,
        help="Path to stylized imagenet for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet_s",
        type=str,
        default=None,
        help="Path to imagenet sketch set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet_r",
        type=str,
        default=None,
        help="Path to imagenet-r set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet_a",
        type=str,
        default=None,
        help="Path to imagenet-a set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store results logs.",
    )
    return parser

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    #############################################################
    #         Define your model
    #############################################################
    if args.model is None:
        model = timm.create_model('resnet50', pretrained=True)
    else:
        model = timm.create_model('resnet50', pretrained=True, pretrained_cfg = {'file': args.model})
    if torch.cuda.is_available(): model = model.cuda()
    evals = easyrobust_eval(model, args)
    print("Run complete. Results: ", evals)
    #############################################################
    #         Save Results
    ############################################################# 
    strf_time = datetime.now().strftime("%Y-%m-%d-%H-%M")   
    if not os.path.exists(args.logs):
        os.makedirs(args.logs)
    with open(os.path.join(args.logs, f"results-{strf_time}.json"), "w") as f:
        json.dump(evals, f)
