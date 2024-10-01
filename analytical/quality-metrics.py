import webdataset as wds
from PIL import Image
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
from torchvision import transforms as pth_transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchmetrics.image.inception import InceptionScore
import torch
torch.manual_seed(42)

#METRICS
iqa_metric = CLIPImageQualityAssessment().to("cuda:0")
cs_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda:0")
inception_metric = InceptionScore().to("cuda:0")

#TRANSFORMS
transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

#FUNCTIONS
def calculate_clip_score(images, prompts):    
    clip_score = cs_metric(images, prompts).detach().cpu().item()
    iqa_score = iqa_metric(images).mean().detach().cpu().item()
    return (clip_score, iqa_score)

def calculate_inception_score(images):
    inception_score = inception_metric(images.type(torch.uint8))[0].detach().cpu().item()
    return inception_score

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def preprocess(sample):
    jpg, js = sample
    return transform(jpg), js

#DATASETS
targ_str = "/scratch/projects/hegdelab/bf996/datasets/sd-imagenet-wds/{00000..01197}.tar"
save_file = "/scratch/projects/hegdelab/bf996/datasets/sd-imagenet-jsons.pkl"

# targ_str = "/scratch/projects/hegdelab/bf996/datasets/oi1k-imagenet/{00000..01227}.tar"
# save_file = "/scratch/projects/hegdelab/bf996/datasets/oi1k-imagenet-jsons.pkl"

# targ_str = "/scratch/projects/hegdelab/bf996/datasets/imagenet-wds/{00000..01282}.tar"
# save_file = "/scratch/projects/hegdelab/bf996/datasets/imagenet-wds-jsons.pkl"

# targ_str = "/scratch/projects/hegdelab/bf996/datasets/laion-imagenet-mod/modified_{00000..01386}.tar"
# save_file = "/scratch/projects/hegdelab/bf996/datasets/laion-imagenet-mod-jsons.pkl"

# targ_str = "/scratch/projects/hegdelab/inpp/laionnet_wds/{00000..01198}.tar"
# save_file = "/scratch/projects/hegdelab/inpp/laionnet_wds-jsons.pkl"

#HPARAMS
bs = 64
n_batches = 1200000 // bs

#BEGIN MAIN LOOP
dataset = wds.WebDataset(targ_str).shuffle(1000).decode('pil', handler=log_and_continue).to_tuple("jpg", "json").map(preprocess, handler=log_and_continue).batched(bs)

jsons = []
scores = []

for idx, batch in tqdm(enumerate(dataset), total=n_batches):
    img, js = batch[0], batch[1]
    img = img.cuda()
    #captions = [j['caption'] for j in js]
    scores.append(calculate_inception_score(img))

print("Inception Score Avg:")
inception_score_avg = np.mean(scores)
print(inception_score_avg)

print("Done!")