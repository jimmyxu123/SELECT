#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

if [[ "$(hostname -s)" =~ ^g[r,v,a,h] ]]; then nv="--nv"; fi

singularity \
  exec $nv \
  $(for sqf in /vast/work/public/ml-datasets/imagenet/winter21_whole/*.sqf; do echo "--overlay $sqf:ro"; done) \
  --overlay /scratch/bf996/singularity_containers/select-bench.ext3:rw \
  --overlay /vast/work/public/ml-datasets/bf996/CaptionNet/in100.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/arboretum_rare_combined.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/arboretum_test_set_16M.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/deepweeds.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/DF20M.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/ip02.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/plantvillage.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/birds525.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/CaptionNet/laion100.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/CaptionNet/openimages1000.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-r.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-a.sqf:ro \
  --overlay /vast/work/public/ml-datasets/bf996/imagenet-sketch.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/objectnet.sqf:ro \
  --overlay /vast/work/public/ml-datasets/open-images-dataset/open-images-dataset.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-c.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-style.sqf:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
 source /ext3/env.sh; export PYTHONPATH=$PYTHONPATH:/scratch/bf996/pytorch-image-models
 $args 
"