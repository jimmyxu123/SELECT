#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --mail-user=wpy2004@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=imgnetpp
#SBATCH --output=full_test_imgnetpp.out
module purge
singularity exec --nv \
 --overlay /scratch/wpy2004/conda_envs/fixed_llm/fixed_llm.ext3:rw \
 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
 /bin/bash -c "source /scratch/wpy2004/conda_envs/fixed_llm/ext3/env.sh; python /scratch/wpy2004/vtab_pytorch/testAllModelsDatasets.py"
