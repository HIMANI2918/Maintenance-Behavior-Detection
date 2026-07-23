#!/bin/bash

#SBATCH --job-name=video_job
#SBATCH --output=video.out
#SBATCH --mail-user=username@example.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:a100_3g.40gb:2
#SBATCH --partition=gpu-a100-mig2
#SBATCH --error=video.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --account=research-ads

module purge
module load cuda/12.1.1

source /home/username/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/username/myenv
python3 -c "import torch; print(torch.cuda.is_available())"
python3 /scratch/username/automate.py
