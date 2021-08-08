#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=16000
#SBATCH --job-name=mistakes_were_made
#SBATCH --output=job.out
#SBATCH --mail-user=eriko1306@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:P100:1
module load PyTorch/1.7.1-fosscuda-2020b
source venv/bin/activate
python3 -m GPTime --task train --cfg_path configs/config_train.yml
