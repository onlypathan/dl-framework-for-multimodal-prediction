#!/bin/bash
#SBATCH --job-name=train_cnn_s
#SBATCH --output=train_cnn_stars.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --account=liu32_1378



module purge
module load python/3.10
source ~/myenv/bin/activate

cd /scratch1/rpathan/yelp_open/cnn

python train_cnn_stars.py