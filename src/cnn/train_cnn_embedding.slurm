#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --output=train_cnn_embedding.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --account=liu32_1378



module purge
module load python/3.10
source ~/myenv/bin/activate

cd /scratch1/rpathan/yelp_open/cnn

python train_cnn_embedding.py