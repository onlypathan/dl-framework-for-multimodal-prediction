#!/bin/bash
#SBATCH --job-name=train_mlp
#SBATCH --output=train_mlp_embedding.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=liu32_1378



module purge
module load python/3.10
source ~/myenv/bin/activate

cd /scratch1/rpathan/yelp_open/mlp

python train_mlp_embedding.py