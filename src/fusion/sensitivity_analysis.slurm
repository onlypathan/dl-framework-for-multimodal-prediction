#!/bin/bash
#SBATCH --job-name=train_sa
#SBATCH --output=sensitivity_analysis.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --account=liu32_1378



module purge
module load python/3.10
source ~/myenv/bin/activate

cd /scratch1/rpathan/yelp_open/fusion

python sensitivity_analysis.py