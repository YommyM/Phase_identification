#!/bin/sh
#SBATCH -J g4b2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
###SBATCH --gres=gpu:1
#SBATCH -w node1
#SBATCH -o g4b2.out
#SBATCH -e g4b2.err
source /data/gulab/yzdai/anaconda3/bin/activate density
SCRIPT="/data/gulab/yzdai/data4/phase_identification/plot/scripts/bin_density.py"
python $SCRIPT