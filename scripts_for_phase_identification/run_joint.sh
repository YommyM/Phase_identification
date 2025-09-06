#!/bin/sh
#SBATCH -J j4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
###SBATCH --gres=gpu:1
#SBATCH -w node1
#SBATCH -o j4.out
#SBATCH -e j4.err
source /data/gulab/yzdai/anaconda3/bin/activate density
SCRIPT="/data/gulab/yzdai/data4/phase_identification/scripts_for_phase_identification/joint_dis.py"
python $SCRIPT