#!/bin/sh
#SBATCH -J j             # Job name
#SBATCH -N 1                      # Number of nodes
#SBATCH -n 1                      # Number of tasks
#SBATCH --cpus-per-task=1         # CPUs per task
##SBATCH --gres=gpu:1             # Uncomment if GPU needed
#SBATCH -w node1                  # Specific node (optional)
#SBATCH -o j.out          # STDOUT
#SBATCH -e j.err          # STDERR

# Activate your conda environment
source /data/gulab/yzdai/anaconda3/bin/activate density

# Path to your Python script
SCRIPT="/data/gulab/yzdai/data4/phase_identification/plot/scripts/joint_dis.py"

# Run the script, with parameters if needed
python "$SCRIPT" --sys psmdopochl --source HMM
python "$SCRIPT" --sys psmdopochl --source density

python "$SCRIPT" --sys dpdochl280k --source HMM
python "$SCRIPT" --sys dpdochl280k --source density
python "$SCRIPT" --sys dpdochl290k --source HMM
python "$SCRIPT" --sys dpdochl290k --source density

python "$SCRIPT" --sys dpdo280k --source HMM
python "$SCRIPT" --sys dpdo280k --source density
python "$SCRIPT" --sys dpdo290k --source HMM
python "$SCRIPT" --sys dpdo290k --source density