#!/bin/sh
#SBATCH -J area8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
###SBATCH --gres=gpu:1
#SBATCH -w node1
#SBATCH -o area8.out
#SBATCH -e area8.err
source /data/gulab/yzdai/anaconda3/bin/activate myenv

SCRIPT=/data/gulab/yzdai/data4/atomdensity/scripts/Voronoi_area_dpdo.py
# 280K
B="9000"
E="10000"
INTERVAL="5"
PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_280k/dppc_dopc_280k_0us.gro"
TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_280k/traj-dpdo280k-0-10us-pbc.xtc"
out='/data/gulab/yzdai/data4/atomdensity/plot_data/dpdo280k_area.xvg'

#290K
# B="9000"
# E="10000"
# INTERVAL="5"
# PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_290k/dppc_dopc_290k_0us.gro"
# TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_290k/traj-dpdo290k-0-10us-pbc.xtc"
# out='/data/gulab/yzdai/data4/atomdensity/plot_data/dpdo290k_area.xvg'

python $SCRIPT -pdb $PDB -trj $TRJ -out $out -b $B -e $E -interval $INTERVAL



