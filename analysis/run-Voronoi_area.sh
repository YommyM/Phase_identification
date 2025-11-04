#!/bin/sh
#SBATCH -J f5A
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
###SBATCH --gres=gpu:1
#SBATCH -w node1
#SBATCH -o f5A.out
#SBATCH -e f5A.err
source /data/gulab/yzdai/anaconda3/bin/activate myenv

#################### dpdo ##############
SCRIPT1=/data/gulab/yzdai/data4/phase_identification/analysis/Voronoi_area_dpdo.py
# 280K
B="5000"
E="6000"
INTERVAL="5"
PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_280k/dppc_dopc_280k_0us.gro"
TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_280k/traj-dpdo280k-0-10us-pbc.xtc"
out='/data/gulab/yzdai/data4/phase_identification/plot/input/area_others/dpdo280k_area_5-6us_gap5.xvg'

#290K
# B="9000"
# E="10000"
# INTERVAL="5"
# PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_290k/dppc_dopc_290k_0us.gro"
# TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_290k/traj-dpdo290k-0-10us-pbc.xtc"
# out='/data/gulab/yzdai/data4/atomdensity/plot_data/dpdo290k_area.xvg'

python $SCRIPT1 -pdb $PDB -trj $TRJ -out $out -b $B -e $E -interval $INTERVAL

#################### dpdochl ##############

# SCRIPT2="/data/gulab/yzdai/data4/phase_identification/analysis/Voronoi_area_dpdochl.py"

# 280K
# B="8000"
# E="9000"
# INTERVAL="5"
# PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_280k/snapshot9.99us.gro"
# XTC="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_280k/dpdochl280k-1-10us-pbc.xtc"
# Leaflet="/data/gulab/yzdai/dyz_project2/leaflet/dpdochl280k-leaflet.xvg"
# # out='/data/gulab/yzdai/dyz_project2/A/all_fr_VoronoiArea_dpdochl280k.xvg'
# out='/data/gulab/yzdai/dyz_project2/A/gap5_VoronoiArea_dpdochl280k.xvg'

#290K
# B="0"
# E="9459"
# INTERVAL="1"
# PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_290k/dppc_dopc_chol_aa_290k_0us.gro"
# XTC="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_290k/dpdochl290k-0-9.45us-pbc.xtc"
# Leaflet="/data/gulab/yzdai/dyz_project2/leaflet/dpdochl290k-leaflet.xvg"
# out='/data/gulab/yzdai/dyz_project2/A/all_fr_VoronoiArea_dpdochl290k.xvg'
# out='/data/gulab/yzdai/dyz_project2/A/gap5_VoronoiArea_dpdochl290k.xvg'

# python $SCRIPT2 -pdb $PDB -trj $XTC -out $out -b $B -e $E -interval $INTERVAL -leaflet $Leaflet

#################### psmdopochl ##############

# SCRIPT3="/data/gulab/yzdai/data4/phase_identification/analysis/Voronoi_area_psmdopochl.py"

# B="19000"
# E="20000"
# INTERVAL="5"
# PDB="/data/gulab/yzdai/dyz_project1/data/psmdopochl/psmdopochl-rho0.8.gro"
# XTC="/data/gulab/yzdai/dyz_project1/data/psmdopochl/trjcat-psmdopochl-rho0.8-1ns-1-22.xtc"
# Leaflet="/data/gulab/yzdai/dyz_project2/leaflet/psmdopochl300k-0.8-15-20us-5ns-leaflet.xvg"
# out='/data/gulab/yzdai/dyz_project2/A/last1us_gap5_Area_psmdopochl.xvg'

# python $SCRIPT3 -pdb $PDB -trj $XTC -out $out -b $B -e $E -interval $INTERVAL -leaflet $Leaflet



