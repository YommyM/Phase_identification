#!/bin/sh
#SBATCH -J v3k9
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
###SBATCH --gres=gpu:1
#SBATCH -w node1
#SBATCH -o v3k9.out
#SBATCH -e v3k9.err

source /data/gulab/yzdai/anaconda3/bin/activate density
SCRIPT1="/data/gulab/yzdai/data4/phase_identification/scripts_for_phase_identification/phase_identification_pure_lipids.py"
SCRIPT2="/data/gulab/yzdai/data4/phase_identification/plot/scripts/plot_phase.py"
SCRIPT3="/data/gulab/yzdai/data4/phase_identification/plot/scripts/plot_voronoi_with_chol.py"
SCRIPT4="/data/gulab/yzdai/data4/phase_identification/plot/scripts/plot_voronoi_binary.py"

N_GAP='5'
BIN_WIDTH='3'
CAL_RATIO='False'
# #-----------------------dpdo-------------------------------------------
# SYS='dpdo280k'
# PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_280k/dppc_dopc_280k_0us.gro"
# TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_280k/traj-dpdo280k-0-10us-pbc.xtc"
# LEAFLET='/data/gulab/yzdai/data4/phase_identification/leaflet/dpdo280k-leaflet.xvg'

# SYS='dpdo290k'
# PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_290k/dppc_dopc_290k_0us.gro"
# TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_290k/traj-dpdo290k-0-10us-pbc.xtc"
# LEAFLET='/data/gulab/yzdai/data4/phase_identification/leaflet/dpdo290k-leaflet.xvg'


# param="/data/gulab/yzdai/data4/phase_identification/phase_out/$SYS/parameters.json"
# all_dppc_id=($(seq 1 346) $(seq 577 922))

# #-----------------------dpdochl-------------------------------------------
SYS='dpdochl290k'
PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_290k/dppc_dopc_chol_aa_290k_0us.gro"
TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_290k/dpdochl290k-0-9.45us-pbc.xtc"
LEAFLET='/data/gulab/yzdai/data4/phase_identification/leaflet/dpdochl290k-leaflet.xvg'

# SYS='dpdochl280k'
# PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_280k/snapshot9.99us.gro"
# TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_280k/dpdochl280k-8us-pbc.xtc"
# LEAFLET='/data/gulab/yzdai/data4/phase_identification/leaflet/dpdochl280k-leaflet.xvg'

param="/data/gulab/yzdai/data4/phase_identification/phase_out/$SYS/parameters.json"
all_dppc_id=($(seq 1 202) $(seq 577 778))

# #-----------------------psm-------------------------------------------
# SYS='psmdopochl'
# PDB="/data/gulab/yzdai/dyz_project1/data/psmdopochl/psmdopochl-rho0.8.gro"
# TRJ="/data/gulab/yzdai/dyz_project1/data/psmdopochl/trjcat-psmdopochl-rho0.8-1ns-1-22.xtc"
# LEAFLET='/data/gulab/yzdai/data4/phase_identification/leaflet/psmdopochl300k-0.8-0-20us-leaflet.xvg'

# param='/data/gulab/yzdai/data4/phase_identification/phase_out/psmdopochl/parameters.json'
# all_dppc_id=($(seq 1 90) $(seq 91 180))   


all_dppc_id_str=$(IFS=,; echo "${all_dppc_id[*]}")

# TOTAL_FRAMES=20000                  #psmdopochl
# TOTAL_FRAMES=10000                #dpdo
# TOTAL_FRAMES=8000                #dpdochl280k
TOTAL_FRAMES=9000                #dpdochl290k

STEP=1000

BASE_OUTPATH="/data/gulab/yzdai/data4/phase_identification/phase_out/${SYS}/"
mkdir -p "$BASE_OUTPATH"

# for ((START=9000; START<TOTAL_FRAMES; START+=STEP)); do
for ((START=0; START<TOTAL_FRAMES; START+=STEP)); do
    END=$(( (START + STEP < TOTAL_FRAMES) ? (START + STEP) : TOTAL_FRAMES ))
    OUTPATH="${BASE_OUTPATH}${START}-${END}/"
    echo "Processing frames ${START} to ${END}..."
    mkdir -p "$OUTPATH"  

    # python $SCRIPT1 -trj $TRJ -pdb $PDB -start $START -end $END \
    #         -bin_width $BIN_WIDTH -n_gap $N_GAP \
    #         -leaflet $LEAFLET -sys $SYS -cal_ratio $CAL_RATIO -outpath $OUTPATH \
    #         -primary_lipid "$all_dppc_id_str" \
    #         -param $param

    # mkdir -p "$OUTPATH/phaseplot/upper" 
    # # mkdir -p "$OUTPATH/phaseplot/lower" 
    # mkdir -p "$OUTPATH/phaseplot/regi" 

    ### Plot Phase Map
    # python $SCRIPT2 -trj $TRJ -pdb $PDB -b $START -e $END \
    #     -bin_width 2 -inte 5 \
    #     -phasepath $OUTPATH -sys $SYS

    ### Plot Vopronoi for the last 1 us
    # # if [ "$START" -eq 19000 ]; then    #psm
    # if [ "$START" -eq 9000 ]; then       #dpdo
    if [ "$START" -eq 8000 ]; then     #dpdochl290k
    # # if [ "$START" -eq 7000 ]; then     #dpdochl280k
        phasepath="${OUTPATH}${SYS}-rawdata.xvg"
        voronoi_out="$OUTPATH/phaseplot/voronoi/"
        mkdir -p "$OUTPATH/phaseplot/voronoi"  
    #     ################################################dpdo#######################################
    #     # HMMphasepath='/data/gulab/yzdai/data4/atomdensity/HMM/dpdo280k/train5-dpdo280k-rawdata.xvg'
    #     HMMphasepath='/data/gulab/yzdai/data4/atomdensity/HMM/dpdo290k/train0-dpdo290k-rawdata.xvg'
    #     python $SCRIPT4 -trj $TRJ -pdb $PDB -start $START -end $END \
    #         -n_gap $N_GAP \
    #         -sys $SYS \
    #         -phasepath $phasepath -HMMphasepath $HMMphasepath \
    #         -voronoi_out $voronoi_out
    #     echo "Finished processing frames ${START} to ${END}."

        ################################################dpdochl+psmdopochl#########################

        HMMphasepath='/data/gulab/yzdai/data4/atomdensity/HMM/dpdochl290k/train1-dpdochl290k-rawdata.xvg'
        # HMMphasepath='/data/gulab/yzdai/data4/atomdensity/HMM/all_train/psmdopochl/train7-psmdopochl300k-rawdata.xvg'

        mkdir -p "$OUTPATH/phaseplot/voronoi"  
        python $SCRIPT3 -trj $TRJ -pdb $PDB -start $START -end $END \
            -n_gap $N_GAP \
            -leaflet $LEAFLET -sys $SYS \
            -phasepath $phasepath -HMMphasepath $HMMphasepath \
            -voronoi_out $voronoi_out
        echo "Finished Voronoi frames ${START} to ${END}."
    fi
done
echo "All frames processed successfully!"
