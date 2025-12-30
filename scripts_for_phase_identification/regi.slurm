#!/bin/bash
#SBATCH -J r3k8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8          # 申请 8 个 CPU 给并行的 8 个子任务
###SBATCH --gres=gpu:1
#SBATCH -w node1
#SBATCH -o r3k8.out
#SBATCH -e r3k8.err

set -euo pipefail

# 避免数学/BLAS 在每个进程内再多线程，确保一进程=一核
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_HOME="/data/gulab/yzdai/anaconda3"
ENV_NAME="density"


# 先确保是 bash，然后用 conda.sh 激活
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  # 兜底：直接把 conda 可执行加进 PATH（仅当 node2 上这个路径确实存在）
  export PATH="$CONDA_HOME/bin:$PATH"
  conda activate "$ENV_NAME"
fi

SCRIPT1="/data/gulab/yzdai/data4/phase_identification/scripts_for_phase_identification/phase_identification_pure_lipids.py"
SCRIPT2="/data/gulab/yzdai/data4/phase_identification/plot/scripts/plot_phase.py"
SCRIPT3="/data/gulab/yzdai/data4/phase_identification/plot/scripts/plot_voronoi_with_chol.py"
SCRIPT4="/data/gulab/yzdai/data4/phase_identification/plot/scripts/plot_voronoi_binary.py"

N_GAP='5'
BIN_WIDTH='3'
CAL_RATIO='False'

# -----------------------dpdochl-------------------------------------------
SYS='dpdochl280k'
PDB="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_280k/snapshot9.99us.gro"
TRJ="/data/gulab/yzdai/dyz_project1/data/dppc_dopc_chl_280k/dpdochl280k-8us-pbc.xtc"
LEAFLET='/data/gulab/yzdai/data4/phase_identification/leaflet/dpdochl280k-leaflet.xvg'

param="/data/gulab/yzdai/data4/phase_identification/phase_out/$SYS/parameters.json"
# 注意：把 primary lipid 的 ID 串按原脚本保持一致
all_dppc_id=($(seq 1 202) $(seq 577 778))
all_dppc_id_str=$(IFS=,; echo "${all_dppc_id[*]}")

TOTAL_FRAMES=8000      # dpdochl280k
STEP=1000

BASE_OUTPATH="/data/gulab/yzdai/data4/phase_identification/phase_out/${SYS}/"
mkdir -p "$BASE_OUTPATH"

# -------------------- 并行控制：8 个并发槽位 --------------------
MAX_JOBS=8
fifo="/tmp/sem.$$"
mkfifo "$fifo"
exec 9<>"$fifo"
rm -f "$fifo"
# 预装 8 个令牌
for _ in $(seq 1 $MAX_JOBS); do
  echo >&9
done

# 定义单个区间的处理函数
process_range() {
  local START=$1
  local END=$2
  local OUTPATH="${BASE_OUTPATH}${START}-${END}/"

  echo "Processing frames ${START} to ${END}..."
  mkdir -p "$OUTPATH"
  mkdir -p "$OUTPATH/phaseplot/regi"

  # 需要时放开相位识别计算
  # python "$SCRIPT1" -trj "$TRJ" -pdb "$PDB" -start "$START" -end "$END" \
  #   -bin_width "$BIN_WIDTH" -n_gap "$N_GAP" \
  #   -leaflet "$LEAFLET" -sys "$SYS" -cal_ratio "$CAL_RATIO" -outpath "$OUTPATH" \
  #   -primary_lipid "$all_dppc_id_str" -param "$param"

  # 画相图
  python "$SCRIPT2" -trj "$TRJ" -pdb "$PDB" -b "$START" -e "$END" \
    -bin_width 2 -inte 5 \
    -phasepath "$OUTPATH" -sys "$SYS"

  # 如需 Voronoi，可按条件放开
  # if [ "$START" -eq 8000 ]; then
  #   phasepath="${OUTPATH}${SYS}-rawdata.xvg"
  #   voronoi_out="$OUTPATH/phaseplot/voronoi/"
  #   mkdir -p "$voronoi_out"
  #   HMMphasepath='/data/gulab/yzdai/data4/atomdensity/HMM/dpdochl290k/train1-dpdochl290k-rawdata.xvg'
  #   python "$SCRIPT3" -trj "$TRJ" -pdb "$PDB" -start "$START" -end "$END" \
  #     -n_gap "$N_GAP" -leaflet "$LEAFLET" -sys "$SYS" \
  #     -phasepath "$phasepath" -HMMphasepath "$HMMphasepath" \
  #     -voronoi_out "$voronoi_out"
  # fi

  echo "Finished frames ${START} to ${END}."
}

export -f process_range
export BASE_OUTPATH SYS TRJ PDB LEAFLET SCRIPT1 SCRIPT2 SCRIPT3 SCRIPT4 \
       N_GAP BIN_WIDTH CAL_RATIO param all_dppc_id_str

# 逐区间提交到并行“槽位”
for ((START=0; START<TOTAL_FRAMES; START+=STEP)); do
  END=$(( (START + STEP < TOTAL_FRAMES) ? (START + STEP) : TOTAL_FRAMES ))

  # 取一个令牌（若 8 个都占满则在此阻塞，起到限流作用）
  read -u 9
  {
    # 真正跑任务（在子 shell 中）
    process_range "$START" "$END"
    # 归还令牌
    echo >&9
  } &
done

wait   # 等所有后台任务结束
exec 9>&- 9<&-

echo "All frames processed successfully!"
