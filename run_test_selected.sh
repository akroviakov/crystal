#!/usr/bin/env bash

# Example: ./run_test_selected.sh ~/somepath/resources/data/ssb_simplified/ 1 PROFILE
# ./run_test_selected.sh ~/TUM/Master/crystal/test/ssb/data/ 1 PROFIL
DATA_DIR=$1
SF=$2
MODE=$3
BATCH_SIZE=$4
SM=75
echo "BATCHSIZE $BATCH_SIZE"

if [ $# -lt 4 ]; then
  BATCH_SIZE=0
fi
echo "BATCHSIZE $BATCH_SIZE"
make clean
make

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

pushd "src/ssb"
sed -i "s/#define SF [^ ]*/#define SF $SF/g" ssb_utils.h
#  "q12.cu" "q13.cu" "q21.cu" "q22.cu" "q23.cu" "q41.cu" "q42.cu" "q43.cu"
files=("q13.cu" "q21.cu") #("q11.cu" "q12.cu" "q13.cu" "q21.cu" "q22.cu" "q23.cu" "q41.cu" "q42.cu" "q43.cu")
QUERIES=()

for file in "${files[@]}"
do
    filename=$(basename "$file" .cu)
    echo "Processing $filename"
    pushd "$SCRIPT_DIR"
    make SM=$SM bin/ssb/$filename
    popd
    QUERIES+=("$filename")
done
popd

MEASUREMENTS_DIR=Measurements/SF_$SF
mkdir -p $MEASUREMENTS_DIR
METRICS_OUTPUT_DIR=$MEASUREMENTS_DIR/profile_metrics
mkdir -p $METRICS_OUTPUT_DIR

REPORT="Crystal_Vec_vs_Comp_SF_$SF"
RAW_OUTPUT_FILE="$MEASUREMENTS_DIR/$REPORT.txt"
CSV_OUTPUT_FILE="$MEASUREMENTS_DIR/$REPORT.csv"

rm -rf $RAW_OUTPUT_FILE
rm -rf $CSV_OUTPUT_FILE

# gld_efficiency - coalescing
# warp_execution_efficiency - thread divergence or the kernel was not launched with a multiple of 32 threads per block
# sm_efficiency - Workload balance. Ratio of cycles that a SM had at least 1 active warp to the total number of cycles executed in the measurement.
# stall_not_selected - if this number is high then part or all of the kernel has sufficient occupancy (active_warps) to hide instruction latency.
metricsNVPROF=inst_per_warp,l2_utilization,,warp_execution_efficiency,sm_efficiency,achieved_occupancy,gld_efficiency,gld_throughput,gld_transactions,stall_memory_dependency
metricsNCU1=""
metricsNCU1+=,dram__bytes.sum
# ---------- Experiment to compare parallelisation models (q11), expect toSM to be better
# L2 pct of peak:
metricsNCU1+=,lts__t_sectors.avg.pct_of_peak_sustained_elapsed
metricsNCU1+=,lts__t_sectors.avg
# L2 hit rate: 100 * lts__t_sectors_lookup_hit.sum / (lts__t_sectors_lookup_hit.sum + lts__t_sectors_lookup_miss.sum)
metricsNCU1+=,lts__t_sectors_lookup_hit.sum
metricsNCU1+=,lts__t_sectors_lookup_miss.sum
# Number of sectors read from L2 to L1 miss stage:
metricsNCU1+=,l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum.pct_of_peak_sustained_elapsed
metricsNCU1+=,l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum
# of cycles where local/global writeback interface was active
metricsNCU1+=,l1tex__lsu_writeback_active_mem_lg.sum.pct_of_peak_sustained_elapsed
metricsNCU1+=,l1tex__lsu_writeback_active_mem_lg.sum
# Read throughput
metricsNCU1+=,dram__bytes_read.sum.per_second
# Warp Memory stalls (% of total stalls)
metricsNCU1+=,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
# Latency per inst (more latency is due to stalls, the long score board should be the main contributor to stalls)
metricsNCU1+=,smsp__average_warp_latency_per_inst_issued.ratio


# ---------- Experiment to compare random access queries (q21)
#  #of wavefronts sent to Data-Stage from T-Stage for global loads
metricsNCU1+=,l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum
#  #of sectors read from L2 into L1TEX M-Stage for local/global loads
metricsNCU1+=,l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum.pct_of_peak_sustained_elapsed
metricsNCU1+=,l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum.sum
# L2 cache throughput for loads orgignated from L1 
metricsNCU1+=,lts__t_sectors_srcunit_tex_op_read.sum.per_second
# Device read throughput
metricsNCU1+=,dram__bytes_read.sum.per_second
# cumulative # of warps eligible to issue an instruction
metricsNCU1+=,smsp__warps_eligible.avg.per_cycle_active
# Warp Memory stalls (% of total stalls)
metricsNCU1+=,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
# Latency per inst (more latency is due to stalls, the long score board should be the main contributor to stalls)
metricsNCU1+=,smsp__average_warp_latency_per_inst_issued.ratio

# #of executed instructions
metricsNCU1+=,smsp__inst_executed.sum


# L2 read coalescing
# metricsNCU1+=,lts__t_sectors_srcunit_tex_op_read.sum
# metricsNCU1+=,lts__t_requests_srcunit_tex_op_read.sum
# Global read coalescing (L1)
metricsNCU1+=,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
metricsNCU1+=,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
# Global write coalescing (L1)
metricsNCU1+=,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
metricsNCU1+=,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
# Efficiency: for how many cycles (%) SM had any work to do
metricsNCU1+=,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed 
# Global mem throughput
metricsNCU1+=,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
# Memory stalls
metricsNCU1+=,smsp__warps_issue_stalled_lg_throttle.avg
metricsNCU1+=,smsp__warps_issue_stalled_long_scoreboard.avg
# Launch stat
# metricsNCU1+=,smsp__warps_launched.sum

# metricsNCU1+=,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
# metricsNCU1+=,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct

# ,dram_read_transactions,dram_write_transactions,gst_transactions,unique_warps_launched
for q in ${QUERIES[@]}
do
  if [[ $MODE == "PROFILE" ]]; then
    METRICS_OUTPUT_FILE=$METRICS_OUTPUT_DIR/metrics_SF_${SF}_${q}
    rm -f $METRICS_OUTPUT_FILE

    ncu --metrics $metricsNCU1 -f --export $METRICS_OUTPUT_FILE ./bin/ssb/$q --batchSize=$BATCH_SIZE --dataSetPath=$DATA_DIR
    ncu --import $METRICS_OUTPUT_FILE.ncu-rep --csv > $METRICS_OUTPUT_FILE.csv
    rm -f $METRICS_OUTPUT_FILE.ncu-rep

    # nvprof --csv --metrics $metrics ./bin/ssb/$q --batchSize=$BATCH_SIZE --dataSetPath=$DATA_DIR #2>> $METRICS_OUTPUT_FILE.csv 
    # grep -v '^==' $METRICS_OUTPUT_FILE.csv > temp.csv && mv temp.csv $METRICS_OUTPUT_FILE.csv
  fi
  echo "Running ./bin/ssb/$q --batchSize=$BATCH_SIZE --dataSetPath=$DATA_DIR >> $RAW_OUTPUT_FILE"
  ./bin/ssb/$q --batchSize=$BATCH_SIZE --t=10 --dataSetPath=$DATA_DIR >> $RAW_OUTPUT_FILE
done

touch $RAW_OUTPUT_FILE
touch $CSV_OUTPUT_FILE

echo "type,name,executionTime" > $CSV_OUTPUT_FILE
while IFS= read -r line
do
  if [[ $line == *"query"* && $line == *"time_query"* ]]; then
    type=$(echo $line | grep -oP '(?<="type":)\w+')
    query=q$(echo $line | grep -oP '(?<="query":)\d+').sql
    time_query=$(echo $line | grep -oP '(?<="time_query":)[\d.]+')
    echo "$type,$query,$time_query" >> $CSV_OUTPUT_FILE
  fi
done < "$RAW_OUTPUT_FILE"

echo "Plotting: python3 plot_vec_comp.py $SF $CSV_OUTPUT_FILE"
python3 plot_vec_comp.py $SF $CSV_OUTPUT_FILE

if [[ $MODE == "PROFILE" ]]; then
  # NVPROF
  # echo "Plotting: python3 plot_metric_comp.py $SF $METRICS_OUTPUT_DIR"
  # python3 plot_metric_comp.py $SF $METRICS_OUTPUT_DIR

  # NCU
  echo "Plotting: python3 ncu_metrics.py $SF $METRICS_OUTPUT_DIR"
  python3 ncu_metrics.py $SF $METRICS_OUTPUT_DIR
fi
