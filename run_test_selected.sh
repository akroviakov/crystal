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
  BATCH_SIZE=20000
fi
echo "BATCHSIZE $BATCH_SIZE"
make clean
make

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

pushd "src/ssb"
sed -i "s/#define SF [^ ]*/#define SF $SF/g" ssb_utils.h
#  "q12.cu" "q13.cu" "q21.cu" "q22.cu" "q23.cu" "q41.cu" "q42.cu" "q43.cu"
files=("q11.cu" "q12.cu" "q13.cu" "q21.cu") #("q11.cu" "q12.cu" "q13.cu" "q21.cu" "q22.cu" "q23.cu" "q41.cu" "q42.cu" "q43.cu")
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
metricsNCU=smsp__average_inst_executed_per_warp,l1tex__m_l1tex2xbar_write_bytes,l1tex__m_xbar2l1tex_read_bytes,l1tex__t_bytes_lookup_hit,l1tex__t_bytes_lookup_miss,sm__sass_data_bytes_mem_global_op_ld
metricsNCU1=gpu__time_duration.sum
# metricsNCU1+=,dram__bytes.sum
# metricsNCU1+=,lts__t_sectors_srcunit_tex_op_read.sum
# metricsNCU1+=,lts__t_requests_srcunit_tex_op_read.sum
# metricsNCU1+=,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
# metricsNCU1+=,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
metricsNCU1+=,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed # For how many cycles (%) SM had any work to do
metricsNCU1+=,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
# metricsNCU1+=,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
# metricsNCU1+=,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
metricsNCU1+=,smsp__warps_issue_stalled_lg_throttle.avg
metricsNCU1+=,smsp__warps_issue_stalled_long_scoreboard.avg
metricsNCU1+=,smsp__warps_launched.sum
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
  ./bin/ssb/$q --batchSize=$BATCH_SIZE --dataSetPath=$DATA_DIR >> $RAW_OUTPUT_FILE
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
