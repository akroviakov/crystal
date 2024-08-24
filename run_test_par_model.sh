#!/usr/bin/env bash

# Example: ./run_test_selected.sh ~/somepath/resources/data/ssb_simplified/ 1 PROFILE
DATA_DIR=$1
SF=$2
SM=75

make clean
make

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

pushd "src/ssb"
sed -i "s/#define SF [^ ]*/#define SF $SF/g" ssb_utils.h
files=("q11.cu" "q21.cu" "q43.cu")
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

MEASUREMENTS_DIR=Measurements_Par/SF_$SF
mkdir -p $MEASUREMENTS_DIR
METRICS_OUTPUT_DIR=$MEASUREMENTS_DIR/profile_metrics
mkdir -p $METRICS_OUTPUT_DIR

REPORT="Crystal_Vec_vs_Comp_SF_$SF"
RAW_OUTPUT_FILE="$MEASUREMENTS_DIR/$REPORT.txt"
CSV_OUTPUT_FILE="$MEASUREMENTS_DIR/$REPORT.csv"

rm -rf $RAW_OUTPUT_FILE
rm -rf $CSV_OUTPUT_FILE
BATCH_SIZES=(5000 10000 20000 40000 50000 100000 200000 400000 800000 1000000 3000000 10000000 32000000)
metrics=inst_per_warp,l2_utilization,unique_warps_launched,achieved_occupancy,gld_throughput,gld_transactions,gst_transactions,stall_memory_dependency
# ,dram_read_transactions,dram_write_transactions
for q in ${QUERIES[@]}
do
  for bsize in ${BATCH_SIZES[@]}
  do
    ./bin/ssb/$q --batchSize=$bsize --dataSetPath=$DATA_DIR >> $RAW_OUTPUT_FILE
  done
done

touch $RAW_OUTPUT_FILE
touch $CSV_OUTPUT_FILE

echo "type,name,batch_size,executionTime" > $CSV_OUTPUT_FILE
while IFS= read -r line
do
  if [[ $line == *"query"* && $line == *"time_query"* ]]; then
    type=$(echo $line | grep -oP '(?<="type":)\w+')
    query=q$(echo $line | grep -oP '(?<="query":)\d+').sql
    time_query=$(echo $line | grep -oP '(?<="time_query":)[\d.]+')
    batch_size=$(echo $line | grep -oP '(?<="batch_size":)[\d.]+')
    echo "$type,$query,$batch_size,$time_query" >> $CSV_OUTPUT_FILE
  fi
done < "$RAW_OUTPUT_FILE"

echo "Plotting: python3 plot_par_model_comp.py $SF $CSV_OUTPUT_FILE"
python3 plot_par_model_comp.py $SF $CSV_OUTPUT_FILE
