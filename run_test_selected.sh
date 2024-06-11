#!/usr/bin/env bash

# Example: ./run_test_selected.sh ~/somepath/resources/data/ssb_simplified/ 1 PROFILE
DATA_DIR=$1
SF=$2
MODE=$3

make clean
make

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

pushd "src/ssb"
sed -i "s/#define SF [^ ]*/#define SF $SF/g" ssb_utils.h
files=("q11.cu" "q12.cu" "q13.cu" "q21.cu" "q22.cu" "q23.cu" "q41.cu" "q42.cu" "q43.cu")
QUERIES=()

for file in "${files[@]}"
do
    filename=$(basename "$file" .cu)
    echo "Processing $filename"
    pushd "$SCRIPT_DIR"
    make bin/ssb/$filename
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

metrics=inst_per_warp,l2_utilization,unique_warps_launched,achieved_occupancy,gld_throughput,gld_transactions,gst_transactions,stall_memory_dependency
# ,dram_read_transactions,dram_write_transactions
for q in ${QUERIES[@]}
do
  if [[ $MODE == "PROFILE" ]]; then
    METRICS_OUTPUT_FILE=$METRICS_OUTPUT_DIR/metrics_SF_${SF}_${q}.csv
    rm -f $METRICS_OUTPUT_FILE
    nvprof --csv --metrics $metrics ./bin/ssb/$q 2>> $METRICS_OUTPUT_FILE 
    grep -v '^==' $METRICS_OUTPUT_FILE > temp.csv && mv temp.csv $METRICS_OUTPUT_FILE
  fi
  ./bin/ssb/$q >> $RAW_OUTPUT_FILE
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
  echo "Plotting: python3 plot_metric_comp.py $SF $METRICS_OUTPUT_DIR"
  python3 plot_metric_comp.py $SF $METRICS_OUTPUT_DIR
fi
