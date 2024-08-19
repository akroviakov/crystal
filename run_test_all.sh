#!/usr/bin/env bash

CRYSTAL_PATH="/home/kroviakov/lingodb/crystal"
LINGODB_PATH="/home/kroviakov/lingodb/lingo-db"
LINGODB_DB_GEN_SCRIPT="$LINGODB_PATH/tools/generate/ssb_simplified.sh"
DATA_DIR="$LINGODB_PATH/resources/data/ssb_simplified_tbl"

SF=$1
GENERATE_DATA=$2
SM=75

if [[ $GENERATE_DATA == 1 ]]; then
  pushd "$CRYSTAL_PATH/test"
  python util.py $DATA_DIR $SF transform
  popd
fi
QUERIES=()

pushd "$CRYSTAL_PATH"
make clean
make
popd

pushd "$CRYSTAL_PATH/src/ssb"
sed -i "s/#define SF [^ ]*/#define SF $SF/g" ssb_utils.h

for file in *.cu
do
    filename=$(basename "$file" .cu)
    echo "Processing $filename"
    pushd "$CRYSTAL_PATH"
    make SM=$SM bin/ssb/$filename
    popd
    QUERIES+=("$filename")
done
popd

REPORT=Crystal_SF_$SF
RAW_OUTPUT_FILE="$REPORT.txt"
CSV_OUTPUT_FILE="$REPORT.csv"

pushd "$CRYSTAL_PATH"
rm -rf $RAW_OUTPUT_FILE
rm -rf $CSV_OUTPUT_FILE

for q in ${QUERIES[@]}
do
  ./bin/ssb/$q --dataSetPath=$DATA_DIR  > $RAW_OUTPUT_FILE
done


echo "type,name,executionTime" > $CSV_OUTPUT_FILE
while IFS= read -r line
do
  if [[ $line == *"query"* && $line == *"time_query"* ]]; then
    query=q$(echo $line | grep -oP '(?<="query":)\d+').sql
    time_query=$(echo $line | grep -oP '(?<="time_query":)[\d.]+')
    echo "$query,$time_query" >> $CSV_OUTPUT_FILE
  fi
done < "$RAW_OUTPUT_FILE"

popd