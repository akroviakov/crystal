import os
import subprocess
import sys
import shutil
import argparse
import re
import glob

import plot_vec_comp
import ncu_metrics
import plot_par_model_comp

# When running the first time: 
# python3 run_tests.py --data-dir=/SOMEPATH/crystal/test/ssb/data/ --sf=10 --mode=PROFILE --create-dataset=TRUE

# Afterwards (--mode is optional): 
# python3 run_tests.py --data-dir=/SOMEPATH/crystal/test/ssb/data/ --sf=10 --mode=PROFILE

def run_command(command, cwd=None, stdout=None):
    """Run a shell command and capture its output."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, stdout=stdout)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout

def runQueryAndAppendOutput(command, raw_output_file):
    output = run_command(command)
    with open(raw_output_file, 'a') as raw_file:
        raw_file.write(output)

def convertCrystalStdOutToCSV(raw_output_file, csv_output_file):
    with open(raw_output_file, 'r') as infile, open(csv_output_file, 'a') as outfile:
        outfile.write("type,name,executionTime,batch_size\n") 
        for line in infile:
            if "query" in line and "time_query" in line:
                type_ = re.search(r'(?<="type":)\w+', line)
                query = re.search(r'(?<="query":)\d+', line)
                time_query = re.search(r'(?<="time_query":)[\d.]+', line)
                batch_size = re.search(r'(?<="batch_size":)[\d.]+', line)

                if all([type_, query, time_query]):
                    if(batch_size):
                        outfile.write(f"{type_.group()},q{query.group()}.sql,{time_query.group()},{batch_size.group()}\n")
                    else:
                        outfile.write(f"{type_.group()},q{query.group()}.sql,{time_query.group()},0\n")

def compileFromList(original_script_dir, cu_files):
    queries = []
    for cu_file in cu_files:
        filename = os.path.splitext(os.path.basename(cu_file))[0]
        print(f"Processing {filename}")
        os.chdir(original_script_dir)
        make_command = f"make SM={args.sm} bin/ssb/{filename}"
        run_command(make_command)
        queries.append(filename)
    return queries

def runOneShotTest(original_script_dir, args, cu_files):
    report_name = f"Crystal_Vec_vs_Comp_SF_{args.sf}"
    measurements_dir = os.path.join(original_script_dir, f"Measurements/SF_{args.sf}")
    metrics_output_dir = os.path.join(measurements_dir, "profiler_metrics")
    os.makedirs(measurements_dir, exist_ok=True)
    os.makedirs(metrics_output_dir, exist_ok=True)

    raw_output_file = os.path.join(measurements_dir, f"{report_name}.txt")
    csv_output_file = os.path.join(measurements_dir, f"{report_name}.csv")

    for file in [raw_output_file, csv_output_file]:
        if os.path.exists(file):
            os.remove(file)

    queries = compileFromList(original_script_dir, cu_files)

    for q in queries:
        if args.mode == "PROFILE":
            metrics_output_file = os.path.join(metrics_output_dir, f"metrics_SF_{args.sf}_{q}")
            if os.path.exists(metrics_output_file):
                os.remove(metrics_output_file)
            metrics_string = ",".join(ncu_metrics.metricToSemantics.keys())
            ncu_command = (
                f"ncu --metrics {metrics_string} -f --export {metrics_output_file} ./bin/ssb/{q} --batchSize={args.batch_size} --dataSetPath={args.data_dir}"
            )
            run_command(ncu_command)
            run_command(f"ncu --import {metrics_output_file}.ncu-rep --csv > {metrics_output_file}.csv")
            os.remove(f"{metrics_output_file}.ncu-rep")

        simple_run_command = f"./bin/ssb/{q} --batchSize={args.batch_size} --dataSetPath={args.data_dir}"
        runQueryAndAppendOutput(simple_run_command, raw_output_file)

    convertCrystalStdOutToCSV(raw_output_file, csv_output_file)
    if args.mode == "PROFILE":
        print(f"Running: python3 ncu_metrics.py {metrics_output_dir} {args.sf}")
        for p in glob.glob(f"{metrics_output_dir}/*.csv"):
            ncu_metrics.plot_parallelism_comparison(p, args.sf)

    print(f"Running: python3 plot_vec_comp.py {csv_output_file} {args.sf}")
    plot_vec_comp.plot_execution_times(csv_output_file, args.sf)

def runParallelismModelTest(original_script_dir, args, cu_files):
    batch_sizes = [5000, 10000, 20000, 40000, 50000, 100000, 200000, 400000, 800000, 1000000, 3000000, 10000000, 32000000, 64000000]
    report_name = f"Crystal_Vec_vs_Comp_SF_{args.sf}"
    measurements_dir = os.path.join(original_script_dir, f"Measurements_Par/SF_{args.sf}")
    os.makedirs(measurements_dir, exist_ok=True)
    raw_output_file = os.path.join(measurements_dir, f"{report_name}.txt")
    csv_output_file = os.path.join(measurements_dir, f"{report_name}.csv")
    for file in [raw_output_file, csv_output_file]:
        if os.path.exists(file):
            os.remove(file)

    queries = compileFromList(original_script_dir, cu_files)

    for q in queries:
        for batch_size in batch_sizes:
            simple_run_command = f"./bin/ssb/{q} --batchSize={batch_size} --dataSetPath={args.data_dir}"
            runQueryAndAppendOutput(simple_run_command, raw_output_file)
    convertCrystalStdOutToCSV(raw_output_file, csv_output_file)

    print(f"Running: python3 plot_par_model_comp.py {csv_output_file} {args.sf}")
    plot_par_model_comp.plot_execution_time(csv_output_file, args.sf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs for the script.")
    parser.add_argument('--data-dir', required=True, help="Data directory")
    parser.add_argument('--sf', required=True, help="Scale factor")
    parser.add_argument('--mode', default="", help="PROFILE for metrics")
    parser.add_argument('--batch-size', default=0, type=int, help="Batch size")
    parser.add_argument('--create-dataset', default=False, type=bool, help="Whether to create dataset (or it already exists)")
    parser.add_argument('--sm', default=75, type=int, help="SM version (default: 75)")
    args = parser.parse_args()
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    if args.create_dataset:
        print("Creating Dataset")
        os.chdir(os.path.join(script_dir, "test"))
        run_command("cd ssb/dbgen && make && cd ../loader && make && cd ../../")
        run_command(f"python3 util.py ssb {args.sf} gen")
        run_command(f"python3 util.py ssb {args.sf} transform")
        os.chdir(script_dir)

    run_command("make clean && make")

    ssb_dir = os.path.join(script_dir, "src/ssb")
    os.chdir(ssb_dir)
    all_queries=["q11.cu", "q12.cu", "q13.cu", "q21.cu", "q22.cu", "q23.cu", "q31.cu", "q32.cu","q33.cu","q34.cu","q41.cu","q42.cu","q43.cu"]
    selected_queries=["q11.cu", "q12.cu", "q13.cu", "q21.cu", "q31.cu", "q43.cu"]
    run_command(f"sed -i 's/#define SF [^ ]*/#define SF {args.sf}/g' ssb_utils.h")
    runOneShotTest(script_dir, args, all_queries)
    runParallelismModelTest(script_dir, args, ["q11.cu", "q21.cu", "q43.cu"])
