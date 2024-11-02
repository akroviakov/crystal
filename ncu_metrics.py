import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
import matplotlib.colors as mcolors
import re

# https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder

# SMSP: each SM is partitioned into four processing blocks, called SM sub partitions. 
#   The SM sub partitions are the primary processing elements on the SM. 
#   A sub partition manages a fixed size pool of warps.

# lts: Level 2 (L2) Cache Slice is a sub-partition of the Level 2 cache. 

metricToSemantics = {
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed" : "L2 requests (of peak)",
    "lts__t_sectors_lookup_hit.sum" : "L2 hits",
    "lts__t_sectors_lookup_miss.sum" : "L2 misses",
    "lts__t_sector_hit_rate.pct" : "L2 hit rate",
    "lts__t_sectors_srcunit_tex_op_read.sum.per_second" : "L2->L1 sectors (per second)",

    "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum.pct_of_peak_sustained_elapsed" : "L2->L1 bandwidth(of peak)",
    "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum" : "L2->L1 sectors",
    "l1tex__lsu_writeback_active_mem_lg.sum.pct_of_peak_sustained_elapsed" : "L1 utilization (of peak)",
    "l1tex__t_sector_hit_rate.pct" : "L1 hit rate",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum" : "L1 sectors loaded",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum" : "L1 load requests",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum" : "L1 sectors written",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum" : "L1 store requests",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum" : "Num. warps hit L1",

    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed" : "Cycles with work",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct" : "Global memory stalls pct",
    "smsp__warps_issue_stalled_long_scoreboard.avg" : "Global memory stalls", #total
    "smsp__average_warp_latency_per_inst_issued.ratio" : "Instruction latency",
    "smsp__warps_eligible.avg.per_cycle_active" : "Eligible warps per cycle",
    "smsp__inst_executed.sum" : "Executed instructions",
    "smsp__warps_issue_stalled_lg_throttle.avg" : "LSU throttle stalls",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct" : "LSU throttle stalls pct",

    "smsp__warps_launched.sum" : "Num. launched warps",

    "dram__bytes_read.sum.per_second" : "Read throughput",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed" : "Read throughput of peak",
    "dram__bytes.sum" : "Total DRAM traffic",

    "gpu__time_duration.sum" : "Kernel duration"
}

hatch_patterns = ['\\', '-', '+', 'x', 'o', 'O', '.', '*']
color_list = ['#0072B2', '#D55E00', '#56B4E9', '#009E73', '#F0E442', '#E69F00', '#CC79A7', '#000000']

def convertMetrics(metric_name):
    found = metricToSemantics.get(metric_name)
    # print(f"{metric_name}  - > {found}")
    return metric_name if found == None else found 

def extract_filename(path):
    base_name = os.path.basename(path) 
    file_name = os.path.splitext(base_name)[0]
    return file_name

def extract_kernel_type(name):
    start = name.find('void ') + len('void ')
    end = name.find('>(')
    return name[start:end]


suffixes = ['_Compiled', 'Compiled', '_compiled', 'compiled']
def extract_name(name):
    filteredName = extract_kernel_type(name)
    for suffix in suffixes:
        if filteredName.endswith(suffix):
            return filteredName[:-len(suffix)]
    return filteredName

def get_kernel_type(name):
    resultStr=""
    if "<0" in name:
        resultStr += "Vectorized"
    elif "<1" in name:
        resultStr += "VectorizedOpt"
    elif "<2" in name:
        resultStr += "CompiledBatchToSM"
    elif "<3" in name:
        resultStr += "CompiledBatchToSMLocals"
    elif "<4" in name:
        resultStr += "CompiledBatchToGPU"
    elif "<5" in name:
        resultStr += "VectorizedSMEM"
    elif "<6" in name:
        resultStr += "VectorizedOptSMEM"
    else:
        if "build" in name:
            resultStr += "Vectorized"
        else:        
            print(f"UnknonwType: {name}")
    return resultStr
        
def determine_type(name):
    return 'Compiled' if any(name.endswith(suffix) for suffix in suffixes) else 'Vectorized'

def convert_metric_value(value):
    if not isinstance(value, str):
        return value
    value = value.replace('.', '')
    value = value.replace(',', '.')
    return pd.to_numeric(value, errors='coerce')


def coalesce_calc(df, col1_req, col2_tx, result_index, drop_originals=True):
    if col1_req in df.columns and col2_tx in df.columns:
        ratio = df[col1_req] / df[col2_tx]
        df[result_index] = ratio * 100
        print(df)
        if drop_originals:
            df = df.drop(columns=[col1_req, col2_tx])
    else:
        raise ValueError(f"Columns {col1_req} and/or {col2_tx} do not exist in the DataFrame.")
    return df

def hitRate(df, col_hit, col_miss, result_colname, drop_originals=True):
    if col_hit in df.columns and col_miss in df.columns:
        df[result_colname] = (df[col_hit] / (df[col_hit] + df[col_miss])) * 100
        if drop_originals:
            df = df.drop(columns=[col_hit, col_miss])
    else:
        print(df)
        raise ValueError(f"Columns {col_hit} and/or {col_miss} do not exist in the DataFrame.")
    return df

def readPreprocess(file_path):
    df = pd.read_csv(file_path)
    df['Metric Name'] = df['Metric Name'].apply(convertMetrics)
    df['Metric Value'] = df['Metric Value'].apply(convert_metric_value)
    df['ShortName'] = df['Kernel Name'].apply(extract_name)
    df['Type'] = df['Kernel Name'].apply(get_kernel_type)
    df['Metric Unit'].fillna('ratio', inplace=True)
    pivot_df = df.pivot_table(
        index=['ShortName', 'Type'],
        columns=['Metric Name', 'Metric Unit'],
        values='Metric Value',
        aggfunc='median'
    )
    mask = ~pivot_df.index.get_level_values('ShortName').str.contains('build', case=False, na=False)
    pivot_df = pivot_df[mask]
    return pivot_df

def plot_metric(file_path, SF):
    df = readPreprocess(file_path)
    # pivot_df = coalesce_calc(pivot_df, ('L2 -> L1 read requests', 'request'), ('L2 -> L1 read TX', 'sector'), ('L2 read coalescing', '%'))
    pivot_df = coalesce_calc(df, ('L1 load requests', 'request'), ('L1 sectors loaded', 'sector'), ('Read coalescing', '%'))
    pivot_df = hitRate(df, ("L2 hits", "sector") , ("L2 misses", "sector"), ("L2 hit rate", "%"))

    pivot_df = pivot_df.applymap(lambda x: np.ceil(x) if isinstance(x, (float, np.float64)) else x)
    pivot_df.columns = [f'{col[0]}({col[1]})' if col[1] else col[0] for col in pivot_df.columns]
    pivot_df = pivot_df.rename_axis(None, axis=1)

    unique_kernels = pivot_df.index.get_level_values('ShortName').unique()
    metric_columns = pivot_df.columns
    # color_list = plt.cm.tab10.colors  # Use a colormap that has at least 10 colors
    color_rgba = {color: mcolors.to_rgba(color) for color in color_list}

    color_to_hatch = dict(zip(color_rgba.values(), hatch_patterns))

    nrows = len(unique_kernels)
    ncols = len(metric_columns)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))
    axes = axes.flatten()
    pivot_df_unindexed = pivot_df.reset_index()
    for rowIdx, kernel_name in enumerate(unique_kernels):
        for colIdx, metric in enumerate(metric_columns):
            ax = axes[rowIdx * ncols + colIdx]
            subset = pivot_df_unindexed[pivot_df_unindexed['ShortName'] == kernel_name]
            pivot_plot_df = subset.pivot(index='ShortName', columns='Type', values=metric)
            pivot_plot_df.plot(kind='bar', color=color_list[:len(pivot_plot_df.columns)], ax=ax, edgecolor='black', legend=False, zorder=2, width=0.6)
            for i, bar in enumerate(ax.patches):
                bar.set_hatch(color_to_hatch[bar.get_facecolor()]) 
                # ax.annotate(f'{bar.get_height():.2f}', 
                #             (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                #             ha='center', va='center', 
                #             xytext=(0, 5), 
                #             textcoords='offset points',
                #             fontsize=6)  
            ax.grid(axis='y',zorder=1)
            ax.set_title(metric)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
            ax.margins(x=0.01)
        last_subplot = axes[rowIdx * ncols + ncols-1]
        handles, labels = last_subplot.get_legend_handles_labels()
        last_subplot.legend(handles, labels, loc='center left', bbox_to_anchor=(0.1, -0.5), fontsize=10)
    fig.tight_layout(pad=0.5)  # Adjust padding as needed
    plots_dir=f"Plots/SF_{SF}/Metrics"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    fig.savefig(f"{plots_dir}/Comparison_{extract_filename(file_path)}.png", dpi=300)
    fig.savefig(f"{plots_dir}/Comparison_{extract_filename(file_path)}.pdf")


def reduceGBUnits(dataframe, column_name):
    conversion_map = {
        'Gbyte': 1024**3, 
        'Mbyte': 1024**2,  
        'Kbyte': 1024   
    }
    for metric_unit in conversion_map:
        column = (column_name, metric_unit)
        if column in dataframe.columns:
            dataframe[(column_name, 'Byte')] = dataframe.get((column_name, 'Byte'), 0) + dataframe[column].fillna(0) * conversion_map[metric_unit]
    columns_to_drop = [((column_name, metric)) for metric in conversion_map.keys() if (column_name, metric) in dataframe.columns]
    dataframe = dataframe.drop(columns=columns_to_drop)
    dataframe = dataframe[[col for col in [(column_name, 'Byte')] + [c for c in dataframe.columns if c != (column_name, 'Byte')]]]
    return dataframe

def reduceGBPerSUnits(dataframe, column_name):
    conversion_map = {
        'Gbyte/second': 1024**3, 
        'Mbyte/second': 1024**2,  
        'Kbyte/second': 1024   
    }
    for metric_unit in conversion_map:
        column = (column_name, metric_unit)
        if column in dataframe.columns:
            dataframe[(column_name, 'Byte/s')] = dataframe.get((column_name, 'Byte/s'), 0) + dataframe[column].fillna(0) * conversion_map[metric_unit]
    columns_to_drop = [((column_name, metric)) for metric in conversion_map.keys() if (column_name, metric) in dataframe.columns]
    dataframe = dataframe.drop(columns=columns_to_drop)
    dataframe = dataframe[[col for col in [(column_name, 'Byte/s')] + [c for c in dataframe.columns if c != (column_name, 'Byte/s')]]]
    return dataframe

def plot_parallelism_comparison(file_path, SF, reduced_plot=False, exclude_batch_to_gpu=False, vector_smem=False, compVsVecOptSMEM=False):
    df = readPreprocess(file_path)
    # pivot_df = hitRate(df, ("L2 hits", "sector") , ("L2 misses", "sector"), ("L2 hit rate", "%"))
    pivot_df = df[["Total DRAM traffic", "Read throughput of peak", "Executed instructions",  "Instruction latency",
                   "L2 hit rate",  "Global memory stalls", "Global memory stalls pct", "LSU throttle stalls pct"]]

    # pivot_df = reduceGBPerSUnits(pivot_df, "Read throughput")
    pivot_df = reduceGBUnits(pivot_df, "Total DRAM traffic")

    pivot_df = pivot_df.applymap(lambda x: np.ceil(x) if isinstance(x, (float, np.float64)) else x)
    pivot_df.columns = [f'{col[0]}({col[1]})' if col[1] else col[0] for col in pivot_df.columns]
    pivot_df = pivot_df.rename_axis(None, axis=1)

    unique_kernels = pivot_df.index.get_level_values('ShortName').unique()
    metric_columns = pivot_df.columns
    pivot_df_unindexed = pivot_df.reset_index()

    # pivot_df_unindexed = merge_traffic_columns(pivot_df_unindexed)

    color_rgba = {color: mcolors.to_rgba(color) for color in color_list}
    color_to_hatch = dict(zip(color_rgba.values(), hatch_patterns))
    ncols = len(metric_columns) // 2 
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(10, 5))
    axes =  axes.flatten()
    for colIdx, metric in enumerate(metric_columns):
        ax = axes[colIdx]
        subset = pivot_df_unindexed[["Type", metric]]
        # ncu versions can have different output formats, this code is for ncu from CUDA 12.6.
        if reduced_plot:
            subset = subset[subset['Type'].isin(["CompiledBatchToSM", "VectorizedOpt"])]
        elif exclude_batch_to_gpu:
            subset = subset[subset['Type'].isin(["CompiledBatchToSM", "VectorizedOpt", "Vectorized"])]
        elif vector_smem:
            subset = subset[subset['Type'].isin(["VectorizedOpt", "Vectorized", "VectorizedOptSMEM", "VectorizedSMEM"])]
        elif compVsVecOptSMEM:
            subset = subset[subset['Type'].isin(["CompiledBatchToSM", "VectorizedOpt", "VectorizedOptSMEM"])]
        if subset.empty:
            continue
        subset.set_index('Type')
        subset = subset.pivot_table(index=None, columns='Type', values=metric)
        subset.plot(kind='bar', color=color_list[:len(subset.columns)], ax=ax, edgecolor='black', legend=False, zorder=2, width=0.6)
        for i, bar in enumerate(ax.patches):
            bar.set_hatch(color_to_hatch[bar.get_facecolor()]) 
        ax.grid(axis='y',zorder=1)
        ax.set_title(metric)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels([])
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
        ax.margins(x=0.01)
    last_subplot = axes[len(metric_columns)-1]
    handles, labels = last_subplot.get_legend_handles_labels()
    last_subplot.legend(handles, labels, loc='center left', bbox_to_anchor=(0.1, -0.6), fontsize=10)
    fig.tight_layout(pad=0.5)  # Adjust padding as needed

    plots_dir=f"Plots/SF_{SF}/Metrics"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    suffix = ""
    if reduced_plot:
        suffix = "_reduced"
    elif exclude_batch_to_gpu:
        suffix = "_no_batch_to_gpu"
    elif vector_smem:
        suffix = "_vector_smem"
    elif compVsVecOptSMEM:
        suffix = "_bestvec_smem"
    fig.savefig(f"{plots_dir}/Comparison_for_{extract_filename(file_path)}{suffix}.png", dpi=300)
    fig.savefig(f"{plots_dir}/Comparison_for_{extract_filename(file_path)}{suffix}.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('CSV_DIR', metavar='CSV_DIR', type=str, help='directory with measurements')
    parser.add_argument('SF', metavar='SF', type=int, help='scale factor')

    args = parser.parse_args()
    for p in glob.glob(f"{args.CSV_DIR}/*.csv"):
        # plot_metric(p, args.SF)
        plot_parallelism_comparison(p, args.SF)
        # plot_parallelism_comparison(p, args.SF, True)
        # plot_parallelism_comparison(p, args.SF, False, True)
        # plot_parallelism_comparison(p, args.SF, False, False, True)
        plot_parallelism_comparison(p, args.SF, False, False, False, True)
