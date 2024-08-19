import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
import matplotlib.colors as mcolors

def convertMetrics(metric_name):
    # https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder

    # SMSP: each SM is partitioned into four processing blocks, called SM sub partitions. 
    #   The SM sub partitions are the primary processing elements on the SM. 
    #   A sub partition manages a fixed size pool of warps.

    metricToSemantics = {
        "dram__bytes.sum" : "Total bytes DRAM ",
        "gpu__time_duration.sum" : "Kernel duration ",
        "smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed" : "Achieved compute bandwidth",
        "sm__sass_thread_inst_executed_op_integer_pred_on.sum.peak_sustained" : "Peak compute bandwidth",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed" : "GPU DRAM throughput (Avg)",
        "smsp__warps_issue_stalled_long_scoreboard.avg" : "Memory dep. stall (Avg)",
        "smsp__warps_issue_stalled_lg_throttle.avg" : "Load store unit is not available (Avg)",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct" : "Proportion of warps waiting memory dep.",
        "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct" : "Proportion of warps waiting on LSU",
        "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed" : "SM efficiency",
        "smsp__warps_launched.sum" : "# Warps launched",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM throughput assuming ideal load balancing (Avg)",
        "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed" : "SM memory instruction throughput (Avg)",
        "sm__warps_active.avg.pct_of_peak_sustained_active" : "Achieved Occupancy",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "Global Load TX (L1 loads)",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum" : "Local Load TX (L1 loads)",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum" : "Global Load (L1 loads) Requests", 
        "lts__t_sectors_srcunit_tex_op_read.sum" : "L2 to L1 read TX",
        "lts__t_requests_srcunit_tex_op_read.sum" : "L2 to L1 read requests",
        "dram__bytes_read.sum" : "DRAM to L2 read",
        "smsp__inst_executed_op_shared_ld.sum" : "SLM loads",
        "l1tex__throughput" : "L1 achieved throughput"
    }
    found = metricToSemantics.get(metric_name)
    # print(f"{metric_name}  - > {found}")

    return metric_name if found == None else found 

def extract_filename(path):
    base_name = os.path.basename(path) 
    file_name = os.path.splitext(base_name)[0]
    return file_name

def extract_kernel_type(name):
    start = name.find('void ') + len('void ')
    end = name.find('<(')
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
    if "(QueryVariant)0" in name:
        resultStr += "Vectorized"
    elif "(QueryVariant)1" in name:
        resultStr += "VectorizedOpt"
    elif "(Parallelism)0" in name:
        resultStr += "Compiled"
    elif "(Parallelism)1" in name:
        resultStr += "CompiledOmniSci"
    else:
        resultStr += "Compiled"
    return resultStr
        
def determine_type(name):
    return 'Compiled' if any(name.endswith(suffix) for suffix in suffixes) else 'Vectorized'

def convert_metric_value(value):
    if not isinstance(value, str):
        return value
    value = value.replace('.', '')
    value = value.replace(',', '.')
    return pd.to_numeric(value, errors='coerce')

def coalesce_calc(df, col1_index, col2_index, result_index, drop_originals=True):
    if col1_index in df.columns and col2_index in df.columns:
        ratio = df[col1_index] / df[col2_index]
        df[result_index] = ratio / 32
        if drop_originals:
            df = df.drop(columns=[col1_index, col2_index])
    else:
        raise ValueError(f"Columns {col1_index} and/or {col2_index} do not exist in the DataFrame.")
    return df

def plot_metric(file, SF):
    df = pd.read_csv(file)

    df['Metric Name'] = df['Metric Name'].apply(convertMetrics)
    df['Metric Value'] = df['Metric Value'].apply(convert_metric_value)
    df['ShortName'] = df['Kernel Name'].apply(extract_name)
    df['Type'] = df['Kernel Name'].apply(get_kernel_type)
    df['Metric Unit'].fillna('ratio', inplace=True)

    pivot_df = df.pivot_table(
        index=['ShortName', 'Type'],
        columns=['Metric Name', 'Metric Unit'],
        values='Metric Value',
        aggfunc='mean'
    )

    # pivot_df = coalesce_calc(pivot_df, ('L2 to L1 read requests', 'request'), ('L2 to L1 read TX', 'sector'), ('L2 read coalescing', '%'))
    # pivot_df = coalesce_calc(pivot_df, ('Global Load (L1 loads) Requests', 'request'), ('Global Load TX (L1 loads)', 'sector'), ('Read coalescing', '%'))


    pivot_df = pivot_df.applymap(lambda x: np.ceil(x) if isinstance(x, (float, np.float64)) else x)
    pivot_df.columns = [f'{col[0]}({col[1]})' if col[1] else col[0] for col in pivot_df.columns]
    pivot_df = pivot_df.rename_axis(None, axis=1)

    unique_kernels = pivot_df.index.get_level_values('ShortName').unique()
    metric_columns = pivot_df.columns
    # color_list = plt.cm.tab10.colors  # Use a colormap that has at least 10 colors
    color_list = ['b', 'r', 'g', 'y', 'c', 'm']
    color_rgba = {color: mcolors.to_rgba(color) for color in color_list}

    hatch_patterns = ['\\', '-', '+', 'x', 'o', 'O', '.', '*']
    color_to_hatch = dict(zip(color_rgba.values(), hatch_patterns))

    nrows = len(unique_kernels)
    ncols = len(metric_columns)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    axes = axes.flatten()
    pivot_df_unindexed = pivot_df.reset_index()
    for rowIdx, kernel_name in enumerate(unique_kernels):
        for colIdx, metric in enumerate(metric_columns):
            ax = axes[rowIdx * ncols + colIdx]
            subset = pivot_df_unindexed[pivot_df_unindexed['ShortName'] == kernel_name]
            pivot_plot_df = subset.pivot(index='ShortName', columns='Type', values=metric)
            pivot_plot_df.plot(kind='bar', color=color_list[:len(pivot_plot_df.columns)], ax=ax, legend=False, zorder=2)
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
        last_subplot = axes[rowIdx * ncols + ncols-1]
        handles, labels = last_subplot.get_legend_handles_labels()
        last_subplot.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Type')
    fig.tight_layout(pad=2.0)  # Adjust padding as needed
    plots_dir=f"Plots/SF_{SF}/Metrics"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    fig.savefig(f"{plots_dir}/Comparison_{extract_filename(file)}.png", dpi=300)
    return

    num_prefixes = len(pivot_df['Prefix'].unique())
    num_metrics = len(metrics)

    subplot_width = 5
    subplot_height = 5
    fig_width = num_metrics * subplot_width
    fig_height = num_prefixes * subplot_height
    fig, axs = plt.subplots(num_prefixes, num_metrics, figsize=(24, 24))

    if num_prefixes == 1 and num_metrics == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    prefixes = pivot_df['Prefix'].unique()
    print(pivot_df)
    for i, prefix in enumerate(prefixes):
        for j, metric in enumerate(metrics):
            ax = axs[i * num_metrics + j]
            data_to_plot = pivot_df[pivot_df['Prefix'] == prefix]
            if data_to_plot.empty:
                continue

            unique_types = data_to_plot['Type'].unique()
            num_types = len(unique_types)
            bar_width = 0.8 / num_types  
            index = np.arange(num_types)
            bars = {t: np.nan for t in unique_types}
            for t in unique_types:
                subset = data_to_plot[data_to_plot['Type'] == t]
                bars[t] = subset[metric].values[0]
            positions = index - (bar_width * (num_types - 1)) / 2
            for k, t in enumerate(unique_types):
                if not np.isnan(bars[t]):
                    ax.bar(positions + k * bar_width, bars[t], bar_width, label=t, color=plt.cm.tab10(k))
            
            ax.set_title(metric)
            ax.set_xticks(index)
            ax.set_xticklabels(unique_types)
            if j == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.0))

    for i, prefix in enumerate(prefixes):
        axs[i * num_metrics].set_ylabel(prefix, labelpad=20, rotation=0, horizontalalignment='right')
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    plots_dir=f"Plots/SF_{SF}/Metrics"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    fig.savefig(f"{plots_dir}/Comparison_{extract_filename(file)}.png", dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('SF', metavar='SF', type=int, help='scale factor')
    parser.add_argument('CSV_DIR', metavar='CSV_DIR', type=str, help='directory with measurements')

    args = parser.parse_args()
    for p in glob.glob(f"{args.CSV_DIR}/*.csv"):
        plot_metric(p, args.SF)
