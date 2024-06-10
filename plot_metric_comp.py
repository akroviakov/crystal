import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob

def extract_filename(path):
    base_name = os.path.basename(path) 
    file_name = os.path.splitext(base_name)[0]
    return file_name

def plot_metric(file, SF):
    colors = ['b', 'r', 'g', 'y', 'c', 'm']
    df = pd.read_csv(file)
    df = df.drop(columns=['Device', 'Invocations', 'Metric Description'])
    melted_df = pd.melt(df, id_vars=["Kernel", "Metric Name"], value_vars=["Min", "Max", "Avg"], var_name="Statistic", value_name="Value")
    pivot_df = melted_df.pivot(index=["Kernel", "Statistic"], columns="Metric Name", values="Value").reset_index()
    percent_cols = pivot_df.select_dtypes(include=['object']).columns[pivot_df.select_dtypes(include=['object']).apply(lambda x: x.str.contains('%')).any()]
    pivot_df[percent_cols] = pivot_df[percent_cols].replace({'%': ''}, regex=True).astype(float)
    pivot_df.rename(columns=lambda x: x + ' (%)' if x in percent_cols else x, inplace=True)
    pivot_df['Kernel'] = pivot_df['Kernel'].str.extract(r'\s(\w+)<')

    # melted_df = melted_df.set_index(["Kernel", "Statistic", "Metric Name"]).unstack()
    pivot_df = pivot_df[pivot_df['Statistic'] == 'Avg']
    pivot_df = pivot_df.drop(columns=["Statistic", 'l2_utilization', "local_hit_rate (%)"])


    pivot_df.set_index('Kernel', inplace=True)
    bar_width = 0.35

    cols = 4
    rows = int(len(pivot_df.columns)//cols + (len(pivot_df.columns) % cols > 0))
    fig, axes_2d = plt.subplots(nrows=rows, ncols=cols, figsize=(24, 8), sharex=True)
    positions = range(len(pivot_df.index))
    axes = axes_2d.flatten()
    pivot_df = pivot_df.astype(float, errors='ignore') 

    kernel_pairs = []

    for kernel in pivot_df.index:
        if kernel.endswith('Compiled'):
            base_kernel = kernel.replace('Compiled', '')
            base_kernel1 = kernel.replace('_Compiled', '')

            if base_kernel in pivot_df.index:
                kernel_pairs.append((base_kernel, kernel))
            elif base_kernel1 in pivot_df.index:
                kernel_pairs.append((base_kernel1, kernel))

    # Number of columns to plot
    columns_to_plot = pivot_df.columns
    # Create subplots
    fig, axes = plt.subplots(len(kernel_pairs), len(columns_to_plot), figsize=(5 * len(kernel_pairs), len(columns_to_plot)))
    bar_width = 0.4

    for i, (base_kernel, compiled_kernel) in enumerate(kernel_pairs):
        for j, column in enumerate(columns_to_plot):
            if len(kernel_pairs) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            ax.bar(1 - bar_width / 2, pivot_df.loc[base_kernel, column], bar_width, label=base_kernel, zorder=2, color='blue')
            ax.bar(1 + bar_width / 2, pivot_df.loc[compiled_kernel, column], bar_width, label=compiled_kernel, zorder=2, color='orange')
            ax.set_title(column)
            if j==0:
                ax.legend(loc='center right', bbox_to_anchor=(-0.5, 0.5))
           

    fig.tight_layout()
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
        exit(-1)