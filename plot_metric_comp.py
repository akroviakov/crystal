import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
import re

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

    percent_cols = pivot_df.select_dtypes(include=['object']).columns[pivot_df.select_dtypes(include=['object']).apply(lambda x: x.str.contains('GB/s')).any()]
    pivot_df[percent_cols] = pivot_df[percent_cols].replace({'GB/s': ''}, regex=True).astype(float)
    pivot_df.rename(columns=lambda x: x + ' GB/s' if x in percent_cols else x, inplace=True)
    pattern = r'(\b\w+<[^>]*>)'
    pivot_df['Kernel'] = pivot_df['Kernel'].str.extract(pattern)

    # melted_df = melted_df.set_index(["Kernel", "Statistic", "Metric Name"]).unstack()
    pivot_df = pivot_df[pivot_df['Statistic'] == 'Avg']
    pivot_df.drop(columns=["Statistic", 'l2_utilization'], inplace=True)
    if 'local_hit_rate (%)' in pivot_df.columns:
        pivot_df.drop(columns=['local_hit_rate (%)'], inplace=True)
    if 'l2_utilization' in pivot_df.columns:
        pivot_df.drop(columns=['l2_utilization'], inplace=True)

    pivot_df.set_index('Kernel', inplace=True)
    bar_width = 0.35

    cols = 4
    rows = int(len(pivot_df.columns)//cols + (len(pivot_df.columns) % cols > 0))
    fig, axes_2d = plt.subplots(nrows=rows, ncols=cols, figsize=(24, 8), sharex=True)
    positions = range(len(pivot_df.index))
    axes = axes_2d.flatten()
    pivot_df = pivot_df.astype(float, errors='ignore') 

    kernel_pairs = []
    matched = set()
    for kernel in pivot_df.index:
        offset = -1      
        if "_Compiled" in kernel:
            offset = kernel.find('_Compiled')
        elif "Compiled" in kernel:
            offset = kernel.find('Compiled')
        if offset == -1:
            continue
        base_prefix = kernel[0:offset]
        base_kernel = [s for s in pivot_df.index if s.startswith(f'{base_prefix}<')][0]
        if base_kernel not in matched:
            pattern = r'\b{}(?:Compiled|_Compiled)?(?:\d*)<[^>]*(?:Parallelism=0|Parallelism=1)*[^>]*>'.format(re.escape(base_prefix))
            matches = re.findall(pattern, ', '.join(pivot_df.index))
            matched.add(base_kernel)
            kernel_pairs.append(matches)

    # Number of columns to plot
    columns_to_plot = pivot_df.columns
    # Create subplots
    num_variants = len(kernel_pairs[0])
    colors=['r','g','b']
    fig, axes = plt.subplots(len(kernel_pairs), len(columns_to_plot), figsize=(3 * len(columns_to_plot), 3* len(kernel_pairs)))
    bar_width = 0.1
    for i, tpl in enumerate(kernel_pairs):
        for j, column in enumerate(columns_to_plot):
            if len(kernel_pairs) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            for var_idx, variant in enumerate(tpl):
                line_label=variant
                if 'Parallelism=' in variant:
                    if 'Parallelism=1' in variant:
                        line_label = variant.replace('Parallelism=1', 'Parallelism=Omnisci') 
                    else:
                        line_label = variant.replace('Parallelism=0', 'Parallelism=Crystal')
                ax.bar((bar_width * var_idx), pivot_df.loc[variant, column], bar_width, label=line_label, zorder=2, color=colors[var_idx])
            ax.set_title(column)
            ax.grid(zorder=1)
            ax.set_xticks([])
            if j==0:
                ax.legend(loc='center right', bbox_to_anchor=(-0.3, 0.5))
           
    fig.suptitle(f"{extract_filename(file)}")
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