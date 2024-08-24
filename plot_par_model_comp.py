import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def extract_filename(path):
    base_name = os.path.basename(path) 
    file_name = os.path.splitext(base_name)[0]
    return file_name

def plot_execution_time(file, SF):
    plots_dir = f"Plots/SF_{SF}/Par_Model"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    df = pd.read_csv(file)
    df = df[~df['type'].isin(["Vector", "VectorOpt"])]
    markers = ["v", "X", "1", "s"]
    unique_names = df['name'].unique()
    for name in unique_names:
        name_df = df[df['name'] == name]
        avg_execution_time = name_df.groupby(['type', 'batch_size'])['executionTime'].min().reset_index()
        pivot_df = avg_execution_time.pivot(index='batch_size', columns='type', values='executionTime')
        fig, ax = plt.subplots(figsize=(3.5, 3))
        ax = pivot_df.plot(ax=ax,
                      style=['-', '--', '-.', ':'],
                      logx=True,
                      logy=True)
        for i, line in enumerate(ax.get_lines()):
            line.set_marker(markers[i])
        ax.set_xlabel('Batch Size (rows)')
        ax.set_ylabel('Execution Time (ms)')
        ax.legend(title='Parallelism model', loc='upper center')
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f"{plots_dir}/Comparison_Par_Model_{name[:-4]}_SF_{SF}.png", dpi=300)
        fig.savefig(f"{plots_dir}/Comparison_Par_Model_{name[:-4]}_SF_{SF}.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('SF', metavar='SF', type=int, help='scale factor')
    parser.add_argument('CSV', metavar='CSV', type=str, help='csv path')

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, args.CSV)

    plot_execution_time(csv_file_path, args.SF)