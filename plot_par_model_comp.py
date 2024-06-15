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
    df = pd.read_csv(file)
    avg_execution_time = df.groupby(['type', 'batch_size'])['executionTime'].mean().reset_index()
    pivot_df = avg_execution_time.pivot(index='batch_size', columns='type', values='executionTime')

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot_df.plot(ax=ax,
        style=['-', '--', '-.'],
        marker='X',
        logx=True,
        logy=True)

    ax.set_xlabel('Batch Size (rows)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title(f'SSB SF{SF}: Crystal vs Omnisci parallelism')
    ax.legend(title='Parallelism model',loc='center left', bbox_to_anchor=(1, 0.5))

    ax.grid(True)

    fig.tight_layout()
    plots_dir=f"Plots/SF_{SF}/Par_Model"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fig.savefig(f"{plots_dir}/Comparison_Par_Model_SF_{SF}.png", dpi=300)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('SF', metavar='SF', type=int, help='scale factor')
    parser.add_argument('CSV', metavar='CSV', type=str, help='csv path')

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, args.CSV)

    plot_execution_time(csv_file_path, args.SF)