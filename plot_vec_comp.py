import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def extract_filename(path):
    base_name = os.path.basename(path) 
    file_name = os.path.splitext(base_name)[0]
    return file_name

def plot_execution_times(file, SF):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['b', 'r', 'g', 'y', 'c', 'm']
    df = pd.read_csv(file)
    df['name'] = df['name'].str[:-4]
    df_mean = df.groupby(['name', 'type'])['executionTime'].mean().unstack()
    if 'comp_omnisci' in df_mean.columns:
        df_mean.drop(columns=['comp_omnisci'], inplace=True)
    df_normalized = df_mean.apply(lambda x: x / x.max() * 100, axis=1)
    bar_width = 0.3

    indices = np.arange(len(df_normalized))
    bar1 = ax.bar(indices - bar_width/2, df_normalized['vec'], bar_width, label='Vec',zorder=2)
    bar2 = ax.bar(indices + bar_width/2, df_normalized['comp'], bar_width, label='Comp',zorder=2)
    for i, (vec_bar, comp_bar) in enumerate(zip(bar1, bar2)):
        vec_value = df_mean['vec'].iloc[i]
        comp_value = df_mean['comp'].iloc[i]
        ax.text(vec_bar.get_x() + vec_bar.get_width() / 2, vec_bar.get_height(), f'{vec_value:.2f}ms', ha='center', va='bottom', fontsize=7)
        ax.text(comp_bar.get_x() + comp_bar.get_width() / 2, comp_bar.get_height(), f'{comp_value:.2f}ms', ha='center', va='bottom', fontsize=7)
        
    ax.set_xlabel('Query')
    ax.set_ylabel('Relative *kernel* time (%)')
    ax.set_title(f'SSB SF{SF}: Crystal vs "Compiled" approach')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xticks(indices)
    ax.set_xticklabels(df_normalized.index)
    ax.grid(zorder=1)
    fig.tight_layout()
    plots_dir=f"Plots/SF_{SF}/Model"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fig.savefig(f"{plots_dir}/Comparison_Model_SF_{SF}.png", dpi=300)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('SF', metavar='SF', type=int, help='scale factor')
    parser.add_argument('CSV', metavar='CSV', type=str, help='csv path')

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, args.CSV)

    plot_execution_times(csv_file_path, args.SF)