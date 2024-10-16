import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse

def extract_filename(path):
    base_name = os.path.basename(path) 
    file_name = os.path.splitext(base_name)[0]
    return file_name

def plot_execution_times(file, SF, reduced_plot=False):
    colors = ['#0072B2', '#D55E00', '#56B4E9', '#009E73', '#F0E442', '#E69F00', '#CC79A7', '#000000']
    # grayscale_colors = ['lightgray', 'dimgray', 'gray', 'darkgray', 'black']
    # colors = ['#e0e0e0', '#a0a0a0', '#707070', '#404040', '#101010']

    hatch_patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    color_rgba = {color: mcolors.to_rgba(color) for color in colors}
    color_to_hatch = dict(zip(color_rgba.values(), hatch_patterns))

    df = pd.read_csv(file)
    df['name'] = df['name'].str[:-4] 
    median_execution_times = df.groupby(['name', 'type'])['executionTime'].median().unstack()
    proportions = median_execution_times.div(median_execution_times.max(axis=1), axis=0) * 100
    if reduced_plot:
        proportions = proportions[["CompiledBatchToSM", "VectorOpt"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = proportions.plot(kind='bar', ax=ax, color=colors[:len(proportions.columns)], edgecolor='black', zorder=2, width=0.75)
    for container in bars.containers:
        for i, bar in enumerate(container): 
            name = proportions.index[i]
            column = container.get_label()
            original_value = median_execution_times.loc[name, column]
            bar.set_hatch(color_to_hatch[bar.get_facecolor()]) #hatch_patterns[i % len(hatch_patterns)])
            ax.annotate(f'{original_value:.2f}', 
                        (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 4), 
                        textcoords='offset points',
                        fontsize=6)  

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    # Customize the plot using the Axes object
    ax.set_xlabel('SSB Query')
    ax.set_ylabel('Relative Execution Time (%)')
    ax.set_title(f'Query implementations comparison (SF={SF})')
    ax.legend(title='Query implementation', loc='upper left', bbox_to_anchor=(0.65, -0.1))

    ax.grid(True,zorder=1)
    fig.tight_layout()

    plots_dir=f"Plots/SF_{SF}/Model"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    suffix = ""
    if reduced_plot:
        suffix = "_reduced"
    fig.savefig(f"{plots_dir}/Comparison_Model_SF_{SF}{suffix}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{plots_dir}/Comparison_Model_SF_{SF}{suffix}.pdf", bbox_inches='tight')   


# def plot_execution_times(file, SF):
#     fig, ax = plt.subplots(figsize=(8, 4))
#     colors = ['b', 'r', 'g', 'y', 'c', 'm']
#     df = pd.read_csv(file)
#     df['name'] = df['name'].str[:-4]
#     print(df)
#     median_execution_times = df.groupby(['name', 'type'])['executionTime'].median().unstack()
#     median_execution_times.plot(kind='bar', figsize=(10, 6))
#     plt.show()

#     print(median_execution_times)


#     df_mean = df.groupby(['name', 'type'])['executionTime'].mean().unstack()
#     if 'comp_omnisci' in df_mean.columns:
#         df_mean.drop(columns=['comp_omnisci'], inplace=True)
#     df_normalized = df_mean.apply(lambda x: x / x.max() * 100, axis=1)
#     bar_width = 0.3

#     indices = np.arange(len(df_normalized))
#     bar1 = ax.bar(indices - bar_width/2, df_normalized['vec'], bar_width, label='Vec',zorder=2)
#     bar2 = ax.bar(indices + bar_width/2, df_normalized['comp'], bar_width, label='Comp',zorder=2)
#     for i, (vec_bar, comp_bar) in enumerate(zip(bar1, bar2)):
#         vec_value = df_mean['vec'].iloc[i]
#         comp_value = df_mean['comp'].iloc[i]
#         ax.text(vec_bar.get_x() + vec_bar.get_width() / 2, vec_bar.get_height(), f'{vec_value:.2f}ms', ha='center', va='bottom', fontsize=7)
#         ax.text(comp_bar.get_x() + comp_bar.get_width() / 2, comp_bar.get_height(), f'{comp_value:.2f}ms', ha='center', va='bottom', fontsize=7)
        
#     ax.set_xlabel('Query')
#     ax.set_ylabel('Relative time (%)')
#     ax.set_title(f'SSB SF{SF}: Crystal vs "Compiled" approach')
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax.set_xticks(indices)
#     ax.set_xticklabels(df_normalized.index)
#     ax.grid(zorder=1)
#     fig.tight_layout()
#     plots_dir=f"Plots/SF_{SF}/Model"
#     if not os.path.exists(plots_dir):
#         os.makedirs(plots_dir)

#     fig.savefig(f"{plots_dir}/Comparison_Model_SF_{SF}.png", dpi=300)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('CSV', metavar='CSV', type=str, help='csv path')
    parser.add_argument('SF', metavar='SF', type=int, help='scale factor')

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, args.CSV)

    plot_execution_times(csv_file_path, args.SF)
    plot_execution_times(csv_file_path, args.SF, True)