
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# def visualize_spearman_correlation(data_list):
#     print("data_List is {}".format(data_list))
#
#     # Group data by lesson
#     grouped_data = {}
#     for data_entry in data_list:
#         lesson = data_entry['Lesson']
#         model = data_entry['Model']
#         spearman_corr = data_entry['Spearman Correlation']
#         if lesson not in grouped_data:
#             grouped_data[lesson] = {}
#         grouped_data[lesson][model] = spearman_corr
#
#     # Set global font settings
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['font.size'] = 20
#
#     # Create subplots
#     fig, axs = plt.subplots(3, 2, figsize=(15, 15))
#     fig.subplots_adjust(hspace=0.5, wspace=0.3)
#
#     # Plot Spearman correlation for each lesson
#     for idx, (lesson, model_correlations) in enumerate(grouped_data.items()):
#         row = idx // 2
#         col = idx % 2
#         ax = axs[row, col]
#
#         # Get min and max Spearman correlation
#         min_corr = min(model_correlations.values())
#         max_corr = max(model_correlations.values())
#
#         # Adjust y-axis limits based on correlation sign
#         if min_corr < 0:
#             min_corr = -1
#         else:
#             min_corr = 0
#         if max_corr > 0:
#             max_corr = 1
#         else:
#             max_corr = 0
#
#         ax.bar(model_correlations.keys(), model_correlations.values(), color='skyblue')
#         ax.set_title(f"{lesson}", fontsize=25)
#         # ax.set_xlabel('Model', fontsize=25)
#         ax.set_ylabel('Spearman Correlation', fontsize=25)
#         ax.tick_params(axis='both', which='major', labelsize=20)
#         ax.tick_params(axis='x', rotation=45)
#         ax.grid(axis='y', linestyle='--', color='#cccccc')
#
#         # Set y-axis limits and ticks
#         ax.set_ylim(min_corr - 0.1, max_corr + 0.1)
#         ax.set_yticks([i * 0.2 for i in range(int(min_corr * 5), int((max_corr + 0.2) * 5))])
#
#     # Adjust layout
#     plt.tight_layout()
#
#     # Save the combined plot
#     plot_file_path = os.path.join(os.getcwd(), "results", f"Spearman_correlation_combined_plot.png")
#     plt.savefig(plot_file_path)
#     plt.close()  # Close the plot to avoid display issues
#
#     # Save data to Excel
#     df = pd.DataFrame(data_list)
#     excel_file_path = os.path.join(os.getcwd(), "results", "data_list.xlsx")
#     df.to_excel(excel_file_path, index=False)

def visualize_spearman_correlation(data_list):
    # Convert data list to DataFrame
    df = pd.DataFrame(data_list)

    matplotlib.rcParams['font.family'] = 'Times New Roman'

    # Specified order of lessons and models
    lesson_order = [
        'ARC Lesson 1', 'ARC Lesson 2', 'ASSISTments Lesson 1', 'ASSISTments Lesson 2',
        'MATHia Lesson 1', 'MATHia Lesson 2'
    ]
    model_order = [
        'Standard_TC', 'Standard_CPD', 'BPTF', 'GAN', 'InfoGAN', 'AmbientGAN', 'GAIN'
    ]

    new_labels = ["Tensor Factorization", "CPD", "BPTF", "GAN", "InfoGAN", "AmbientGAN", "GAIN"]

    # Pivot data to have models as rows, lessons as columns
    pivot_df = df.pivot(index='Model', columns='Lesson', values='Spearman Correlation').reindex(model_order)[lesson_order]

    # Lighten colors
    base_colors = ['#045275', '#089099', '#7CCBA2', '#FCDE9C', '#F0746E', '#DC3977', '#7C1D6F']
    colors = [plt.matplotlib.colors.to_rgba(c, alpha=0.7) for c in base_colors]  # Lighten by increasing transparency

    # Set global font settings
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # Create a bar plot
    ax = pivot_df.plot(kind='bar', figsize=(20, 10), color=colors, edgecolor='black', width=0.75)

    # Set titles and labels
    plt.ylabel('Spearman Correlation', fontsize=24)  # Set y label font size
    ax.set_xlabel('')
    plt.xticks(rotation=0)  # No rotation for x-axis labels
    plt.tick_params(axis='x', labelsize=21)  # Adjust x-tick label size
    plt.tick_params(axis='y', labelsize=16)
    ax.set_xticklabels(new_labels)  # Set new x-axis labels
    plt.grid(axis='y', linestyle='--', color='#cccccc')

    # Setting y-axis limits based on the data range
    min_corr = df['Spearman Correlation'].min()
    max_corr = df['Spearman Correlation'].max()
    plt.ylim([min_corr - 0.1, max_corr + 0.1])
    plt.yticks(np.arange(np.floor((min_corr) * 5) / 5, np.ceil((max_corr) * 5) / 5 + 0.1, 0.2))

    # Add legend closer to the plot area, adjust font size
    # legend = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=18, title_fontsize=20)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0.001, 0.99), fontsize=17.5, frameon=True)

    # Save the plot
    plot_file_path = os.path.join(os.getcwd(), "results", "combined_spearman_correlation_plot.png")
    plt.savefig(plot_file_path, bbox_inches='tight')
    plt.close()  # Close the plot to avoid display issues

    print(f"Combined plot saved to {plot_file_path}")

