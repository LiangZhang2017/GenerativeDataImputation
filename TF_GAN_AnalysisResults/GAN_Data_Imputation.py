
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from TF_GAN_AnalysisResults.gan_data_imput_helper import visualize_spearman_correlation

if __name__ == '__main__':
    print("GAN")

    '''
    1.
    Lesson_Id 
    "CSAL": "lesson17", "lesson21", "lesson20", "lesson28"(too small dataset)
    "MATHia": 'worksheet_grapher_a1_patterns_2step_expr', 'scale_drawings_3', 'analyzing_models_2step_integers'
    
    2. 
    Imput_model: 
    "Standard_TC", "Standard_CPD", "BPTF", "AE", "VAE", "GAN", "InfoGAN", "AmbientGAN", "GAIN"
    '''

    # Define your lists and metric
    InputModel_list = ["Standard_TC", "Standard_CPD", "BPTF", "GAN", "InfoGAN", "AmbientGAN", "GAIN"]
    CourseType_list = ["CSAL", "CSAL", "ASSISMENTS", "ASSISMENTS", "MATHia", "MATHia"]
    Lesson_list = ["lesson21", "lesson20", "assismentsmath2008-2009", "2012-2013-data-with-predictions-4-final", "scale_drawings_3", "analyzing_models_2step_integers",
                   "analyzing_models_2step_integers"]

    metric = "RMSE"

    subplots_title_list=["ARC Lesson 1", "ARC Lesson 2", "ASSISTments Lesson 1", "ASSISTments Lesson 2", "MATHia Lesson 1", "MATHia Lesson 2"]

    # Initialize variables for plotting
    line_styles = ['-', '-', '-', '-','-','-']
    markers = ['s', '^', 'd', '*', 'P', 'v','o']
    # colors = plt.cm.tab10(np.linspace(0, 1, len(InputModel_list)))
    colors=['#045275', '#089099', '#7CCBA2', '#FCDE9C', '#F0746E', '#DC3977', '#7C1D6F']

    # Set global font settings
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 14

    # Create figure and axes for the subplots
    fig, axs = plt.subplots(2, 3, figsize=(28, 18), sharex=False, sharey=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.1,bottom=0.01, top=1)

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Define a list to store the data
    data_list = []

    # Loop over each CourseType and Lesson combination
    for i, (CourseType, Lesson, title) in enumerate(zip(CourseType_list, Lesson_list, subplots_title_list)):
        all_models_data = {}
        max_attempts = []  # List to store all max_attempts for current subplot

        for InputModel in InputModel_list:
            file_name = f"{InputModel}_version_0_{CourseType}_{Lesson}_Medium_evaluation_iter_100.txt"
            file_path = os.path.join(os.path.dirname(os.getcwd()), "results", InputModel, file_name)

            if os.path.exists(file_path):
                data = pd.read_csv(file_path, sep="\t")

                print("data is {}".format(data))
                max_attempt_local = max(data['Prune_slice_number'])

                data['Prune_slice_number']=max_attempt_local+1-data['Prune_slice_number']

                mean_std_metric = data.groupby(['Prune_slice_number'])[metric].agg(['mean', 'std']).reset_index()
                all_models_data[InputModel] = mean_std_metric
                max_attempts.extend(data['Prune_slice_number'].unique())
            else:
                print(f"File not found: {file_path}")

        # Plotting on the current subplot
        ax = axs[i]
        ax.set_title(title, fontsize=30)

        for idx, (model, data) in enumerate(all_models_data.items()):
            print("idx is {}".format(idx))
            ax.errorbar(data['Prune_slice_number'], data['mean'], yerr=data['std'],
                        label=model if i == 0 else "_nolegend_",
                        fmt=line_styles[idx % len(line_styles)], marker=markers[idx % len(markers)], markersize=16,
                        capsize=14, color=colors[idx],linewidth=2)

            # Calculate Spearman correlation coefficient
            spearman_corr, _ = spearmanr(data['Prune_slice_number'], data['mean'])

            # Add data to the list
            data_list.append({
                'Lesson': title,
                'Model': model,
                'Spearman Correlation': spearman_corr
            })

        # Determine unique x-ticks for the current subplot based on the accumulated max_attempts
        unique_ticks = np.unique(max_attempts)
        ax.set_xticks(unique_ticks)  # Set the unique x-ticks for the current subplot

        ax.tick_params(axis='x', labelsize=26)  # Set x-ticks size
        ax.tick_params(axis='y', labelsize=26)  # Set y-ticks size

        ax.set_xlabel('Max Attempt', fontsize=30)
        ax.set_ylabel(metric, fontsize=30)
        ax.grid(True, linestyle='--', color='#bbbbbb')
        ax.set_ylim([0., 1])
        ax.set_yticks(np.arange(0, 1.1, 0.1))

    visualize_spearman_correlation(data_list)

    # Shared legend
    handles, labels = axs[0].get_legend_handles_labels()

    new_labels = ["Tensor Factorization", "CPD", "BPTF", "GAN", "InfoGAN", "AmbientGAN", "GAIN"]
    # Replace the original labels with the new labels

    print("Agent labels are {}".format(labels))

    fig.legend(handles, new_labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=len(InputModel_list), fontsize=30)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the combined plot
    plot_file_path = os.path.join(os.getcwd(), "results", f"GAN_{metric}_combined_plot.png")
    plt.savefig(plot_file_path)
    plt.close()  # Close the plot to avoid display issues
    print(f"Combined plot saved to {plot_file_path}")