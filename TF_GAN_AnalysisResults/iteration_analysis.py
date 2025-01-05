
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("iteration analysis")

    Model="GAIN"
    learning_stage="Medium"
    max_iter=100
    metric="RMSE" # MAE, RMSE, RSE, AUC, CROSS-Entropy

    # Define a dictionary to map metric names to their respective indices
    metric_indices = {
        "MAE": 0,
        "RMSE": 1,
        "RSE": 2,
        "AUC": 3,
        "CROSS-Entropy": 4
    }

    CourseType_list = ["CSAL", "CSAL", "ASSISMENTS", "ASSISMENTS", "MATHia", "MATHia"]
    Lesson_list = ["lesson21", "lesson20", "assismentsmath2008-2009", "2012-2013-data-with-predictions-4-final",
                   "scale_drawings_3", "analyzing_models_2step_integers",
                   "analyzing_models_2step_integers"]

    subplots_title_list = ["CSAL Lesson 1", "CSAL Lesson 2", "ASSISTments Lesson 1", "ASSISTments Lesson 2",
                           "MATHia Lesson 1", "MATHia Lesson 2"]

    matplotlib.rcParams['font.family'] = 'Times New Roman'

    for Iter in range(5):
        for fold in range(5):
            plt.figure(figsize=(10, 10))
            plt.xlabel('Iteration',fontsize=24)
            plt.ylabel(metric,fontsize=24)
            plt.grid(True)

            course_labels = []
            index = 0  # Index to track position in subplots_title_list

            for CourseType, Lesson in zip(CourseType_list, Lesson_list):
                filename = f"{Model}_{CourseType}_{Lesson}_{learning_stage}_train_MaxIter{max_iter}_Iter{Iter}_fold{fold}_metrics.npy"
                file_path = os.path.join(os.path.dirname(os.getcwd()), "results", Model, "training", filename)

                try:
                    data = np.load(file_path)
                    metric_value = data[:, metric_indices[metric]]

                    # Find the first index where value is less than 0.1
                    cutoff_indices = np.where(metric_value < 0.1)[0]
                    if cutoff_indices.size > 0:
                        cutoff_index = cutoff_indices[0]
                        metric_value = metric_value[:cutoff_index + 6]  # Keep next five values after cutoff

                    label = subplots_title_list[index]  # Use custom label from subplots_title_list

                    plt.plot(metric_value, 'o-', label=label)
                    index += 1

                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    continue

            plt.ylim(0, 1)
            plt.xlim(0, max_iter)
            plt.xticks(range(0, max_iter+10, 10), fontsize=19)  # Adjust x-axis ticks
            plt.yticks(np.arange(0, 1.1, 0.1), fontsize=19)  # Adjust y-axis ticks
            plt.legend(fontsize=23)
            plt.tight_layout()
            save_path = os.path.join(os.path.dirname(os.getcwd()), "TF_GAN_AnalysisResults", "results", f"GAIN_{metric}_{Model}_Iteration{Iter}_Fold{fold}_combined_plot.png")
            plt.savefig(save_path)