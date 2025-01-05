
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import os
import pandas as pd

def save_indices(indices, max_iter, mode, model_name, course, lesson_id,
                 learning_state, fold, model_iter):

    indices_file_path = os.path.join(os.getcwd(), "results", f"{model_name}", "training",
                                     f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}_indices.npy")
    np.save(indices_file_path, indices)


def save_ori_tensor(tensor,max_iter, mode, model_name, course, lesson_id,
                 learning_state, fold, model_iter):
    # Define file path
    data_file_path = os.path.join(os.getcwd(), "results", f"{model_name}", "training",
                                  f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}.npy")

    # Save the tensor as a .npy file
    np.save(data_file_path, tensor)


def save_metrics(train_perf, max_iter, mode, model_name, course, lesson_id, learning_state, fold, model_iter):

    # Define file path
    metrics_data_file = os.path.join(os.getcwd(), "results", f"{model_name}", "training",
                             f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}_metrics.npy")

    np.save(metrics_data_file,train_perf)