
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import os
import pandas as pd

from sklearn.decomposition import TruncatedSVD

def Partition(tensor,slice_size,mode,filter):
    """
    Partition a tensor into sub-tensors along the first dimension.

    :param tensor: The input tensor to be partitioned.
    :param slice_size: The size of the first dimension of each sub-tensor.
    :param mode: If "Average", partitions the tensor into sub-tensors of equal size.
    :param filter: If "normal", the rest dimension after partition will be filtered.
    :return: A list of sub-tensors.
    """

    sub_tensors = []
    sub_tensor = []
    sub_tensor_missing_indices =[]

    if mode == "Average" and filter == "normal":
        total_elements = tensor.shape[0]
        print("tensor.shape[0] is {}".format(tensor.shape[0]))
        num_slices = total_elements // slice_size
        for i in range(num_slices):
            start_idx = i * slice_size
            sub_tensor = tensor[start_idx:(start_idx + slice_size)]
            # sub_tensor, tensor_missing_indices = svd_impute(sub_tensor)
            sub_tensors.append(sub_tensor)
            # sub_tensor_missing_indices.append(tensor_missing_indices)

    return sub_tensors, tf.size(sub_tensor)


def svd_impute(tensor):

    #Identify Missing Data Indices
    tensor_missing_indices = np.where(np.isnan(tensor))

    # Replace np.nan with 0 for simplicity;
    tensor_filled = np.nan_to_num(tensor, nan=0)

    # Assuming your tensor is [num_slices, height, width], and you want to flatten the last two dimensions
    tensor_2d = tf.squeeze(tensor_filled, axis=0)

    # Apply SVD
    svd = TruncatedSVD(n_components=min(tensor_2d.shape) - 1, n_iter=7, random_state=42)
    tensor_reduced = svd.fit_transform(tensor_2d)
    tensor_reconstructed = svd.inverse_transform(tensor_reduced)

    # Reshape back to original shape if needed
    tensor_imputed = tensor_reconstructed.reshape(tensor_filled.shape)

    return tensor_imputed, tensor_missing_indices

# def generate_hints(mask, hint_rate):
#     # Hint mask is similar to the data mask but with some random values set to 0 (indicating imputed values)
#     hint_mask = np.random.rand(*mask.shape) < hint_rate
#     # Combine the hint mask with the original data mask to generate the hints
#     hints = mask * hint_mask
#     return hints

def generate_hints(mask, hint_rate):
    hint_mask = np.random.binomial(1, hint_rate, size=mask.shape)
    return mask * hint_mask

def normalization (data, parameters=None):

    train_set_tensor = tf.convert_to_tensor(data, dtype=tf.float32)  # Assuming train_set is your data

    # Flatten the tensor except for the first dimension to find global min and max
    flat_tensor = tf.reshape(train_set_tensor, (train_set_tensor.shape[0], -1))

    # Find the global min and max
    min_val = tf.reduce_min(flat_tensor, axis=1, keepdims=True)
    max_val = tf.reduce_max(flat_tensor, axis=1, keepdims=True)

    # Normalize the tensor
    normalized_tensor = (train_set_tensor - min_val) / (max_val - min_val)

    # Reshape min_val and max_val for broadcasting if necessary
    min_val = tf.reshape(min_val, (-1, 1, 1, 1))
    max_val = tf.reshape(max_val, (-1, 1, 1, 1))

    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

    print("normalized_tensor is {}".format(normalized_tensor))

    return normalized_tensor, norm_parameters


def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.

    Args:
      - low: low limit
      - high: high limit
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - uniform_random_matrix: generated uniform random matrix.
    '''
    return np.random.uniform(low, high, size=[rows, cols])


def save_indices(indices, max_iter, mode, model_name, course, lesson_id,
                 learning_state, prune_slice_number, fold, model_iter):

    # Define the directory path
    directory_path = os.path.join(os.getcwd(), "results", f"{model_name}", "training", "new")

    # Check if the directory exists, if not, create it
    os.makedirs(directory_path, exist_ok=True)

    indices_file_path = os.path.join(directory_path,
                                     f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_Slice{prune_slice_number}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}_indices.npy")

    np.save(indices_file_path, indices)


def save_ori_tensor(tensor,max_iter, mode, model_name, course, lesson_id,
                 learning_state, prune_slice_number, fold, model_iter):

    # Define the directory path
    directory_path = os.path.join(os.getcwd(), "results", f"{model_name}", "training", "new")

    # Ensure the directory exists, create it if not
    os.makedirs(directory_path, exist_ok=True)

    data_file_path = os.path.join(directory_path,
                                  f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_Slice{prune_slice_number}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}.npy")

    # Save the tensor as a .npy file
    np.save(data_file_path, tensor)


def save_metrics(train_perf, max_iter, mode, model_name, course, lesson_id, learning_state, prune_slice_number, fold, model_iter):
    # Define the directory path
    directory_path = os.path.join(os.getcwd(), "results", f"{model_name}", "training", "new")

    # Ensure the directory exists, create it if not
    os.makedirs(directory_path, exist_ok=True)

    # Define the file path for saving the metrics
    metrics_data_file = os.path.join(directory_path,
                                     f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_Slice{prune_slice_number}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}_metrics.npy")

    np.save(metrics_data_file,train_perf)
