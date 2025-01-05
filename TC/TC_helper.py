import os

import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
from itertools import combinations


def perform_k_fold_cross_validation_numpy_tensor(numpy_T, k=5, shuffle=True, random_state=22):
    np.random.seed(random_state)

    # Shuffle the dataset if required
    if shuffle:
        shuffled_indices = np.random.permutation(len(numpy_T))
        numpy_T_shuffled = numpy_T[shuffled_indices]
    else:
        numpy_T_shuffled = numpy_T

    unique_values = np.unique(numpy_T[:, 0])
    kf = KFold(n_splits=k, shuffle=False)  # Shuffle is False because dataset was potentially already shuffled

    train_test_splits = []
    all_included = True

    train_indices_list = []
    test_indices_list = []

    for train_indices, test_indices in kf.split(numpy_T_shuffled):
        train_fold = numpy_T_shuffled[train_indices]
        test_fold = numpy_T_shuffled[test_indices]

        # Check if all unique values are present in the training fold
        train_unique_values = np.unique(train_fold[:, 0])
        all_included &= np.all(np.isin(unique_values, train_unique_values))

        train_test_splits.append((train_fold, test_fold))

        train_indices_list.append(train_indices)
        test_indices_list.append(test_indices)

    # # Print sizes and unique values for the first fold as a demonstration
    # print("Train and test fold sizes for the first fold:")
    # print("Train fold size:", len(train_test_splits[0][0]))
    # print("Test fold size:", len(train_test_splits[0][1]))

    # print("Train fold unique is {}".format(np.unique(train_test_splits[0][0][:, 0])))
    # print("Test fold unique is {}".format(np.unique(train_test_splits[0][1][:, 0])))

    # Returns the list of train-test splits and the status if all unique values are included in each training fold
    # print("train_test_splits are {}".format(train_test_splits))

    # output train_data and test_data
    return train_test_splits

    # Assuming numpy_T is defined and structured as per your dataset
    # Run the defined function with your numpy_T data
    # train_test_splits, all_included = perform_k_fold_cross_validation_numpy_tensor(numpy_T, k=5, shuffle=True, random_state=20)

    # Since numpy_T is not currently defined in this new code execution context, this call is commented out.
    # You would uncomment and run this line with your actual numpy_T dataset.

def perform_k_fold_cross_validation(tensor, k=5, shuffle=True, random_state=22):
    """
       Randomly sample non-missing entries from the tensor to create a test set, preserving existing missing values.

       Args:
       - tensor (np.ndarray): The original tensor, possibly with missing values (np.nan).
       - test_size (float): The proportion of the non-missing tensor to be used as the test set.

       Returns:
       - train_tensor (np.ndarray): The tensor with test entries masked, preserving original missing values.
       - test_tensor (np.ndarray): A tensor of the same shape as the original, with non-test entries masked.
       - test_indices (list of tuples): The indices of entries sampled for the test set.
    """

    np.random.seed(42)  # For reproducibility
    test_size=1/k

    # Convert TensorFlow tensor to NumPy array for item assignment
    tensor_np = tensor.numpy()

    # Identify non-missing entries
    non_missing_indices = np.where(~np.isnan(tensor_np))

    # Calculate the number of entries to sample for the test set, based only on non-missing entries
    total_non_missing_entries = len(non_missing_indices[0])
    test_entries = int(total_non_missing_entries * test_size)

    # Randomly select indices from the non-missing entries
    selected_indices = np.random.choice(range(total_non_missing_entries), size=test_entries, replace=False)
    test_indices = tuple(non_missing_indices[dim][selected_indices] for dim in range(len(tensor_np.shape)))

    # Create the test tensor with all entries masked initially
    test_tensor_np = np.full(tensor_np.shape, np.nan)

    # Assign the selected test entries to the test tensor and mask them in the original tensor
    for idx in zip(*test_indices):
        test_tensor_np[idx] = tensor_np[idx]
        tensor_np[idx] = np.nan  # Masking the selected test entries in the original tensor

    # Convert back to TensorFlow tensors
    train_tensor = tf.convert_to_tensor(tensor_np)
    test_tensor = tf.convert_to_tensor(test_tensor_np)

    print("train_tensor is {}".format(train_tensor.shape))
    print("test_tensor is {}".format(test_tensor.shape))

    non_nan_train = np.sum(~np.isnan(train_tensor))
    non_nan_test = np.sum(~np.isnan(test_tensor))

    print("All number of non_nan_train is {}".format(non_nan_train))
    print("All number of non_nan_test is {}".format(non_nan_test))

    return train_tensor, test_tensor, list(zip(*test_indices))

def prepare_k_fold_indices(tensor, k=5, random_state=22):
    """
    Prepare indices for k-fold cross-validation, considering only non-missing (non-nan) entries.

    Args:
    - tensor (tf.Tensor): Input tensor with possible missing values (np.nan).
    - k (int): Number of folds.
    - random_state (int): Seed for the random number generator.

    Returns:
    - folds (list of tuples): A list where each element is a tuple containing train and test indices for a fold.
    """
    np.random.seed(random_state)  # For reproducibility

    # Convert TensorFlow tensor to NumPy array to work with indices
    tensor_np = tensor.numpy()

    # Identify non-missing entries
    non_missing_indices = np.array(np.where(~np.isnan(tensor_np))).T

    # Shuffle indices of non-missing entries
    np.random.shuffle(non_missing_indices)

    # Split indices into k approximately equal parts
    folds_indices = np.array_split(non_missing_indices, k)

    folds = []
    for i in range(k):
        test_indices = folds_indices[i]

        # Use the rest of the folds as training data
        train_indices = np.vstack([folds_indices[j] for j in range(k) if j != i])

        folds.append((train_indices, test_indices))

    return folds


def perform_k_fold_cross_validation_tf(tensor, k=5):
    """
    A generator that performs k-fold cross-validation, yielding train and test tensors for each fold.

    Args:
    - tensor (tf.Tensor): Input tensor with possible missing values (np.nan).
    - k (int): Number of folds.

    Yields:
    - train_tensor (tf.Tensor): Training tensor for the fold.
    - test_tensor (tf.Tensor): Testing tensor for the fold.
    - test_indices (np.ndarray): Indices of the test set in the original tensor.
    """
    folds = prepare_k_fold_indices(tensor, k)

    for train_indices, test_indices in folds:
        # Initialize entire tensors as NaN
        train_tensor_np = np.full(tensor.shape, np.nan, dtype=tensor.dtype.as_numpy_dtype)
        test_tensor_np = np.full(tensor.shape, np.nan, dtype=tensor.dtype.as_numpy_dtype)

        # Fill in the relevant data for each tensor from the original tensor
        train_tensor_np[tuple(train_indices.T)] = tensor.numpy()[tuple(train_indices.T)]
        test_tensor_np[tuple(test_indices.T)] = tensor.numpy()[tuple(test_indices.T)]

        # Convert back to TensorFlow tensors
        train_tensor = tf.convert_to_tensor(train_tensor_np, dtype=tensor.dtype)
        test_tensor = tf.convert_to_tensor(test_tensor_np, dtype=tensor.dtype)

        # Calculate and print the number of non-NaN entries
        non_nan_train = np.sum(~np.isnan(train_tensor_np))
        non_nan_test = np.sum(~np.isnan(test_tensor_np))

        print("Number of non-NaN values in train_tensor: {}".format(non_nan_train))
        print("Number of non-NaN values in test_tensor: {}".format(non_nan_test))

        # Yield the tensors for the current fold
        yield train_tensor, test_tensor, train_indices, test_indices


def tensor_to_numpy(tensor):
    indices = np.indices(tensor.shape)
    four_d_array = np.stack((indices[0], indices[1], indices[2], tensor), axis=-1)
    four_d_array = four_d_array.reshape(-1, 4)

    return four_d_array

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save_indices(indices, max_iter, mode, model_name, course, lesson_id, learning_state, fold, model_iter):
    indices_file_path = os.path.join(os.getcwd(), "results", f"{model_name}", "training",
                             f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}_indices.npy")

    np.save(indices_file_path, indices)

def save_tensor(tensor, max_iter, mode, model_name, course, lesson_id, learning_state, prune_slice_number, fold, model_iter):
    # Convert tensor to NumPy array if it's not already
    tensor_np = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor

    # Define the directory path
    directory_path = os.path.join(os.getcwd(), "results", f"{model_name}", "training", "new")

    # Ensure the directory exists, create it if not
    os.makedirs(directory_path, exist_ok=True)

    # Define the file path for saving the data
    data_file_path = os.path.join(directory_path,
                                  f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_Slice{prune_slice_number}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}.npy")

    # Save the tensor as a .npy file
    np.save(data_file_path, tensor_np)
    print(f"Tensor saved successfully at: {data_file_path}")
    print("file_path is {}".format(data_file_path))


def save_metrics(train_perf, max_iter, mode, model_name, course, lesson_id, learning_state, fold, model_iter):

    # Define file path
    metrics_data_file = os.path.join(os.getcwd(), "results", f"{model_name}", "training",
                             f"{model_name}_{course}_{lesson_id}_{learning_state}_{mode}_MaxIter{max_iter}_Iter{model_iter}_fold{fold}_metrics.npy")

    np.save(metrics_data_file,train_perf)