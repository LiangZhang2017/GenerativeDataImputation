import tensorflow as tf
import numpy as np
import os

def gaussian_kernel(size, mean, std):
    """Creates a 2D Gaussian kernel with specified size, mean, and standard deviation."""
    d = np.arange(-size, size+1)
    x, y = np.meshgrid(d, d)
    kernel = np.exp(-((x-mean)**2 + (y-mean)**2) / (2 * std**2))
    kernel = kernel / np.sum(kernel)
    kernel_tf = tf.constant(kernel, dtype=tf.float32)
    return kernel_tf


def apply_measurement(image_tensor, kernel_size=5, sigma=1.0):
    """Applies Gaussian blur as a measurement process to the given image tensor."""
    channels = image_tensor.shape[-1]
    kernel = gaussian_kernel(kernel_size, 0.0, sigma)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, channels, 1])

    # Apply the kernel to each image
    blurred_image_tensor = tf.nn.depthwise_conv2d(image_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')

    return blurred_image_tensor


def adjust_sigma_decreasing(initial_sigma, reduction_rate, current_epoch, min_sigma=0.1):
    """
    Dynamically adjust sigma by reducing it at a specified rate per epoch, with a lower bound.

    Parameters:
    - initial_sigma: The starting value of sigma at epoch 0.
    - reduction_rate: The rate at which sigma is reduced each epoch.
    - current_epoch: The current epoch number.
    - min_sigma: The minimum value that sigma can take.

    Returns:
    - adjusted_sigma: The adjusted value of sigma for the current epoch.
    """
    adjusted_sigma = initial_sigma * (reduction_rate ** current_epoch)
    # Ensure sigma does not fall below a minimum threshold
    adjusted_sigma = max(adjusted_sigma, min_sigma)
    return adjusted_sigma

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