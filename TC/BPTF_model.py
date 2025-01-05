import time

import tensorflow as tf
import os
import numpy as np

from AE.AE_helper import save_summary
from TC.TC_helper import tensor_to_numpy, sigmoid, perform_k_fold_cross_validation_tf, save_indices
from scipy.special import expit

from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from math import sqrt

from TC.TC_helper import save_tensor, save_metrics

'''
Reference:
https://github.com/xinychen/transdim 
https://github.com/xinychen/transdim/blob/master/imputer/BPTF.ipynb
'''

class BPTF:
    def __init__(self, tensor, parameters):

        np.random.seed(22)
        self.model_name = parameters['model_str']
        self.course = parameters['course']
        self.lesson_id = parameters['lesson_id']
        self.learning_state = parameters['learning_stage']

        self.T = None
        self.train_tensor = None
        self.test_tensor = None
        self.tensor = tensor
        self.parameters = parameters
        self.num_learner = None
        self.num_question = None
        self.num_attempt = None
        self.num_features = parameters["features_dim"]

        self.use_bias_t = True
        self.use_global_bias = True
        self.binarized_question = True

        self.U_bias = None  # Learner biases
        self.V_bias = None  # Question biases
        self.X_bias = None  # Attempt biases
        self.global_bias = 0.0  # Global bias

        self.max_iter = parameters['max_iter']

        self.gibbs_iteration = 200

        self.Version = 0

        self.lambda_u = parameters.get("lambda_u", 1e-3)  # Regularization for U
        self.lambda_v = parameters.get("lambda_v", 1e-3)  # Regularization for V
        self.lambda_x = parameters.get("lambda_x", 1e-3)  # Regularization for X

        self.weight_u_bias = parameters.get("lambda_bias", 1.0)  # Weight for U_bias
        self.weight_v_bias = parameters.get("lambda_bias", 1.0)  # Weight for V_bias
        self.weight_x_bias = parameters.get("lambda_bias", 1.0)  # Weight for X_bias
        self.weight_global_bias = parameters.get("lambda_bias", 1.0)  # Weight for global_bias

    def mvnrnd_pre(self, mu, Lambda):
        src = normrnd(size=(mu.shape[0],))
        return solve_ut(cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
                        src, lower=False, check_finite=False, overwrite_b=True) + mu

    def cov_mat(self, mat, mat_bar):
        mat = mat - mat_bar
        return mat.T @ mat

    def ten2mat(self, tensor, mode):
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

    def sample_factor_u(self, tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
        """Sampling M-by-R factor matrix U and its hyperparameters (mu_u, Lambda_u)."""

        dim1, rank = U.shape
        U_bar = np.mean(U, axis=0)
        temp = dim1 / (dim1 + beta0)
        var_mu_hyper = temp * U_bar
        var_U_hyper = inv(np.eye(rank) + self.cov_mat(U, U_bar) + temp * beta0 * np.outer(U_bar, U_bar))

        # Add regularization for avoiding over-fitting
        reg_term = self.lambda_u * np.eye(rank)  # Regularization term
        var_U_hyper = inv(inv(var_U_hyper) + reg_term)  # Adjusting for regularization

        var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_U_hyper)
        var_mu_hyper = self.mvnrnd_pre(var_mu_hyper, (dim1 + beta0) * var_Lambda_hyper)

        var1 = kr_prod(X, V).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ self.ten2mat(tau_ind, 0).T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
        var4 = var1 @ self.ten2mat(tau_sparse_tensor, 0).T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
        for i in range(dim1):
            U[i, :] = self.mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

        return U

    def sample_factor_v(self, tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
        """Sampling N-by-R factor matrix V and its hyperparameters (mu_v, Lambda_v)."""

        dim2, rank = V.shape
        V_bar = np.mean(V, axis=0)
        temp = dim2 / (dim2 + beta0)
        var_mu_hyper = temp * V_bar
        var_V_hyper = inv(np.eye(rank) + self.cov_mat(V, V_bar) + temp * beta0 * np.outer(V_bar, V_bar))

        # Add regularization for avoiding over-fitting
        reg_term = self.lambda_v * np.eye(rank)  # Regularization term
        var_V_hyper = inv(inv(var_V_hyper) + reg_term)  # Adjusting for regularization

        var_Lambda_hyper = wishart.rvs(df=dim2 + rank, scale=var_V_hyper)
        var_mu_hyper = self.mvnrnd_pre(var_mu_hyper, (dim2 + beta0) * var_Lambda_hyper)

        var1 = kr_prod(X, U).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ self.ten2mat(tau_ind, 1).T).reshape([rank, rank, dim2]) + var_Lambda_hyper[:, :, None]
        var4 = var1 @ self.ten2mat(tau_sparse_tensor, 1).T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
        for j in range(dim2):
            V[j, :] = self.mvnrnd_pre(solve(var3[:, :, j], var4[:, j]), var3[:, :, j])

        return V

    def sample_factor_x(self, tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
        """Sampling T-by-R factor matrix X and its hyperparameters."""

        dim3, rank = X.shape

        # Adjustments for Gaussian-Wishart prior
        var_mu_hyper = X[0, :] / (beta0 + 1)
        if dim3 > 1:
            dx = X[1:, :] - X[:-1, :]
        else:
            dx = np.zeros((dim3, rank))  # No difference when there's only one timestep

        var_V_hyper = inv(np.eye(rank) + dx.T @ dx + beta0 * np.outer(X[0, :], X[0, :]) / (beta0 + 1))

        # Add regularization for avoiding over-fitting
        reg_term = self.lambda_x * np.eye(rank)  # Regularization term
        var_V_hyper = inv(inv(var_V_hyper) + reg_term)  # Adjusting for regularization

        var_Lambda_hyper = wishart.rvs(df=dim3 + rank, scale=var_V_hyper)
        var_mu_hyper = self.mvnrnd_pre(var_mu_hyper, (beta0 + 1) * var_Lambda_hyper)

        var1 = kr_prod(V, U).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ self.ten2mat(tau_ind, 2).T).reshape([rank, rank, dim3])
        var4 = var1 @ self.ten2mat(tau_sparse_tensor, 2).T

        for t in range(dim3):
            if dim3 == 1:
                # Special handling for a single time step
                temp1 = var4[:, t] + var_Lambda_hyper @ var_mu_hyper
                temp2 = var3[:, :, t] + var_Lambda_hyper
                X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
            elif t == 0:
                X[t, :] = self.mvnrnd_pre((X[t + 1, :] + var_mu_hyper) / 2, var3[:, :, t] + 2 * var_Lambda_hyper)
            elif t == dim3 - 1:
                temp1 = var4[:, t] + var_Lambda_hyper @ X[t - 1, :]
                temp2 = var3[:, :, t] + var_Lambda_hyper
                X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
            else:
                temp1 = var4[:, t] + var_Lambda_hyper @ (X[t - 1, :] + X[t + 1, :])
                temp2 = var3[:, :, t] + 2 * var_Lambda_hyper
                X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
        return X

    def sample_precision_tau(self, sparse_tensor, tensor_hat, ind):
        var_alpha = 1e-6 + 0.5 * np.sum(ind)
        var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)
        return np.random.gamma(var_alpha, 1 / var_beta)

    def compute_rmse(self, var, var_hat):
        return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

    def compute_binary_cross_entropy(self, var, var_hat):
        epsilon = 1e-15
        var_hat = np.clip(var_hat, epsilon, 1 - epsilon)  # To avoid log(0)
        return -np.mean(var * np.log(var_hat) + (1 - var) * np.log(1 - var_hat))

    def compute_mae(self, var, var_hat):
        """Compute Mean Absolute Error (MAE) between var and var_hat."""
        return np.mean(np.abs(var - var_hat))

    def compute_auc(self, var, var_hat):
        """Compute the AUC score between true labels and predicted scores."""
        return roc_auc_score(var, var_hat)

    def compute_rse(self, var, var_hat):
        # Compute the numerator as the total squared error between predicted and actual values
        numerator = np.sum((var - var_hat) ** 2)

        # Compute the denominator as the total squared error between actual values and the mean of actual values
        mean_var = np.mean(var)
        denominator = np.sum((var - mean_var) ** 2)

        # Calculate RSE
        rse = numerator / denominator

        return rse

    def update_biases(self, sparse_tensor, tensor_hat, ind):
        residuals = (sparse_tensor - tensor_hat) * ind  # Calculate residuals where data is observed
        # Update biases
        self.U_bias = np.mean(residuals, axis=(1, 2))  # Assuming the mean residual per learner as the bias
        self.V_bias = np.mean(residuals, axis=(0, 2))  # Assuming the mean residual per question as the bias
        self.X_bias = np.mean(residuals, axis=(0, 1))  # Assuming the mean residual per attempt as the bias
        self.global_bias = np.mean(residuals)  # Assuming the overall mean residual as the global bias

    def reconstruct(self, U, V, X):
        """Reconstruct the tensor with biases included, applying weights to biases."""
        bias_matrix = (self.U_bias[:, np.newaxis, np.newaxis] * self.weight_u_bias +
                       self.V_bias[np.newaxis, :, np.newaxis] * self.weight_v_bias +
                       self.X_bias[np.newaxis, np.newaxis, :] * self.weight_x_bias)
        tensor_hat = np.einsum('is, js, ts -> ijt', U, V, X) + bias_matrix + self.global_bias * self.weight_global_bias
        tensor_hat = expit(tensor_hat)  # Applying sigmoid function if necessary
        return tensor_hat

    def train(self, train_tensor, burn_iter, gibbs_iter, fold, model_iter):
        """Bayesian Probabilistic Tensor Factorization, BPTF."""

        train_perf = []

        dim = np.array(train_tensor.shape)
        U = self.U
        V = self.V
        X = self.X

        # print("U shape is {}".format(U.shape))
        # print("V shape is {}".format(V.shape))
        # print("X shape is {}".format(X.shape))

        show_iter = 1
        tau = 1

        tensor_hat_plus = np.zeros(dim)
        ind = ~np.isnan(train_tensor)
        pos_test = np.where(~np.isnan(train_tensor))

        temp_hat = np.zeros(len(pos_test[0]))

        train_tensor[np.isnan(train_tensor)] = 0

        for it in range(burn_iter + gibbs_iter):
            tau_ind = tau * ind
            tau_sparse_tensor = tau * train_tensor

            self.U = self.sample_factor_u(tau_sparse_tensor, tau_ind, self.U, self.V, self.X)
            self.V = self.sample_factor_v(tau_sparse_tensor, tau_ind, self.U, self.V, self.X)
            self.X = self.sample_factor_x(tau_sparse_tensor, tau_ind, self.U, self.V, self.X)

            # tensor_hat = expit(np.einsum('is, js, ts -> ijt', self.U, self.V, self.X))
            # Assuming reconstruct() method properly incorporates biases
            tensor_hat = self.reconstruct(self.U, self.V, self.X)

            if it + 1 > burn_iter:
                # Fine-tune biases after the burn-in period
                self.update_biases(train_tensor, tensor_hat, ind)

            temp_hat += tensor_hat[pos_test]
            tau = self.sample_precision_tau(train_tensor, tensor_hat, ind)

            if it + 1 > burn_iter:
                tensor_hat_plus += tensor_hat
            if (it + 1) % show_iter == 0 and it < burn_iter:
                temp_hat = temp_hat / show_iter
                print('Iter: {}'.format(it + 1))

                MAE = self.compute_mae(train_tensor[pos_test], temp_hat)
                RMSE=self.compute_rmse(train_tensor[pos_test], temp_hat)
                RSE=self.compute_rse(train_tensor[pos_test], temp_hat)
                AUC=self.compute_auc(train_tensor[pos_test],temp_hat)
                cross_entropy=self.compute_binary_cross_entropy(train_tensor[pos_test],temp_hat)

                print('MAE: {}'.format(MAE),'RMSE: {}'.format(RMSE),'RSE: {}'.format(RSE),'AUC: {}'.format(AUC))

                train_perf.append([MAE,RMSE,RSE,AUC,cross_entropy])

                temp_hat = np.zeros(len(pos_test[0]))

        tensor_hat = tensor_hat_plus / gibbs_iter

        self.T=tensor_hat

        mode = "train"
        save_tensor(tensor_hat, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state, fold,
                    model_iter)

        save_metrics(train_perf, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state,
                     fold,
                     model_iter)

    def test(self, test_tensor, test_indices, fold, model_iter):
        """Training the BPTF model on the test dataset."""

        perf_dict = []
        curr_pred_list = []
        curr_obs_list = []

        test_indices_tf = tf.constant(test_indices)

        # Use tf.gather_nd to extract values at these indices
        test_real_values = tf.gather_nd(self.test_tensor, test_indices_tf)
        # Convert to numpy array if needed
        test_real_values_np = test_real_values.numpy()

        pred_test_values = tf.gather_nd(self.T, test_indices_tf)
        pred_test_values_np = pred_test_values.numpy()

        # print("test_real_values_np are {}".format(len(test_real_values_np)))
        # print("pred_test_values_np are {}".format(len(pred_test_values_np)))

        # Filter out pairs where train_obs is NaN
        curr_obs_list, curr_pred_list = zip(
            *[(obs, pred) for obs, pred in zip(test_real_values_np, pred_test_values_np) if not np.isnan(obs)])

        test_mae = mean_absolute_error(curr_obs_list, curr_pred_list)
        test_rmse = sqrt(mean_squared_error(curr_obs_list, curr_pred_list))

        mse = mean_squared_error(curr_obs_list, curr_pred_list)
        test_rse = mse / np.mean(np.square(curr_obs_list))

        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        cross_entropy = bce_loss(curr_obs_list, curr_pred_list)

        try:
            test_auc_score = roc_auc_score(curr_obs_list, curr_pred_list)
            test_auc_score = tf.reduce_mean(test_auc_score)
        except ValueError as e:
            if 'Only one class present' in str(e):
                test_auc_score = np.nan
            else:
                raise  # re-raise the exception if it's not the "Only one class" error

        perf_dict.append((test_mae, test_rmse, test_rse, float(test_auc_score), float(cross_entropy)))

        print(test_mae, test_rmse, test_rse, float(test_auc_score), float(cross_entropy))

        mode = "test"
        save_metrics(perf_dict, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state,
                     fold, model_iter)

        return perf_dict

    def RunModel(self, model_iter):
        print("Run of BPTF")

        start_time = time.time()
        cv_results = []

        fold_summary = {}

        for fold, (train_tensor, test_tensor, train_indices, test_indices) in enumerate(
                perform_k_fold_cross_validation_tf(self.tensor, k=5)):
            print(f"Training on fold {fold + 1}")

            # # Train and test your model here using train_tensor and test_tensor
            # print("train_tensor.shape is : {}".format(train_tensor.shape))
            # print("test_tensor.shape is: {}".format(test_tensor.shape))

            tf.random.set_seed(22)
            np.random.seed(22)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            self.num_learner = train_tensor.shape[0]
            self.num_question = train_tensor.shape[1]
            self.num_attempt = train_tensor.shape[2]

            self.U_bias = np.zeros(self.num_learner)
            self.V_bias = np.zeros(self.num_question)
            self.X_bias = np.zeros(self.num_attempt)

            self.U = 0.1 * np.random.randn(self.num_learner, self.num_features)
            self.V = 0.1 * np.random.randn(self.num_question, self.num_features)
            self.X = 0.1 * np.random.randn(self.num_attempt, self.num_features)

            self.train_tensor_np = train_tensor.numpy()
            self.test_tensor_np = test_tensor.numpy()

            self.test_tensor = test_tensor
            self.train_tensor = train_tensor

            train_mode = "origin_train"
            save_indices(train_indices, self.max_iter, train_mode, self.model_name, self.course, self.lesson_id,
                         self.learning_state, fold, model_iter)
            save_tensor(self.train_tensor_np, self.max_iter, train_mode, self.model_name, self.course, self.lesson_id,
                        self.learning_state, fold, model_iter)

            test_mode = "origin_test"
            save_indices(test_indices, self.max_iter, test_mode, self.model_name, self.course, self.lesson_id,
                         self.learning_state, fold, model_iter)
            save_tensor(self.train_tensor_np, self.max_iter, test_mode, self.model_name, self.course, self.lesson_id,
                        self.learning_state, fold, model_iter)

            self.train(self.train_tensor_np, self.max_iter, self.gibbs_iteration, fold, model_iter)

            test_results = self.test(self.test_tensor_np, test_indices, fold, model_iter)
            cv_results.append((fold, test_results[0][0], test_results[0][1], test_results[0][2], test_results[0][3],
                               test_results[0][4], self.max_iter, self.Version))

            # End the timer
            end_time = time.time()

            # Calculate and print the elapsed time
            elapsed_time = end_time - start_time

            # Store the results in a dictionary
            summary = save_summary(cv_results, elapsed_time)

            fold_summary[fold] = elapsed_time

        return cv_results, fold_summary
