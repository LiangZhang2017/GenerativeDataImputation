import numpy as np
import tensorly as tl
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from math import sqrt

from AE.AE_helper import save_summary
from TC.TC_helper import tensor_to_numpy, perform_k_fold_cross_validation_numpy_tensor, \
    perform_k_fold_cross_validation_tf, save_indices, save_metrics

from scipy.special import expit

from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
import time
import tensorflow as tf

from TC.TC_helper import save_tensor


class CPDecomposition:
    def __init__(self, tensor, parameters):
        print("CP Decomposition")

        np.random.seed(22)

        self.model_name = parameters['model_str']
        self.course = parameters['course']
        self.lesson_id = parameters['lesson_id']
        self.learning_state = parameters['learning_stage']

        self.is_rank = True

        self.tensor = tensor

        self.parameters = parameters
        self.num_learner = None
        self.num_question = None
        self.num_attempt = None
        self.num_features = parameters["features_dim"]

        self.lambda_t = parameters['lambda_t']
        self.lambda_q = parameters['lambda_q']
        self.lr = parameters['lr']
        self.lambda_w = parameters['lambda_w']

        self.U = None
        self.V = None
        self.W = None
        self.T = [0]

        self.use_bias = True
        self.binarized_question = True
        self.train_tensor_np = None
        self.is_rank = True

        self.bias_s = None
        self.bias_t = None
        self.bias_q = None
        self.global_bias = None

        self.max_iter = parameters['max_iter']

        self.min_iter = 10

        self.Version = 0

        self.train_tensor = None
        self.test_tensor = None

    def get_question_prediction(self, learner, attempt, question):
        # CP Decomposition prediction

        pred = np.sum(self.U[learner, :] * self.V[attempt, :] * self.W[question, :])

        # Add biases
        if self.use_bias:
            pred += self.bias_s[learner] + self.bias_q[question] + self.bias_t[attempt] + self.global_bias

        if self.binarized_question:
            pred = expit(pred)

        return pred

    def grad_U(self, learner, attempt, question, obs=None):
        grad = np.zeros_like(self.U[learner, :])

        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad = -2. * (obs - pred) * self.V[attempt, :] * self.W[question, :] + 2. * self.lambda_t * self.U[learner,
                                                                                                        :]

        return grad

    def grad_V(self, learner, attempt, question, obs=None):
        grad = np.zeros_like(self.V[attempt, :])

        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad = -2. * (obs - pred) * self.U[learner, :] * self.W[question, :] + 2. * self.lambda_t * self.V[attempt,
                                                                                                        :]

        return grad

    def grad_W(self, learner, attempt, question, obs=None):
        grad = np.zeros_like(self.W[question, :])

        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad = -2. * (obs - pred) * self.U[learner, :] * self.V[attempt, :] + 2. * self.lambda_q * self.W[question,
                                                                                                       :]

        return grad

    def grad_bias_q(self, learner, attempt, question, obs=None):
        grad = 0.
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad -= 2. * (obs - pred)
            if self.binarized_question:
                grad *= pred * (1. - pred)
        return grad

    def grad_bias_s(self, learner, attempt, question, obs=None):
        grad = 0.
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad -= 2. * (obs - pred)
            if self.binarized_question:
                grad *= pred * (1. - pred)
        return grad

    def grad_bias_t(self, learner, attempt, question, obs=None):
        grad = 0.
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad -= 2. * (obs - pred)
            if self.binarized_question:
                grad *= pred * (1. - pred)
        return grad

    def grad_global_bias(self, learner, attempt, question, obs=None):
        grad = 0.

        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad -= 2.0 * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2.0 * (obs - pred)
        return grad

    def optimize_sgd(self, learner, attempt, question, obs=None):
        grad_u = self.grad_U(learner, attempt, question, obs)
        grad_v = self.grad_V(learner, attempt, question, obs)
        grad_w = self.grad_W(learner, attempt, question, obs)

        self.U[learner, :] -= self.lr * grad_u
        self.V[attempt, :] -= self.lr * grad_v
        self.W[question, :] -= self.lr * grad_w

        # Update biases using their respective gradients
        self.bias_q[question] -= self.lr * self.grad_bias_q(learner, attempt, question, obs)
        self.bias_s[learner] -= self.lr * self.grad_bias_s(learner, attempt, question, obs)
        self.bias_t[attempt] -= self.lr * self.grad_bias_t(learner, attempt, question, obs)
        self.global_bias -= self.lr * self.grad_global_bias(learner, attempt, question, obs)

    def get_loss(self):
        loss, square_loss_q, ranking_gain = 0., 0., 0.

        train_obs = []
        train_pred = []

        for (learner, question, attempt, obs) in self.train_tensor_np:
            learner = int(learner)
            question = int(question)
            attempt = int(attempt)

            pred = self.get_question_prediction(learner, attempt, question)

            if pred is not None:
                train_obs.append(obs)
                train_pred.append(pred)

        # Filter out pairs where train_obs is NaN
        train_obs, train_pred = zip(
            *[(obs, pred) for obs, pred in zip(train_obs, train_pred) if not np.isnan(obs)])

        q_mae = mean_absolute_error(train_obs, train_pred)
        q_auc = roc_auc_score(train_obs, train_pred)
        q_rmse = sqrt(mean_squared_error(train_obs, train_pred))

        mse = mean_squared_error(train_obs, train_pred)

        # Calculate Relative Squared Error (RSE)
        q_rse = mse / np.mean(np.square(train_obs))

        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        cross_entropy = bce_loss(train_obs, train_pred)

        square_loss_q = q_rmse

        # Assuming self.U, self.V, self.W are the factor matrices obtained from CP decomposition
        factors = [self.U, self.V, self.W]
        weights = np.ones(self.U.shape[1])  # or the number of columns in any factor matrix

        # Creating a CP tensor representation with weights and factor matrices
        cp_tensor = (weights, factors)

        # Reconstruct the tensor from the factor matrices
        pred_tensor = cp_to_tensor(cp_tensor)

        sum_n = 0.
        if self.is_rank is True:
            for attempt in range(0, self.num_attempt):
                if attempt > 0:
                    for n in range(attempt - 1, attempt):
                        slice_n = np.subtract(pred_tensor[:, attempt, :], pred_tensor[:, n, :])
                        slice_sig = np.log(expit(slice_n))
                        sum_n += np.sum(slice_sig)
                    ranking_gain = self.lambda_w * sum_n
                else:
                    ranking_gain = 0.

        loss = square_loss_q + self.lambda_t * np.linalg.norm(self.U) ** 2 + self.lambda_q * np.linalg.norm(
            self.W) ** 2 - ranking_gain

        print("Overall Loss {}".format(loss))

        metrics_all = [q_mae, q_rmse, q_rse, q_auc, cross_entropy]

        return loss, metrics_all

    def train(self, fold, model_iter):
        print("training tensor")

        train_perf = []
        converge = False

        loss, metrics_all = self.get_loss()
        loss_list = [loss]
        best_U, best_V, best_W = [0] * 3
        best_bias_s, best_bias_t, best_bias_q = [0] * 3

        for iter_num in range(self.max_iter):
            while not converge:
                for (learner, question, attempt, obs) in self.train_tensor_np:
                    learner = int(learner)
                    question = int(question)
                    attempt = int(attempt)

                    self.optimize_sgd(learner, attempt, question, obs)

                loss, metrics_all = self.get_loss()
                train_perf.append([metrics_all[0], metrics_all[1], metrics_all[2], metrics_all[3], metrics_all[4]])

                best_U = np.copy(self.U)
                best_V = np.copy(self.V)
                best_W = np.copy(self.W)

                best_bias_s = np.copy(self.bias_s)
                best_bias_t = np.copy(self.bias_t)
                best_bias_q = np.copy(self.bias_q)

                iter_num += 1
                if iter_num == self.max_iter:
                    converge = True

                loss_list.append(loss)

                print(iter_num, "MAE:", metrics_all[0], "RMSE:", metrics_all[1], "RSE:", metrics_all[2], "AUC:", metrics_all[3], "Cross Entropy:", metrics_all[4])

        self.U = best_U
        self.V = best_V
        self.W = best_W
        self.bias_s = best_bias_s
        self.bias_t = best_bias_t
        self.bias_q = best_bias_q

        print("all shape", self.U.shape, self.V.shape, self.W.shape)

        # Determine the dimensions from the factor matrices
        dim1 = self.U.shape[0]
        dim2 = self.V.shape[0]
        dim3 = self.W.shape[0]

        # Create a zero-initialized tensor with inferred dimensions
        T = np.zeros((dim1, dim3, dim2))

        # Compute the original tensor by summing outer products of factor matrices
        for r in range(self.U.shape[1]):  # Assuming all factor matrices have the same number of columns
            T += np.outer(np.outer(self.U[:, r], self.W[:, r]), self.V[:, r]).reshape(dim1, dim3, dim2)

        print("The obtained tensor shape is {}".format(T.shape))

        # print("The obtained tensor is {}".format(T))

        print("self.bias_s.shape is {}".format(self.bias_s.shape))
        print("self.bias_t.shape is {}".format(self.bias_t.shape))
        print("self.bias_q.shape is {}".format(self.bias_q.shape))

        # Corrected expansions
        bias_s_expanded = self.bias_s[:, np.newaxis, np.newaxis]
        bias_t_expanded = self.bias_t[np.newaxis, np.newaxis, :]
        bias_q_expanded = self.bias_q[np.newaxis, :, np.newaxis]

        # Add the biases along different dimensions
        T = T + bias_s_expanded + bias_t_expanded + bias_q_expanded

        T = np.where(T > 100, 1, T)
        T = np.where(T < -100, 0, T)

        T = expit(T)

        self.T = T

        mode = "train"
        save_tensor(T, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state, fold,
                    model_iter)

        save_metrics(train_perf, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state,
                     fold, model_iter)

    def test(self, fold, model_iter, test_indices):
        perf_dict = []
        curr_pred_list = []
        curr_obs_list = []

        print("test")

        test_indices_tf = tf.constant(test_indices)

        # Use tf.gather_nd to extract values at these indices
        test_real_values = tf.gather_nd(self.test_tensor, test_indices_tf)
        # Convert to numpy array if needed
        test_real_values_np = test_real_values.numpy()

        pred_test_values = tf.gather_nd(self.T, test_indices_tf)
        pred_test_values_np = pred_test_values.numpy()

        # print("test_real_values_np are {}".format(test_real_values_np))
        # print("pred_test_values_np are {}".format(pred_test_values_np))

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

        # n_learners = len(set(learner_list))
        # n_questions = len(set(question_list))
        # n_attempts = len(set(attempt_list))

        perf_dict.append((test_mae, test_rmse, test_rse, float(test_auc_score), float(cross_entropy)))

        mode = "test"
        save_metrics(perf_dict, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state,
                     fold, model_iter)

        return perf_dict

    def RunModel(self, model_iter):
        print("start of RunCPDecomposition")

        start_time = time.time()

        cv_results = []

        fold_summary = {}

        for fold, (train_tensor, test_tensor, train_indices, test_indices) in enumerate(perform_k_fold_cross_validation_tf(self.tensor, k=5)):
            print(f"Training on fold {fold + 1}")

            # Train and test your model here using train_tensor and test_tensor
            print("train_tensor.shape is : {}".format(train_tensor.shape))
            print("test_tensor.shape is: {}".format(test_tensor.shape))

            print("self.tensor shape is {}".format(self.tensor.shape))

            tl.set_backend('numpy')

            # print("train tensor is {}".format(train_tensor[:,0]))

            # self.num_learner = len(np.unique(self.numpy_T[:,0]))
            # self.num_question = len(np.unique(self.numpy_T[:,1]))
            # self.num_attempt = len(np.unique(self.numpy_T[:,2]))

            self.num_learner = self.tensor.shape[0]
            self.num_question = self.tensor.shape[1]
            self.num_attempt = self.tensor.shape[2]

            self.U = np.random.random_sample((self.num_learner, self.num_features))
            self.V = np.random.random_sample((self.num_attempt, self.num_features))
            self.W = np.random.random_sample((self.num_question, self.num_features))

            self.bias_s = np.zeros(self.num_learner)
            self.bias_t = np.zeros(self.num_attempt)
            self.bias_q = np.zeros(self.num_question)
            self.global_bias = np.nanmean(train_tensor)

            self.train_tensor_np = tensor_to_numpy(train_tensor)
            self.test_tensor_np = tensor_to_numpy(test_tensor)

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

            self.train(fold, model_iter)
            test_results = self.test(fold, model_iter, test_indices)

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
