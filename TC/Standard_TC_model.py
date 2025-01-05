import tensorflow as tf
import os
import numpy as np

from AE.AE_helper import save_summary
from TC.TC_helper import tensor_to_numpy, sigmoid, perform_k_fold_cross_validation_tf
from scipy.special import expit
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from numpy import linalg as LA
from math import sqrt

from TC.TC_helper import save_tensor, save_indices, save_metrics

import time


class Standard_TC:
    def __init__(self, tensor, parameters):
        print("TC")

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

        self.use_bias_t = True
        self.use_global_bias = True
        self.binarized_question = True
        self.lambda_t = parameters['lambda_t']
        self.lambda_q = parameters['lambda_q']
        self.lambda_bias = parameters['lambda_bias']
        self.lambda_w = parameters['lambda_w']
        self.lr = parameters['lr']

        self.U = None
        self.V = None
        self.T = [0]

        self.bias_s = None
        self.bias_t = None
        self.bias_q = None
        self.global_bias = None

        self.train_tensor_np = None

        self.max_iter = parameters['max_iter']

        self.min_iter = 10

        self.Version = 0

        self.train_tensor = None
        self.test_tensor = None

    def get_question_prediction(self, learner, attempt, question):
        """
        predict value at tensor T[attempt, student, question]
        all the following indexes start from zero indexing
        :param attempt: attempt index
        :param student: student index
        :param question: question index
        :return: predicted value of tensor T[learner, attempt, question]
        """

        pred = np.dot(self.U[learner, :], self.V[:, attempt, question])  # vector*vector

        if self.use_bias_t:
            if self.use_global_bias:
                pred += self.bias_s[learner] + self.bias_t[attempt] + self.bias_q[
                    question] + self.global_bias  # MTF Equation
            else:
                pred += self.bias_s[learner] + self.bias_t[attempt] + self.bias_q[question]
        else:
            if self.use_global_bias:
                pred += self.bias_s[learner] + self.bias_q[question] + self.global_bias
            else:
                pred += self.bias_s[learner] + self.bias_q[question]

        if self.binarized_question:
            pred = sigmoid(pred)  # Sigmoid functions most often show a return value (y axis) in the range 0 to 1.

        return pred

    def grad_Q_K(self, learner, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept-question-attempts association
        of a question in Tensor,
        :param attempt: index
        :param student:  index
        :param question:  index
        :param obs: the value at Y[attempt, student, question]
        :return:
        """
        grad = np.zeros_like(self.V[:, attempt, question])

        # print("obs is {}".format(obs))
        # print("obs is nan {}".format(np.isnan(obs)))
        # print("obs is not None:{}".format(np.isnan(obs)))

        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad = -2. * (obs - pred) * pred * (1. - pred) * self.U[learner, :] + 2. * self.lambda_q * self.V[:,
                                                                                                           attempt,
                                                                                                           question]
            else:
                grad = -2. * (obs - pred) * self.U[learner, :] + 2. * self.lambda_q * self.V[:, attempt, question]
        return grad

    def grad_T_ij(self, learner, attempt, question, obs=None):
        grad = np.zeros_like(self.U[learner, :])

        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if (self.binarized_question):
                grad = -2. * (obs - pred) * pred * (1. - pred) * self.V[:, attempt,
                                                                 question] + 2. * self.lambda_t * self.U[learner, :]
            else:
                grad = -2. * (obs - pred) * self.V[:, attempt, question] + 2. * self.lambda_t * self.U[learner, :]
        return grad

    def grad_bias_q(self, learner, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param attempt:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = 0.

        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad -= 2. * (obs - pred) * pred * (1. - pred) + 2. * self.lambda_bias * self.bias_q[question]
            else:
                grad -= 2. * (obs - pred) + 2. * self.lambda_bias * self.bias_q[question]
        return grad

    def grad_bias_s(self, learner, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_s
        :param attempt:
        :param student:
        :param material: material material of that resource, here is the question
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad -= 2. * (obs - pred) * pred * (1. - pred) + 2.0 * self.lambda_bias * self.bias_s[learner]
            else:
                grad -= 2. * (obs - pred) + 2.0 * self.lambda_bias * self.bias_s[learner]
        return grad

    def grad_bias_t(self, learner, attempt, question, obs=None):
        grad = 0.
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad -= 2. * (obs - pred) * pred * (1. - pred) + 2.0 * self.lambda_bias * self.bias_t[attempt]
            else:
                grad -= 2. * (obs - pred) + 2.0 * self.lambda_bias * self.bias_t[attempt]

        return grad

    def grad_global_bias(self, learner, attempt, question, obs=None):
        grad = 0.

        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad -= 2.0 * (obs - pred) * pred * (1. - pred) + 2. * self.lambda_bias * self.global_bias
            else:
                grad -= 2.0 * (obs - pred) + 2. * self.lambda_bias * self.global_bias
        return grad

    def optimize_sgd(self, learner, attempt, question, obs=None):
        """
       training the T and Q with stochastic gradient descent
       :param attempt:
       :param student:
       :param material: material material of that resource, it's question here
       :return:
        """
        grad_q = self.grad_Q_K(learner, attempt, question, obs)

        self.V[:, attempt, question] -= self.lr * grad_q
        self.V[:, attempt, question][self.V[:, attempt, question] < 0.] = 0.
        if self.lambda_q == 0.:
            sum_val = np.sum(self.V[:, attempt, question])
            if sum_val != 0:
                self.V[:, attempt, question] /= sum_val

        grad_t = self.grad_T_ij(learner, attempt, question, obs)
        self.U[learner, :] -= self.lr * grad_t

        self.bias_q[question] -= self.lr * self.grad_bias_q(learner, attempt, question, obs)
        self.bias_s[learner] -= self.lr * self.grad_bias_s(learner, attempt, question, obs)

        if self.use_bias_t:
            self.bias_t[attempt] -= self.lr * self.grad_bias_t(learner, attempt, question, obs)

        if self.use_global_bias:
            self.global_bias -= self.lr * self.grad_global_bias(learner, attempt, question, obs)

    def get_loss(self):
        """
        Override the function in super class
        """
        loss, square_loss, reg_bias, ranking_gain = 0., 0., 0., 0.
        square_loss_q = 0.
        q_count = 0.

        train_obs = []
        train_pred = []

        for (learner, question, attempt, obs) in self.train_tensor_np:  # obs refers the score that we observe

            learner = int(learner)
            question = int(question)
            attempt = int(attempt)

            pred = self.get_question_prediction(learner, attempt, question)

            train_obs.append(obs)
            train_pred.append(pred)

            q_count += 1

        # Filter out pairs where train_obs is NaN
        train_obs, train_pred = zip(
            *[(obs, pred) for obs, pred in zip(train_obs, train_pred) if not np.isnan(obs)])

        # print("train_obs is {}".format(train_obs))
        # print("train_pred is {}".format(train_pred))

        q_mae = mean_absolute_error(train_obs, train_pred)
        q_auc = roc_auc_score(train_obs, train_pred)
        q_rmse = sqrt(mean_squared_error(train_obs, train_pred))

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(train_obs, train_pred)

        # Calculate Relative Squared Error (RSE)
        q_rse = mse / np.mean(np.square(train_obs))

        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        cross_entropy = bce_loss(train_obs, train_pred)

        square_loss_q = q_rmse

        reg_U = LA.norm(self.U) ** 2  # Frobenius norm, 2-norm
        reg_V = LA.norm(self.V) ** 2

        reg_features = self.lambda_t * reg_U + self.lambda_q * reg_V

        if self.lambda_bias:
            if self.use_bias_t:
                # reg_bias=self.lambda_bias*(LA.norm(self.bias_s)**2+LA.norm(self.bias_t)**2+LA.norm(self.bias_q)**2)
                reg_bias = self.lambda_bias * (
                        LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_t) ** 2 + LA.norm(self.bias_q) ** 2)
            else:
                reg_bias = self.lambda_bias * (LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_q) ** 2)
        else:
            reg_bias = 0

        trans_V = np.transpose(self.V, (1, 0, 2))

        pred_tensor = np.dot(self.U, trans_V)  # learner*attempt*question

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

            loss = square_loss_q + reg_features + reg_bias - ranking_gain
        else:
            loss = square_loss_q + reg_features + reg_bias

        print("Overall Loss {}".format(loss))

        metrics_all = [q_mae, q_rmse, q_rse, q_auc, cross_entropy]

        print("Test ", "MAE:", q_mae, "RMSE:", q_rmse, "RSE:", q_rse, "AUC:", q_auc, "cross_entropy:",cross_entropy)

        return loss, metrics_all, reg_features, reg_bias

    def train(self, fold, model_iter):
        print("training tensor")

        train_perf = []
        converge = False

        loss, metrics_all, reg_features, reg_bias = self.get_loss()
        loss_list = [loss]
        best_U, best_V = [0] * 2
        best_bias_s, best_bias_t, best_bias_q = [0] * 3

        for iter_num in range(self.max_iter):
            while not converge:
                for (learner, question, attempt, obs) in self.train_tensor_np:
                    learner = int(learner)
                    question = int(question)
                    attempt = int(attempt)

                    self.optimize_sgd(learner, attempt, question, obs)

                loss, metrics_all, reg_features, reg_bias = self.get_loss()

                train_perf.append([metrics_all[0], metrics_all[1], metrics_all[2], metrics_all[3], metrics_all[3], metrics_all[4]])

                best_U = np.copy(self.U)
                best_V = np.copy(self.V)
                best_bias_s = np.copy(self.bias_s)
                best_bias_t = np.copy(self.bias_t)
                best_bias_q = np.copy(self.bias_q)

                # if iter_num == self.max_iter:
                #     loss_list.append(loss)
                #     converge = True
                # elif iter_num >= self.min_iter and loss >= np.mean(loss_list[-5:]):
                #     converge = True
                # elif loss == np.nan:
                #     self.lr *= 0.1
                # elif loss > loss_list[-1]:
                #     loss_list.append(loss)
                #     self.lr *= 0.5
                #     iter_num += 1
                #     print("The iter_num is: " + str(iter_num))
                # else:
                #     loss_list.append(loss)
                #     iter_num += 1
                #     print("The iter_num is: " + str(iter_num))

                iter_num += 1
                if iter_num == self.max_iter:
                    converge = True

                print(iter_num, metrics_all[0], metrics_all[1], metrics_all[2],
                      metrics_all[3])  # [iter_num,q_mae,q_rmse,q_auc]

        # print("train_perf is: {}".format(train_perf))

        self.U = best_U
        self.V = best_V
        self.bias_s = best_bias_s
        self.bias_t = best_bias_t
        self.bias_q = best_bias_q

        trans_V = np.transpose(self.V, (1, 0, 2))

        T = np.dot(self.U, trans_V) + self.global_bias
        for i in range(0, self.num_learner):
            T[i, :, :] = T[i, :, :] + self.bias_s[i]
        for j in range(0, self.num_attempt):
            T[:, j, :] = T[:, j, :] + self.bias_t[j]
        for k in range(0, self.num_question):
            T[:, :, k] = T[:, :, k] + self.bias_q[k]

        T = np.where(T > 100, 1, T)
        T = np.where(T < -100, 0, T)

        T = expit(T)

        self.T = T

        mode = "train"
        save_tensor(T, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state, fold,
                    model_iter)

        save_metrics(train_perf, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state, fold,
                    model_iter)

    def test(self, fold, model_iter, test_indices):

        """
        :return: performance metrics mean squared error, RMSE and RSE
        """

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

        perf_dict.append((test_mae, test_rmse, test_rse, float(test_auc_score), float(cross_entropy)))

        mode = "test"
        save_metrics(perf_dict, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state,
                     fold, model_iter)

        return perf_dict

    def RunModel(self, model_iter):
        print("start of RunTC")

        start_time = time.time()
        cv_results = []
        fold_summary = {}

        for fold, (train_tensor, test_tensor, train_indices, test_indices) in enumerate(
                perform_k_fold_cross_validation_tf(self.tensor, k=5)):

            print(f"Training on fold {fold}")

            # Train and test your model here using train_tensor and test_tensor
            print("train_tensor.shape is : {}".format(train_tensor.shape))
            print("test_tensor.shape is: {}".format(test_tensor.shape))

            # print("train_indices are: {}".format(train_indices))
            # print("test_indices are: {}".format(test_indices))

            tf.random.set_seed(22)
            np.random.seed(22)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            self.num_learner = self.tensor.shape[0]
            self.num_question = self.tensor.shape[1]
            self.num_attempt = self.tensor.shape[2]

            self.U = np.random.random_sample((self.num_learner, self.num_features))
            self.V = np.random.random_sample((self.num_features, self.num_attempt, self.num_question))

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

            '''
            Train data
            1. loss
            2. iteration
            '''

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