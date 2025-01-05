
import warnings
warnings.filterwarnings('ignore')

import time

import tensorflow as tf
import numpy as np
import os

from tensorflow import keras
from sklearn.metrics import roc_auc_score

from AE.AE_helper import Partition, k_fold_split,save_summary

from AmbientGAN.AmbientGAN_Algorithms import AmbientGAN_Generator, AmbientGAN_Standard_Discriminator, \
    AmbientGANInputPreparation
from AmbientGAN.AmbientGAN_helper import apply_measurement,adjust_sigma_decreasing

from TC.TC_helper import save_tensor
from InfoGAN.infoGAN_helper import save_metrics, save_indices, save_ori_tensor

class AmbientGAN:
    def __init__(self,tensor,parameters):
        self.tensor = tensor
        self.parameters = parameters

        # Assuming you've already defined noise_dim, cont_code1_dim, and cont_code2_dim somewhere in your code
        self.noise_dim = 100
        self.cont_code1_dim = 20

        self.max_iter = parameters['max_iter']
        self.learning_rate = parameters['lr']

        self.is_training = True
        self.slice_size = None

        '''
        Version is:
            'Standard'=>0
            'WithTensor'=>1
        '''
        self.AmbientGAN_mode = 'Standard' # 'Standard','WithTensor'
        self.Version = np.nan

        self.model_name = parameters['model_str']
        self.course = parameters['course']
        self.lesson_id = parameters['lesson_id']
        self.learning_state = parameters['learning_stage']

    def RunModel(self,model_iter):
        global measured_test_fake_data
        print("Run New AmbientGAN")

        tf.random.set_seed(22)
        np.random.seed(22)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        assert tf.__version__.startswith('2.')

        self.tensor = tf.where(tf.math.is_nan(self.tensor), -1.0, self.tensor)

        self.slice_size = 1

        sub_tensors, self.output_size = Partition(self.tensor, slice_size=self.slice_size, mode="Average", filter='normal')

        start_time = time.time()

        if self.slice_size != self.tensor.shape[0]:
            '''
            This should be few-shot scenario
            '''
            cv_results = []
            summary={}

            fold_summary = {}

            for fold, ((train_set, test_set), (train_index, test_index)) in enumerate(k_fold_split(sub_tensors, n_splits=5)):
                print(f"Fold {fold + 1}:")
                print(f"  Number of training sub-tensors: {len(train_set)}")
                print(f"  Number of testing sub-tensors: {len(test_set)}")

                train_set_tensor = tf.convert_to_tensor(train_set)
                test_set_tensor = tf.convert_to_tensor(test_set)

                train_mode = "origin_train"
                save_indices(train_index, self.max_iter, train_mode, self.model_name, self.course, self.lesson_id,
                             self.learning_state, fold, model_iter)

                save_ori_tensor(train_set, self.max_iter, train_mode, self.model_name, self.course, self.lesson_id,
                                self.learning_state, fold, model_iter)

                test_mode = "origin_test"
                save_indices(test_index, self.max_iter, test_mode, self.model_name, self.course, self.lesson_id,
                             self.learning_state, fold, model_iter)

                save_ori_tensor(test_set, self.max_iter, test_mode, self.model_name, self.course, self.lesson_id,
                                self.learning_state, fold, model_iter)

                input_shape = [train_set_tensor.shape[1], train_set_tensor.shape[2], train_set_tensor.shape[3]]

                mask_train_x = tf.math.not_equal(train_set_tensor, -1.0)

                batch_size = train_set_tensor.shape[0]

                if self.AmbientGAN_mode=='Standard':
                    self.Version=0

                    preparation = AmbientGANInputPreparation(self.noise_dim, self.cont_code1_dim)
                    combined_input = preparation([train_set_tensor, batch_size])

                    generator=AmbientGAN_Generator(channel=train_set_tensor.shape[-1], output_size=self.output_size, input_shape=input_shape)

                    discriminator=AmbientGAN_Standard_Discriminator(self.cont_code1_dim)

                    g_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)
                    d_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_2=0.5)

                    train_perf = []
                    test_perf = []

                    converge = False
                    epoch = 0
                    test_mae_loss, test_rmse_loss, test_rse_loss, test_auc_score = 0., 0., 0., 0.

                    initial_sigma = 1.0  # Starting value of sigma
                    reduction_rate = 0.99  # Sigma reduction rate per epoch
                    min_sigma = 0.8  # Minimum value of sigma

                    noise, cont_code1 = combined_input

                    while not converge:
                        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                           fake_data = generator(combined_input, self.is_training)
                           current_sigma=adjust_sigma_decreasing(initial_sigma, reduction_rate, epoch, min_sigma=min_sigma)
                           measured_fake_data = apply_measurement(fake_data, sigma=current_sigma)

                           fake_out,fake_cont_out1=discriminator(measured_fake_data,self.is_training)
                           real_out,real_cont_out1=discriminator(train_set_tensor,self.is_training)

                           bce_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)

                           '''
                           Error measurement
                           '''
                           train_set_tensor = tf.cast(train_set_tensor, tf.float32)
                           measured_fake_data = tf.cast(measured_fake_data, tf.float32)

                           train_mae = tf.losses.mean_absolute_error(train_set_tensor[mask_train_x],
                                                                     measured_fake_data[mask_train_x])
                           train_mae = tf.reduce_mean(train_mae)

                           train_mse = tf.losses.mean_squared_error(train_set_tensor[mask_train_x],
                                                                        measured_fake_data[mask_train_x])
                           train_mse = tf.reduce_mean(train_mse)
                           train_rmse = tf.sqrt(train_mse)

                           # Calculate MSE
                           mse = tf.reduce_mean(tf.square(train_set_tensor[mask_train_x] - measured_fake_data[mask_train_x]))

                           # Compute RSE
                           mean_squared_true_values = tf.reduce_mean(tf.square(train_set_tensor[mask_train_x]))
                           train_rse = mse / mean_squared_true_values

                           # Compute AUC

                           # Filter the true labels and predictions based on the mask
                           true_labels = train_set_tensor[mask_train_x]
                           predictions = measured_fake_data[mask_train_x]

                           # Initialize the AUC metric
                           auc_metric = tf.keras.metrics.AUC()

                           # Update the state of the AUC metric with your true labels and predictions
                           auc_metric.update_state(true_labels, predictions)

                           # Calculate the AUC
                           train_auc = auc_metric.result().numpy()

                           cross_entropy=bce_loss(train_set_tensor[mask_train_x],measured_fake_data[mask_train_x])

                           '''
                           Generator Loss
                           '''
                           fake_loss_g = bce_loss(tf.ones_like(fake_out),fake_out)
                           cont1_loss = tf.reduce_mean(tf.square(fake_cont_out1 - cont_code1))

                           gen_loss=fake_loss_g+cont1_loss+train_mae+train_rmse

                           '''
                           Discriminator Loss
                           '''
                           real_loss= bce_loss(tf.ones_like(real_out),real_out)
                           fake_loss_d = bce_loss(tf.zeros_like(fake_out), fake_out)
                           cont1_loss = tf.reduce_mean(tf.square(real_cont_out1-cont_code1))

                           disc_loss=real_loss+fake_loss_d+cont1_loss

                        gen_grad=gen_tape.gradient(gen_loss,generator.trainable_variables)
                        disc_grad=disc_tape.gradient(disc_loss,discriminator.trainable_variables)

                        g_optimizer.apply_gradients(zip(gen_grad,generator.trainable_variables))
                        d_optimizer.apply_gradients(zip(disc_grad,discriminator.trainable_variables))

                        print(epoch, "d-loss:", float(disc_loss), "g-loss:", float(gen_loss), "train_rmse:", float(train_rmse))

                        train_perf.append([train_mae, train_rmse, train_rse, train_auc, cross_entropy])

                        loss=gen_loss+disc_loss

                        # if loss == np.nan:
                        #     self.learning_rate *= 0.1
                        # else:
                        #     train_loss_list.append(loss)
                        #     epoch += 1
                        #
                        #     if epoch > 1 and epoch < self.epoches:
                        #         if loss > train_loss_list[-2]:
                        #             self.learning_rate *= 0.5
                        #
                        #         # if loss > np.mean(train_loss_list[-5:]):
                        #         #     converge = True
                        #
                        #     elif epoch == self.epoches:
                        #         converge = True

                        epoch += 1
                        if epoch == self.max_iter:
                            converge = True

                    mode = "train"
                    save_metrics(train_perf, self.max_iter, mode, self.model_name, self.course, self.lesson_id,
                                 self.learning_state, fold, model_iter)

                    '''
                    Training dataset
                    '''
                    test_batch_size = test_set_tensor.shape[0]
                    mask_test_x = tf.math.not_equal(test_set_tensor, -1.0)

                    # Prepare combined input for test data generation
                    preparation = AmbientGANInputPreparation(self.noise_dim, self.cont_code1_dim)
                    test_combined_input = preparation([None, test_batch_size])  # Use test_batch_size

                    # Generate fake data using the generator
                    test_fake_data = generator(test_combined_input, training=False)
                    measured_test_fake_data = apply_measurement(test_fake_data)
                    measured_test_fake_data = tf.cast(measured_test_fake_data, tf.float32)

                    test_mae_loss = tf.losses.mean_absolute_error(test_set_tensor[mask_test_x],measured_test_fake_data[mask_test_x])
                    test_mae_loss = tf.reduce_mean(test_mae_loss)

                    test_mse_loss = tf.losses.mean_squared_error(test_set_tensor[mask_test_x],measured_test_fake_data[mask_test_x])
                    test_mse_loss = tf.reduce_mean(test_mse_loss)
                    test_rmse_loss = tf.sqrt(test_mse_loss)

                    test_set_tensor = tf.cast(test_set_tensor, tf.float32)

                    # Calculate MSE
                    test_mse = tf.reduce_mean(tf.square(test_set_tensor[mask_test_x] - measured_test_fake_data[mask_test_x]))

                    # Compute RSE
                    mean_squared_true_values = tf.reduce_mean(tf.square(test_set_tensor[mask_test_x]))
                    test_rse_loss = test_mse / mean_squared_true_values

                    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                    cross_entropy = bce_loss(test_set_tensor[mask_test_x],measured_test_fake_data[mask_test_x])

                    try:
                        test_auc_score = roc_auc_score(test_set_tensor[mask_test_x], measured_test_fake_data[mask_test_x])
                        test_auc_score = tf.reduce_mean(test_auc_score)
                    except ValueError as e:
                        if 'Only one class present' in str(e):
                            test_auc_score = np.nan
                        else:
                            raise  # re-raise the exception if it's not the "Only one class" error

                    print("epoch:", epoch, "test_mae_loss:", float(test_mae_loss), "test_rmse_loss:",
                          float(test_rmse_loss), "test_rse_loss:", float(test_rse_loss), "test_auc_score:", float(test_auc_score), "cross_entropy:",float(cross_entropy))

                    test_perf.append([test_mae_loss, test_rmse_loss, test_rse_loss, test_auc_score, cross_entropy])

                    mode = "test"
                    save_metrics(test_perf, self.max_iter, mode, self.model_name, self.course, self.lesson_id,
                                 self.learning_state, fold, model_iter)

                    combined_tensor = tf.zeros((self.tensor.shape))

                    train_set_tensor_reshape = tf.squeeze(fake_data, axis=1)
                    test_set_tensor_reshape = tf.squeeze(measured_test_fake_data, axis=1)

                    train_indices_tensor = tf.constant(train_index)
                    test_indices_tensor = tf.constant(test_index)

                    combined_tensor = tf.tensor_scatter_nd_update(combined_tensor, train_indices_tensor[:, tf.newaxis],
                                                                  train_set_tensor_reshape)
                    combined_tensor = tf.tensor_scatter_nd_update(combined_tensor, test_indices_tensor[:, tf.newaxis],
                                                                  test_set_tensor_reshape)

                    mode = "train_combined"
                    save_tensor(combined_tensor, self.max_iter, mode, self.model_name, self.course, self.lesson_id,
                                self.learning_state, fold, model_iter)

                    cv_results.append((fold, float(test_mae_loss), float(test_rmse_loss), float(test_rse_loss),float(test_auc_score),float(cross_entropy),
                                       self.max_iter, self.Version))

                    # End the timer
                    end_time = time.time()

                    # Calculate and print the elapsed time
                    elapsed_time = end_time - start_time

                    # Store the results in a dictionary
                    summary = save_summary(cv_results, elapsed_time)

                    fold_summary[fold] = elapsed_time

                return cv_results, fold_summary