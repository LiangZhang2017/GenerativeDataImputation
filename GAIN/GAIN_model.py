import warnings

from GAIN.GAIN_Algorithms import GAIN_Gennerator_CNN, GAIN_Discriminator_CNN

warnings.filterwarnings('ignore')

import time

import tensorflow as tf
import numpy as np
import os

from tensorflow import keras
from sklearn.metrics import roc_auc_score

from AE.AE_helper import k_fold_split, save_summary
from CGAN.CGAN_helper import Partition
from GAIN.GAIN_helper import generate_hints, normalization

from TC.TC_helper import save_tensor
from GAIN.GAIN_helper import save_indices, save_ori_tensor, save_metrics

class GAIN:
    def __init__(self, tensor, parameters):

        # self.embedding_value = None
        self.output_size = None
        self.train_tensor = None
        self.tensor = tensor
        self.parameters = parameters

        self.z_dim = 100
        self.max_iter = parameters['max_iter']

        self.learning_rate = parameters['lr']
        self.hint_rate=0.9
        self.alpha=1
        self.h_dim = 6

        self.is_training = True
        self.slice_size = 1

        self.prune_slice_number=parameters['prune_slice_number']

        '''
        Version is:
            'Standard'=>0
            'CNN'=>1
        '''
        self.GAIN_mode = 'CNN'  # 'Standard','CNN'
        self.Version = np.nan

        self.model_name = parameters['model_str']
        self.course = parameters['course']
        self.lesson_id = parameters['lesson_id']
        self.learning_state = parameters['learning_stage']
        self.batch_size = 10

    def compute_generator_loss(self, d_fake, mask_train_x, G_sample, train_set_tensor):
        # Generator tries to fool discriminator so we want the discriminator to output 1 for fake data
        g_loss_temp = -tf.reduce_mean((1 - mask_train_x) * tf.math.log(d_fake + 1e-8))

        # Mean Squared Error loss for the generator
        mse_loss = tf.reduce_mean((mask_train_x * train_set_tensor - mask_train_x*G_sample) ** 2) / tf.reduce_mean(mask_train_x)
        rmse_loss = tf.sqrt(mse_loss)

        # Total generator loss: adversarial loss + alpha * reconstruction loss
        g_loss = g_loss_temp + self.alpha * rmse_loss

        return g_loss

    def compute_discriminator_loss(self, d_real, d_fake):
        # Log loss for real data, discriminator should output 1 for real data
        real_loss = -tf.reduce_mean(tf.math.log(d_real + 1e-8))

        # Log loss for fake data, discriminator should output 0 for fake data
        fake_loss = -tf.reduce_mean(tf.math.log(1. - d_fake + 1e-8))

        # Total discriminator loss
        d_loss = real_loss + fake_loss
        return d_loss

    def compute_mae(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        mae = tf.reduce_mean(tf.abs(true_data[mask] - predicted_data[mask]))
        return mae.numpy()

    def compute_rmse(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        mse = tf.reduce_mean(tf.square(true_data[mask] - predicted_data[mask]))
        rmse = tf.sqrt(mse)
        return rmse.numpy()

    def compute_rse(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        true_mean = tf.reduce_mean(true_data[mask])
        total_variance = tf.reduce_mean(tf.square(true_data[mask] - true_mean))
        mse = tf.reduce_mean(tf.square(true_data[mask] - predicted_data[mask]))
        rse = mse / total_variance
        return rse.numpy()

    def compute_auc(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(true_data[mask], predicted_data[mask])
        return auc_metric.result().numpy()

    def compute_cross_entropy(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_data[mask], logits=predicted_data[mask]))
        return cross_entropy.numpy()

    def train_step(self, train_batch, generator, discriminator, g_optimizer, d_optimizer):
        # Convert types as necessary
        train_set_tensor = tf.cast(train_batch, tf.float32)

        # Compute the mask based on NaN values in the tensor
        mask_train_x = 1 - tf.cast(tf.math.is_nan(train_set_tensor), tf.float32)

        # Replace NaNs with zeros in the input tensor for processing
        train_set_tensor = tf.where(tf.math.is_nan(train_set_tensor), tf.zeros_like(train_set_tensor), train_set_tensor)

        # Generate noise
        noise_z = tf.random.uniform(shape=train_set_tensor.shape, minval=0, maxval=0.01)
        noise_X = mask_train_x * train_set_tensor + (1 - mask_train_x) * noise_z  # Noise Matrix

        # Generate hints
        hints_train = generate_hints(mask_train_x, hint_rate=self.hint_rate)
        hints_train_tensor = tf.convert_to_tensor(hints_train, dtype=tf.float32)

        # Prepare inputs for the generator
        generator_inputs = [noise_X, mask_train_x]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate synthetic data with generator
            G_sample = generator(generator_inputs, training=True)

            fake_data = mask_train_x * train_set_tensor + (1 - mask_train_x) * G_sample

            # Predict with discriminator
            d_fake = discriminator(fake_data, hints_train_tensor, training=True)
            d_real = discriminator(train_set_tensor, hints_train_tensor, training=True)

            # Calculate losses
            g_loss = self.compute_generator_loss(d_fake, mask_train_x, G_sample, train_set_tensor)
            d_loss = self.compute_discriminator_loss(d_real, d_fake)

        # Compute gradients
        gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
        disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

        # Apply gradients
        g_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        return {'g_loss': g_loss.numpy(), 'd_loss': d_loss.numpy()}, {'G_sample': G_sample.numpy(), 'fake_data': fake_data.numpy()}

    def test_step(self, test_batch, generator):
        test_set_tensor = tf.cast(test_batch, tf.float32)
        mask_test_x = 1 - tf.cast(tf.math.is_nan(test_set_tensor), tf.float32)
        test_set_tensor = tf.where(tf.math.is_nan(test_set_tensor), tf.zeros_like(test_set_tensor), test_set_tensor)
        G_sample = generator([test_set_tensor, mask_test_x], training=False)
        return G_sample.numpy()

    def RunModel(self, model_iter):
        global test_fake_data, ori_test_tensor, full_generated_train_data
        print("Run New GAIN")
        print("self.tensor shape is {}".format(self.tensor.shape))

        tf.random.set_seed(22)
        np.random.seed(22)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        assert tf.__version__.startswith('2.')

        sub_tensors, self.output_size = Partition(self.tensor, slice_size=self.slice_size, mode="Average", filter='normal')

        start_time = time.time()

        if self.slice_size != self.tensor.shape[0]:
            '''
            This should be few-shot scenario
            '''
            cv_results = []
            fold_summary = {}

            for fold, ((train_set, test_set), (train_index, test_index)) in enumerate(k_fold_split(sub_tensors, n_splits=5)):
                print(f"Fold {fold + 1}:")
                print(f"Number of training sub-tensors: {len(train_set)}")
                print(f"Number of testing sub-tensors: {len(test_set)}")

                train_mode = "origin_train"
                save_indices(train_index, self.max_iter, train_mode, self.model_name, self.course, self.lesson_id,
                                self.learning_state, self.prune_slice_number, fold, model_iter)

                save_ori_tensor(train_set, self.max_iter, train_mode, self.model_name, self.course, self.lesson_id,
                                self.learning_state, self.prune_slice_number, fold, model_iter)

                test_mode = "origin_test"
                save_indices(test_index, self.max_iter, test_mode, self.model_name, self.course, self.lesson_id,
                             self.learning_state, self.prune_slice_number, fold, model_iter)

                save_ori_tensor(test_set, self.max_iter, test_mode, self.model_name, self.course, self.lesson_id,
                                self.learning_state, self.prune_slice_number, fold, model_iter)

                train_set_tensor_ori = tf.convert_to_tensor(train_set)
                test_set_tensor_ori = tf.convert_to_tensor(test_set)

                input_shape = [train_set_tensor_ori.shape[1], train_set_tensor_ori.shape[2], train_set_tensor_ori.shape[3]]

                mask_train_x = 1 - np.isnan(train_set_tensor_ori)

                # Embedding Issue
                # Set the placeholder
                train_set_tensor = tf.where(tf.math.is_nan(train_set_tensor_ori), tf.zeros_like(train_set_tensor_ori), train_set_tensor_ori)
                train_set_tensor = tf.cast(train_set_tensor, tf.float32)  # X

                test_set_tensor = tf.where(tf.math.is_nan(test_set_tensor_ori), tf.zeros_like(test_set_tensor_ori), test_set_tensor_ori)
                test_set_tensor = tf.cast(test_set_tensor, tf.float32)

                train_dataset = tf.data.Dataset.from_tensor_slices(train_set_tensor).batch(self.batch_size)
                test_dataset = tf.data.Dataset.from_tensor_slices(test_set_tensor).batch(self.batch_size)

                if self.GAIN_mode == 'CNN':
                    self.Version = 0

                    generator = GAIN_Gennerator_CNN(channel=train_set_tensor.shape[-1], output_size=self.output_size, input_shape=input_shape)
                    discriminator = GAIN_Discriminator_CNN()

                    g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)
                    d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_2=0.5)

                    # Initialize performance storage
                    train_perf = []
                    test_perf = []

                    # Early stopping parameters
                    patience = 5
                    epochs_since_last_improvement = 0
                    best_rmse = 0.1
                    best_weights = None
                    improvement_flag = False  # Flag to indicate if an improvement was mad

                    for epoch in range(self.max_iter):
                        print("epoch is {}".format(epoch))

                        train_generated_data=[]

                        ori_train_tensor=tf.cast(tf.squeeze(train_set_tensor_ori,axis=1), tf.float32)
                        ori_test_tensor=tf.cast(tf.squeeze(test_set_tensor_ori,axis=1), tf.float32)

                        for train_batch in train_dataset:
                            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                                train_loss, train_generate = self.train_step(train_batch, generator, discriminator, g_optimizer, d_optimizer)
                                train_generated_data.append(train_generate['G_sample'])

                        full_generated_train_data = tf.concat(train_generated_data, axis=0)
                        full_generated_train_data = tf.cast(tf.squeeze(full_generated_train_data, axis=1), tf.float32)

                        train_mae = self.compute_mae(ori_train_tensor, full_generated_train_data)
                        train_rmse = self.compute_rmse(ori_train_tensor, full_generated_train_data)
                        train_rse = self.compute_rse(ori_train_tensor, full_generated_train_data)
                        train_auc = self.compute_auc(ori_train_tensor, full_generated_train_data)
                        train_cross_entropy = self.compute_cross_entropy(ori_train_tensor, full_generated_train_data)

                        print("mae: ", train_mae, "rmse: ", train_rmse, "rse: ", train_rse, "auc: ", train_auc, "cross_entropy: ", train_cross_entropy)

                        # Early Stopping

                        if train_rmse < best_rmse:
                            best_rmse = train_rmse
                            best_weights = generator.get_weights()  # Save best model weights if needed
                            improvement_flag = True  # Mark that an improvement has occurred

                        if improvement_flag:
                            # If an improvement has happened before, start counting epochs
                            epochs_since_last_improvement += 1
                            print(f"No improvement at epoch {epoch}, epochs since last improvement: {epochs_since_last_improvement}")

                        print("epochs_since_last_improvement is {}".format(epochs_since_last_improvement))

                        if epochs_since_last_improvement >= patience:
                            print(f"Stopping early due to no improvement after {patience} additional epochs since last improvement.")
                            break

                        train_perf.append([train_mae, train_rmse, train_rse, train_auc, train_cross_entropy])

                    mode = "train"
                    save_metrics(train_perf, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state, self.prune_slice_number, fold, model_iter)

                    mode = "train_tensor"
                    save_tensor(full_generated_train_data, self.max_iter, mode, self.model_name, self.course, self.lesson_id,
                                self.learning_state, self.prune_slice_number, fold, model_iter)

                    # Testing phase
                    test_generated_data = []
                    for test_batch in test_dataset:
                        test_generate = self.test_step(test_batch, generator)
                        test_generated_data.append(test_generate)

                    full_generated_test_data = tf.concat(test_generated_data, axis=0)
                    full_generated_test_data = tf.cast(tf.squeeze(full_generated_test_data, axis=1), tf.float32)

                    # print("ori_test_tensor is {}".format(ori_test_tensor))
                    # print("full_generated_test_data is {}".format(full_generated_test_data))

                    # Evaluate test metrics
                    test_mae = self.compute_mae(ori_test_tensor, full_generated_test_data)
                    test_rmse = self.compute_rmse(ori_test_tensor, full_generated_test_data)
                    test_rse = self.compute_rse(ori_test_tensor, full_generated_test_data)
                    test_auc = self.compute_auc(ori_test_tensor, full_generated_test_data)
                    test_cross_entropy = self.compute_cross_entropy(ori_test_tensor, full_generated_test_data)

                    print(f"Testing - MAE: {test_mae}, RMSE: {test_rmse}, RSE: {test_rse}, AUC: {test_auc}, Cross Entropy: {test_cross_entropy}")

                    test_perf.append([test_mae, test_rmse, test_rse, test_auc, test_cross_entropy])

                    mode = "test"
                    save_metrics(test_perf, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state, self.prune_slice_number, fold, model_iter)

                    mode = "test_tensor"
                    save_tensor(full_generated_test_data, self.max_iter, mode, self.model_name, self.course, self.lesson_id, self.learning_state, self.prune_slice_number, fold, model_iter)

                cv_results.append((fold, float(test_mae), float(test_rmse), float(test_rse), float(test_auc), float(test_cross_entropy), self.max_iter, self.Version))

                # End the timer
                end_time = time.time()

                # Calculate and print the elapsed time
                elapsed_time = end_time - start_time

                # Store the results in a dictionary
                summary = save_summary(cv_results, elapsed_time)

                fold_summary[fold] = elapsed_time

            return cv_results, fold_summary