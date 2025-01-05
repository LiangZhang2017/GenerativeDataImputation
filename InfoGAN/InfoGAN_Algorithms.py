
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


class InfoGAN_Q_Network(keras.Model):
    def __init__(self, cont_code1_dim, cont_code2_dim):
        super(InfoGAN_Q_Network, self).__init__()
        self.fc = keras.layers.Dense(cont_code1_dim + cont_code2_dim)

    def call(self, inputs, training=None):
        return self.fc(inputs)


'''
Version 1:
Including the Original Tensor
'''
class InfoGANInputPreparation(layers.Layer):
    def __init__(self, noise_dim, cont_code1_dim, cont_code2_dim):
        super(InfoGANInputPreparation, self).__init__()
        self.noise_dim = noise_dim
        self.cont_code1_dim = cont_code1_dim
        self.cont_code2_dim = cont_code2_dim

    def call(self, inputs):
        original_tensor, batch_size = inputs[0], inputs[1]

        # Generate noise
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Generate continuous code 1 (e.g., initial knowledge)
        # Here, you might choose a specific range or distribution that reflects initial knowledge
        cont_code1 = tf.random.uniform(shape=(batch_size, self.cont_code1_dim), minval=0, maxval=1) # Example range [0, 1]

        # Generate continuous code 2 (e.g., learning rate)
        # This code might have a different range or distribution if that better reflects learning rates
        cont_code2 = tf.random.uniform(shape=(batch_size, self.cont_code2_dim), minval=-1, maxval=1) # Example range [-1, 1]

        # Combine the generated inputs without the original tensor
        combined_input = [noise, cont_code1, cont_code2]

        return combined_input

class InfoGAN_Standard_Generator(keras.Model):
    def __init__(self,channel,output_size,input_shape):
        super(InfoGAN_Standard_Generator,self).__init__()

        self.fc=layers.Dense(3*3*512,use_bias=False)
        self.conv1 = layers.Conv2DTranspose(256, 3, 2, 'valid')
        self.bn1 = layers.BatchNormalization()
        self.dp1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.dp2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2DTranspose(64, 4, 2, 'valid')
        self.bn3 = layers.BatchNormalization()
        self.dp3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv2DTranspose(32, 4, 2, 'valid')
        self.bn4 = layers.BatchNormalization()
        self.dp4 = layers.Dropout(0.5)

        self.conv5 = layers.Conv2DTranspose(channel, 4, 2, 'valid')
        self.bn5 = layers.BatchNormalization()
        self.dp5 = layers.Dropout(0.5)

        self.fl = layers.Flatten()
        self.fc_output = layers.Dense(output_size)
        self.output_layer = layers.Reshape(target_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        noise, cont_code1, cont_code2 = inputs #

        x = tf.concat([noise, cont_code1, cont_code2], axis=1)

        x = self.fc(x)

        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.relu(x)

        x=self.conv1(x)
        x=self.dp1(self.bn1(x, training=training))
        x = tf.nn.relu(x)

        x=self.conv2(x)
        x=self.dp2(self.bn2(x, training=training))
        x = tf.nn.relu(x)

        x=self.conv3(x)
        x=self.dp3(self.bn3(x, training=training))
        x = tf.nn.relu(x)

        x=self.conv4(x)
        x=self.dp4(self.bn4(x, training=training))
        x = tf.nn.relu(x)

        x=self.conv5(x)
        # x=self.dp5(self.bn5(x, training=training))
        x = tf.nn.relu(x)

        x = self.fl(x)
        x = self.fc_output(x)
        x = self.output_layer(x)

        imputed_tensor = tf.sigmoid(x)

        return imputed_tensor

class InfoGAN_Standard_Discriminator(keras.Model):
    def __init__(self,cont_code1_dim, cont_code2_dim):
        super(InfoGAN_Standard_Discriminator,self).__init__()

        self.cont_code1_dim=cont_code1_dim
        self.cont_code2_dim=cont_code2_dim

        self.conv1 = layers.Conv2D(32, 1, 2, 'valid')
        self.leaky_relu_1 = layers.LeakyReLU()

        self.conv2 = layers.Conv2D(64, 1, 2, 'valid')
        self.leaky_relu_2 = layers.LeakyReLU()
        self.bn2 = layers.BatchNormalization()
        self.dp2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2D(128, 1, 2)
        self.leaky_relu_3 = layers.LeakyReLU()
        self.bn3 = layers.BatchNormalization()
        self.dp3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv2D(256, 1, 2)
        self.leaky_relu_4 = layers.LeakyReLU()
        self.bn4 = layers.BatchNormalization()
        self.dp4 = layers.Dropout(0.5)

        self.conv5 = layers.Conv2D(512, 1, 2)
        self.leaky_relu_5 = layers.LeakyReLU()
        self.bn5 = layers.BatchNormalization()
        self.dp5 = layers.Dropout(0.5)

        # [b,h,w,3]= [b,-1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

        # Output layer for continuous codes, combining dimensions of both codes
        self.fc_cont_codes = layers.Dense(cont_code1_dim + cont_code2_dim)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)

        x=self.conv2(x)
        x=self.dp2(self.bn2(x, training=training))

        x=self.conv3(x)
        x=self.dp3(self.bn3(x, training=training))

        x=self.conv4(x)
        x=self.dp4(self.bn4(x, training=training))

        x=self.conv5(x)
        x=self.dp5(self.bn5(x, training=training))

        # [b,h,w,c]=>[b,-1]
        x = self.flatten(x)
        logits = tf.sigmoid(self.fc(x))

        # Continuous codes combined prediction output
        cont_codes_combined = self.fc_cont_codes(x)

        # Split the combined continuous codes into cont_code1 and cont_code2 based on their dimensions
        cont_code1, cont_code2 = tf.split(cont_codes_combined, [self.cont_code1_dim, self.cont_code2_dim], axis=1)

        return logits, cont_code1, cont_code2

class InfoGANInputPreparationWithTensor(layers.Layer):
    def __init__(self, noise_dim, cont_code1_dim, cont_code2_dim):
        super(InfoGANInputPreparationWithTensor, self).__init__()
        self.noise_dim = noise_dim
        self.cont_code1_dim = cont_code1_dim
        self.cont_code2_dim = cont_code2_dim

    def call(self, inputs):
        original_tensor, batch_size = inputs[0], inputs[1]

        # Generate noise
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Generate continuous code 1 (e.g., initial knowledge)
        # Here, you might choose a specific range or distribution that reflects initial knowledge
        cont_code1 = tf.random.uniform(shape=(batch_size, self.cont_code1_dim), minval=0, maxval=1) # Example range [0, 1]

        # Generate continuous code 2 (e.g., learning rate)
        # This code might have a different range or distribution if that better reflects learning rates
        cont_code2 = tf.random.uniform(shape=(batch_size, self.cont_code2_dim), minval=-1, maxval=1) # Example range [-1, 1]

        # Combine the generated inputs without the original tensor
        combined_input = [original_tensor, noise, cont_code1, cont_code2]

        return combined_input

class InfoGAN_Standard_Generator_WithTensor(keras.Model):
    def __init__(self,channel,output_size,input_shape):
        super(InfoGAN_Standard_Generator_WithTensor,self).__init__()

        # Assume original_tensor_shape is the shape of the flattened original tensor
        self.preprocess_original = layers.Flatten()  # Example preprocessing layer for the original tensor

        # All layers

        self.fc=layers.Dense(3*3*512,use_bias=False)
        self.conv1 = layers.Conv2DTranspose(256, 3, 2, 'valid')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(64, 4, 2, 'valid')
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2DTranspose(32, 4, 2, 'valid')
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2DTranspose(channel, 4, 2, 'valid')
        self.bn5 = layers.BatchNormalization()

        self.fl = layers.Flatten()
        self.fc_output = layers.Dense(output_size)
        self.output_layer = layers.Reshape(target_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        original_tensor, noise, cont_code1, cont_code2 = inputs #

        # # Preprocess original tensor
        processed_original = self.preprocess_original(original_tensor)

        print("processed_original shape is {}".format(processed_original.shape))

        print("noise shape is {}".format(noise.shape))

        x = tf.concat([processed_original, noise, cont_code1, cont_code2], axis=1)

        x = self.fc(x)

        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.relu(x)

        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        x = tf.nn.relu(self.bn5(self.conv5(x), training=training))

        x = self.fl(x)
        x = self.fc_output(x)
        x = self.output_layer(x)

        imputed_tensor = tf.sigmoid(x)

        return imputed_tensor

class InfoGAN_Standard_Discriminator_WithTensor(keras.Model):
    def __init__(self,cont_code1_dim, cont_code2_dim):
        super(InfoGAN_Standard_Discriminator_WithTensor,self).__init__()

        self.cont_code1_dim=cont_code1_dim
        self.cont_code2_dim=cont_code2_dim

        self.conv1 = layers.Conv2D(32, 1, 2, 'valid')
        self.leaky_relu_1 = layers.LeakyReLU()

        self.conv2 = layers.Conv2D(64, 1, 2, 'valid')
        self.leaky_relu_2 = layers.LeakyReLU()
        self.bn2 = layers.BatchNormalization()
        self.dp2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2D(128, 1, 2)
        self.leaky_relu_3 = layers.LeakyReLU()
        self.bn3 = layers.BatchNormalization()
        self.dp3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv2D(256, 1, 2)
        self.leaky_relu_4 = layers.LeakyReLU()
        self.bn4 = layers.BatchNormalization()
        self.dp4 = layers.Dropout(0.5)

        self.conv5 = layers.Conv2D(512, 1, 2)
        self.leaky_relu_5 = layers.LeakyReLU()
        self.bn5 = layers.BatchNormalization()
        self.dp5 = layers.Dropout(0.5)

        # [b,h,w,3]= [b,-1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

        # Output layer for continuous codes, combining dimensions of both codes
        self.fc_cont_codes = layers.Dense(cont_code1_dim + cont_code2_dim)

    def call(self, inputs, training=None, mask=None):

        x = self.conv1(inputs)

        x=self.conv2(x)
        # x=self.bn2(x, training=training)
        # x = self.dp2(x)

        x=self.conv3(x)
        # x=self.bn3(x, training=training)
        # x = self.dp3(x)

        x=self.conv4(x)
        # x=self.bn4(x, training=training)
        # x = self.dp4(x)

        x=self.conv5(x)
        # x=self.bn5(x, training=training)
        # x = self.dp5(x)

        # [b,h,w,c]=>[b,-1]
        x = self.flatten(x)
        logits = tf.sigmoid(self.fc(x))

        # Continuous codes combined prediction output
        cont_codes_combined = self.fc_cont_codes(x)

        # Split the combined continuous codes into cont_code1 and cont_code2 based on their dimensions
        cont_code1, cont_code2 = tf.split(cont_codes_combined, [self.cont_code1_dim, self.cont_code2_dim], axis=1)

        return logits, cont_code1, cont_code2