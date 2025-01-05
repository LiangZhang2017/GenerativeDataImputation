
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

'''
Version 1:
Including the Original Tensor
'''

class AmbientGANInputPreparation(layers.Layer):
    def __init__(self, noise_dim, cont_code1_dim):
        super(AmbientGANInputPreparation, self).__init__()
        self.noise_dim = noise_dim
        self.cont_code1_dim = cont_code1_dim

    def call(self, inputs):
        original_tensor, batch_size = inputs[0], inputs[1]

        # Generate noise
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Generate continuous code 1 (e.g., initial knowledge)
        # Here, you might choose a specific range or distribution that reflects initial knowledge
        cont_code1 = tf.random.uniform(shape=(batch_size, self.cont_code1_dim), minval=0, maxval=1) # Example range [0, 1]

        # Combine the generated inputs without the original tensor
        combined_input = [noise, cont_code1]

        return combined_input

class AmbientGAN_Generator(keras.Model):
    def __init__(self, channel,output_size,input_shape):
        super(AmbientGAN_Generator, self).__init__()

        self.fc = layers.Dense(3 * 3 * 512, use_bias=False)
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

        z, y = inputs # noise + condition

        z = tf.concat([z, y], 1)

        x = self.fc(z)

        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.relu(x)

        x = self.conv1(x)
        x=self.dp1(self.bn1(x, training=training))
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x=self.dp2(self.bn2(x, training=training))
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x=self.dp3(self.bn3(x, training=training))
        x = tf.nn.relu(x)

        x = self.conv4(x)
        x=self.dp4(self.bn4(x, training=training))
        x = tf.nn.relu(x)

        x = self.conv5(x)
        # x=self.dp5(self.bn5(x, training=training))
        x = tf.nn.relu(x)

        x = self.fl(x)
        x = self.fc_output(x)
        x = self.output_layer(x)

        x_gen = tf.sigmoid(self.output_layer(x))

        return x_gen

class AmbientGAN_Standard_Discriminator(keras.Model):
    def __init__(self, cont_code1_dim):
        super(AmbientGAN_Standard_Discriminator,self).__init__()

        df_dim = 32  # Dimension of discriminator filters in first conv layer
        dfc_dim = 1024  # Dimension of discriminator units for fully connected layer

        self.conv1 = layers.Conv2D(32, 1, 2, 'valid')
        self.bn1 = layers.BatchNormalization()
        self.dp1 = layers.Dropout(0.5)
        self.leaky_relu1 = layers.LeakyReLU()

        self.conv2 = layers.Conv2D(64, 1, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.dp2 = layers.Dropout(0.5)
        self.leaky_relu2 = layers.LeakyReLU()

        self.conv3 = layers.Conv2D(128, 1, 2)
        self.bn3 = layers.BatchNormalization()
        self.dp3 = layers.Dropout(0.5)
        self.leaky_relu3 = layers.LeakyReLU()

        self.conv4 = layers.Conv2D(256, 1, 2)
        self.bn4 = layers.BatchNormalization()
        self.dp4 = layers.Dropout(0.5)
        self.leaky_relu4 = layers.LeakyReLU()

        self.conv5 = layers.Conv2D(df_dim * 4, (5, 5), strides=(2, 2), padding='same')
        self.bn5 = layers.BatchNormalization()
        self.dp5 = layers.Dropout(0.5)
        self.leaky_relu5 = layers.LeakyReLU()

        # [b,h,w,3]= [b,-1]
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(dfc_dim)
        self.leaky_relu5 = layers.LeakyReLU()

        self.fc = layers.Dense(cont_code1_dim)

    def call(self, inputs, training=None, mask=None):

        x = self.conv1(inputs)
        x = self.leaky_relu1(x)
        x = self.bn1(x)

        x=self.conv2(x)
        x=self.dp2(self.bn2(x))
        x = self.leaky_relu2(x)

        x=self.conv3(x)
        x=self.dp3(self.bn3(x))
        x=self.leaky_relu3(x)

        x=self.conv4(x)
        x=self.dp4(self.bn4(x))
        x=self.leaky_relu4(x)

        x = self.conv5(x)
        x = self.dp5(self.bn5(x))
        x = self.leaky_relu5(x)

        # [b,h,w,c]=>[b,-1]
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.leaky_relu5(x)

        d_logit = self.fc(x)
        d = tf.sigmoid(d_logit)

        return d_logit, d