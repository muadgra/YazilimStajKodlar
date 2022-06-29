# -*- coding: utf-8 -*-
"""
@author: Mertcan Gökmen
"""

import os
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#Set global variables
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNEL = 3
w_init = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)


def load_image(images_path):
    image = tf.io.read_file(images_path)
    image = tf.io.decode_png(image)
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = tf.cast(image, tf.float32)
    #normalize the image between -1 and +1
    image = (image - 127.5) / 127.5
    return image

def tf_dataset(images_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
  
#used in generator
def deconv_block(inputs, num_filters, kernel_size, strides, bn = True):
    x = Conv2DTranspose(filters = num_filters,
                        kernel_size = kernel_size,
                        kernel_initializer = w_init,
                        padding = "same",
                        strides = strides,
                        use_bias=False)(inputs)
    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.2)(x)
    return x
    
def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
    return x

def build_generator(latent_dim):
    f = [2 ** i for i in range(5)] [::-1]
    filters = 32
    output_strides = 16
    h_output = IMG_HEIGHT // output_strides
    w_output = IMG_WIDTH // output_strides
    
    noise = Input(shape = (latent_dim,), name = "gen_noise_input")
    
    x = Dense(f[0] * filters * h_output * w_output, use_bias = False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Reshape((h_output, w_output, f[0] * filters))(x)
    print(x.shape)
    
    #start from 1, since we start with a dense already
    for i in range(1, 5):
        x = deconv_block(x, 
                         num_filters = f[i] * filters, 
                         kernel_size = 5, 
                         strides = 2,
                         bn = True)
    #Since it is an RGB image, filter number should be 3
    x = conv_block(x,
                   num_filters = 3,
                   kernel_size = 5,
                   strides = 1,
                   activation = False)
    fake_output = Activation("tanh")(x)
    
    return Model(noise, fake_output, name = "generator")

def build_discriminator():
    f = [2 ** i for i in range(4)]
    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
    x = image_input
    filters = 64
    output_strides = 16
    h_output = IMG_HEIGHT // output_strides
    w_output = IMG_WIDTH // output_strides
    
    
    
    for i in range(0, 4):
        x = conv_block(x,
                       num_filters = f[i] * filters,
                       kernel_size = 5,
                       strides = 2)
        x = Flatten()(x)
        x = Dense(1)(x)
        
        return Model(image_input, x, name = "discriminator")

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        
    def compile(self, discriminator_optimizer, generator_optimizer, loss_function):
        super(GAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss_function = loss_function
        
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        #train discriminator twice
        for _ in range(2):
            #on fake images first
            random_latent_vectors = tf.random.normal(shape = (batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))
            
            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                discriminator_loss_1 = self.loss_function(generated_labels, predictions)
            grads = ftape.gradient(discriminator_loss_1, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            labels = tf.ones((batch_size, 1))
            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                discriminator_loss_2 = self.loss_function(labels, predictions)
            grads = rtape.gradient(discriminator_loss_2, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            
        #train generator
        random_latent_vectors = tf.random.normal(shape = (batch_size, self.latent_dim))
        false_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as gtape:
            predictions = self.discriminator(random_latent_vectors)
            generator_loss = self.loss_function(false_labels, predictions)
        grads = gtape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return {"discriminator_loss_1": discriminator_loss_1, "generator_loss": generator_loss}
        
if __name__ == "__main__":
    #Hyperparameters
    #since image size is small, batch size is high
    batch_size = 128
    latent_dim = 128
    epochs = 100
    
    #Dataset
    images_path = glob("C:/Users/Mertcan/Desktop/mydata/data/*")
    dataset = tf_dataset(images_path, batch_size)
    #dataset instance's size contains batchsize, height, width, channels
    discriminator_model = build_discriminator()
    
    generator_model = build_generator(latent_dim)
    discriminator_model.summary()
    generator_model.summary()
    
    gan = GAN(discriminator_model, generator_model, latent_dim)
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True, label_smoothing=0.1)
    discriminator_optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5)        
    generator_optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5)
    gan.compile(discriminator_optimizer, generator_optimizer, bce_loss_fn)
    images_dataset = tf_dataset(images_path, batch_size)
    
    for epoch in range(epochs):
        gan.fit(images_dataset, epochs = 1)
        generator_model.save("saved_model/g_model.h5")
        discriminator_model.save("saved_model/d_model.h5")