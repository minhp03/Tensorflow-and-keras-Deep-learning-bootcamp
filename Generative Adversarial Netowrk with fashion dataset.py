#
# 
#  -*- coding: utf-8 -*-

#"Implement initial Generative Adversarial Network (GAN) using TensorFlow and Keras. 
# This commit includes the complete setup of both the generator and discriminator models, 
# data preprocessing with Fashion MNIST dataset, and training functions. Added visualization of generated images to assess model performance.
#  Configured optimizers and loss functions for training stability and efficiency."
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential,Model

from keras.layers import Dense,Flatten,Conv2D,Input
from keras.layers import Conv2DTranspose, Dropout, BatchNormalization,Reshape
from keras.layers import Input,LeakyReLU

ds1_minh,ds2_minh = dict(),dict()
(ds1_minh['images'],ds1_minh['labels']),(ds2_minh['images'],ds2_minh['labels']) = fashion_mnist.load_data()

ds1_minh['images'] = ds1_minh['images']/255*2-1
ds2_minh['images'] = ds2_minh['images']/255*2-1

print("Shape of ['images']:", ds1_minh['images'].shape)
print("Shape of ['images']:", ds2_minh['images'].shape)

dataset_minh = np.concatenate([ds1_minh['images'][ds1_minh['labels'] == 1],     ds2_minh['images'][ds2_minh['labels'] == 1]],axis = 0)

print(dataset_minh.shape)

plt.figure(figsize=(8, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.imshow(dataset_minh[i], cmap='gray')
    plt.axis('off')
plt.show()



train_dataset_minh = tf.data.Dataset.from_tensor_slices(dataset_minh).shuffle(7000).batch(256)

print(train_dataset_minh)

#model



#i , ii ,
generator_model_minh = Sequential()
generator_model_minh.add(Input(shape=(100,)))
generator_model_minh.add(Dense(7*7*256,activation = 'relu'))

#iii , iv , v
generator_model_minh.add(tf.keras.layers.BatchNormalization())
generator_model_minh.add(tf.keras.layers.LeakyReLU())

#vi , vii , vii
generator_model_minh.add(Reshape((7,7,256)))
generator_model_minh.add(Conv2DTranspose(128,(5,5),strides = (1,1),padding='same'))
generator_model_minh.add(tf.keras.layers.BatchNormalization())

#viii , #ix , #x , xi
generator_model_minh.add(Conv2DTranspose(64,(5,5),strides = (2,2),padding='same'))
generator_model_minh.add(tf.keras.layers.BatchNormalization())
generator_model_minh.add(tf.keras.layers.LeakyReLU())
generator_model_minh.add(Conv2DTranspose(1,(5,5),strides =(2,2),padding ='same',activation ='tanh'))

generator_model_minh.summary()

tf.keras.utils.plot_model(generator_model_minh, show_shapes=True, show_layer_names=True)

#D

vector_minh = tf.random.normal((1, 100))
generated_image_minh = generator_model_minh(vector_minh, training=False)

plt.imshow(generated_image_minh[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

#E i , ii , iii
discriminator_model_minh = Sequential()

discriminator_model_minh.add(Input(shape=(28,28,1)))
discriminator_model_minh.add(Conv2D(64, (5,5), strides=(2,2), padding='same'))
discriminator_model_minh.add(tf.keras.layers.LeakyReLU())

#iv , v ,vi
discriminator_model_minh.add(Dropout(0.3))
discriminator_model_minh.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
discriminator_model_minh.add(tf.keras.layers.LeakyReLU())

# vii , viii ,ix
discriminator_model_minh.add(Dropout(0.3))
discriminator_model_minh.add(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'))
discriminator_model_minh.add(tf.keras.layers.BatchNormalization())

#x , xi

discriminator_model_minh.add(tf.keras.layers.LeakyReLU())
discriminator_model_minh.add(Flatten())

discriminator_model_minh.add(Dense(1))

discriminator_model_minh.summary()

tf.keras.utils.plot_model(discriminator_model_minh, show_shapes=True, show_layer_names=True)

#F

cross_entropy_minh = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizier_minh = tf.keras.optimizers.Adam()

discriminator_optimizier_minh = tf.keras.optimizers.Adam()

def train_step(images):
  noise = tf.random.normal([256,100])
  with tf.GradientTape() as gen_tape, tf.GradientTape() as discri_tape:
    generated_images = generator_model_minh(noise,
                                            training = True)
    real_output = discriminator_model_minh(images,
                                           training = True)
    fake_output = discriminator_model_minh(generated_images,
                                           training = True)
    gen_loss = cross_entropy_minh(tf.ones_like(fake_output),
                                  fake_output)
    real_loss = cross_entropy_minh(tf.ones_like(real_output),
                                   real_output)
    fake_loss = cross_entropy_minh(tf.zeros_like(fake_output),
                                   fake_output)
    disc_lost = real_loss + fake_loss

    gradient_of_generator = gen_tape.gradient(gen_loss, generator_model_minh.trainable_variables)
    gradient_of_discriminator = discri_tape.gradient(disc_lost, discriminator_model_minh.trainable_variables)

    generator_optimizier_minh.apply_gradients(zip(gradient_of_generator,generator_model_minh.trainable_variables))
    discriminator_optimizier_minh.apply_gradients(zip(gradient_of_discriminator, discriminator_model_minh.trainable_variables))



#h: Visualize Trained Generator
def generate_and_plot_images(model, num_examples=16):
    noise = tf.random.normal([num_examples, 100])
    generated_images = model(noise, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show()

# Generate and plot images from the trained generator
generate_and_plot_images(generator_model_minh)