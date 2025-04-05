# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:55:18 2024

@author: Lin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:36:48 2024
@author: IC
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(X_train, _), (X_test, _) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Input shape
input_shape = X_train.shape[1:]  # (32, 32, 3)

# Dimension of latent space
latent_dim = 64

# Encoder
encoder_input = Input(shape=input_shape, name="encoder_input")  # Input shape: (32, 32, 3)
x = Flatten()(encoder_input)  # Flatten input for Dense layers
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# Latent space (mean and log variance)
z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

# Decoder
decoder_input = Input(shape=(latent_dim,), name="decoder_input")
x = Dense(256, activation='relu')(decoder_input)
x = Dense(512, activation='relu')(x)
x = Dense(np.prod(input_shape), activation='sigmoid')(x)  # Output size: 32*32*3 = 3072
decoded_output = Reshape(input_shape)(x)  # Reshape back to (32, 32, 3)

# Define encoder and decoder models
encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
decoder = Model(decoder_input, decoded_output, name="decoder")

# Variational Autoencoder (VAE) Model
vae_output = decoder(encoder(encoder_input)[2])  # Pass the sampled latent vector through decoder
vae = Model(encoder_input, vae_output, name="vae")

# Custom VAE Loss Layer
class VAELossLayer(Layer):
    def call(self, inputs):
        encoder_input, vae_output, z_mean, z_log_var = inputs
        # Flatten inputs for binary_crossentropy
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                Flatten()(encoder_input), Flatten()(vae_output)
            )
        ) * np.prod(input_shape)
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        
        # Total loss
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)
        return vae_output

# Add loss layer
vae_output_with_loss = VAELossLayer()([encoder_input, vae_output, z_mean, z_log_var])

# Re-define VAE model with the loss layer
vae = Model(encoder_input, vae_output_with_loss, name="vae_with_loss")

# Compile the model
optimizer = Adam(learning_rate=0.001)
vae.compile(optimizer=optimizer)

# Train the VAE model
epochs = 50
batch_size = 128
vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None))

# Reconstruct and visualize images
decoded_images = vae.predict(X_test)

# Display original and reconstructed images
import matplotlib.pyplot as plt

n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i])
    plt.title('Original')
    plt.axis('off')

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i])
    plt.title('Reconstructed')
    plt.axis('off')

plt.show()
