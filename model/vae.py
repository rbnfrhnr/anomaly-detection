import tensorflow as tf
import numpy as mp
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):

    def __init__(self, encoder_inputs, encoder_layers, decoder_inputs, decoder_layers, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.kl_weight = 0.5 if 'kl_weight' not in kwargs else kwargs['kl_weight']

        z_mean = layers.Dense(decoder_inputs.shape[1], name="z_mean")(encoder_layers)
        z_log_var = layers.Dense(decoder_inputs.shape[1], name="z_log_var")(encoder_layers)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        self.decoder = keras.Model(decoder_inputs, decoder_layers, name="decoder")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.z_mean_tracker = keras.metrics.Mean(name='z_mean')
        self.z_log_var_tacker = keras.metrics.Mean(name='z_log_var')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=0
                )
            )
            kl_loss = -self.kl_weight * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.z_mean_tracker.update_state(z_mean)
        self.z_log_var_tacker.update_state(z_log_var)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "z_mean": self.z_mean_tracker.result(),
            "z_log_var": self.z_log_var_tacker.result()
        }
