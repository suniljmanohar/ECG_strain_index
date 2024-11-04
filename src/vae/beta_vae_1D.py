import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Layer, BatchNormalization, \
    Flatten, Dense, Reshape, Conv1DTranspose, Lambda, AveragePooling2D, \
    UpSampling1D, Concatenate, AveragePooling1D
from tensorflow.keras.callbacks import Callback
import ECG_models as models


# tf.compat.v1.disable_eager_execution()
""" stop TF using all VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
"""


class MyCallBack(Callback):
    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

    def on_epoch_begin(self, epoch, logs=None):
        """
        if epoch == 0:
            K.set_value(self.loss_weight["encoder"], 1)
            K.set_value(self.loss_weight["decoder"], 1)
        """
        pass


class VAE:
    def __init__(self,
                 input_shape,
                 layers,
                 use_bn,
                 latent_space_dim,
                 condition_dim,
                 kl_weight,
                 condition_weight):
        self.input_shape = input_shape  # [None, 100, 8]
        self.layers = layers    # [['Conv1D', [40, 5, 0]], ...]
        self.bn = use_bn
        self.latent_space_dim = latent_space_dim
        self.condition_dim = condition_dim
        self.kl_weight = kl_weight
        self.condition_weight = condition_weight
        self.reconstruction_loss_weight = 1.

        self.encoder = None
        self.decoder = None
        self.classifier = None
        self.model = None

        self._n_layers = len(layers)
        self._shape_before_bottleneck = None
        self._model_input = None
        self._model_condition = None

        self.loss_weight = {
            "encoder": K.variable(self.kl_weight),
            "decoder": K.variable(self.reconstruction_loss_weight)
        }

        self.trainingHistory = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        # optimizer = Adam(learning_rate=learning_rate)
        optimizer = tf.keras.optimizers.legacy.Adadelta()
        self.model.compile(optimizer=optimizer,
                           loss={
                               "encoder": self._calculate_kl_loss,
                               "decoder": self._calculate_reconstruction_loss
                           },
                           loss_weights=self.loss_weight,
                           )

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        return kl_loss

    def _calculate_classification_loss(self, E, risk):
        hazard_ratio = K.exp(risk)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -K.sum(censored_likelihood)
        return neg_likelihood

    def train(self, x_tr, c_tr, x_val, c_val, batch_size, num_epochs):
        history = self.model.fit([x_tr, c_tr],
                                 {
                                     "encoder": x_tr,
                                     "decoder": x_tr
                                 },
                                 batch_size=batch_size,
                                 epochs=num_epochs,
                                 validation_data=([x_val, c_val], [x_val, x_val]),
                                 callbacks=[MyCallBack(self.loss_weight)],
                                 shuffle=False)
        self.trainingHistory = history

    def save(self, save_folder="."):
        self._create_save_folder_if_it_doesnt_exsit(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
        self._save_training_history(save_folder)

    def _create_save_folder_if_it_doesnt_exsit(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.layers,
            self.bn,
            self.latent_space_dim,
            self.condition_dim,
            self.kl_weight,
            self.condition_weight
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _save_training_history(self, save_folder):
        save_path = os.path.join(save_folder, "training history.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.trainingHistory.history, f)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        history_path = os.path.join(save_folder, "training history.pkl")
        with open(history_path, "rb") as f:
            history = pickle.load(f)
        vae = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        vae.load_weights(weights_path)
        vae.trainingHistory = history
        return vae

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        # self._build_classifier()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        condition = self._add_condition_input()
        conv_layers = self._add_layers(encoder_input)
        mu, logvar = self._add_bottleneck(conv_layers, condition)
        bottleneck = self._add_sampling()
        self._model_input = encoder_input
        self._model_condition = condition
        self.encoder = Model([encoder_input, condition], [bottleneck, mu, logvar], name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_condition_input(self):
        return Input(shape=self.condition_dim, name="condition_input")

    def _add_layers(self, encoder_input):
        x = encoder_input
        output = models.add_layers(x, self.layers, ['vae', None], self.bn)
        return output

    def _add_bottleneck(self, x, condition):
        """Mean Sampling"""
        x = AveragePooling1D(pool_size=2)(x)

        """Flatten data and add bottleneck with Gaussian Sampling(Dense Layer). """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)

        condition = tf.keras.layers.Lambda(lambda x: x * self.condition_weight)(condition)
        x = Concatenate(axis=-1)([x, condition])
        x = BatchNormalization()(x)

        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        return self.mu, self.log_variance

    def _add_sampling(self):
        class Sampling(Layer):
            """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Sampling()([self.mu, self.log_variance])
        return z


    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        condition = self._add_condition_input()
        con_decoder_input = self._add_con_decoder_input(decoder_input, condition)
        dense_layer = self._add_dense_layer(con_decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model([decoder_input, condition], decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_con_decoder_input(self, x, condition):
        condition = tf.keras.layers.Lambda(lambda x: x * self.condition_weight)(condition)
        return Concatenate(axis=-1)([x, condition])

    def _add_dense_layer(self, decoder_input):
        dense_layer = BatchNormalization()(decoder_input)
        num_neurons = np.prod(self._shape_before_bottleneck)  # [44,200] -> 44*200
        dense_layer = Dense(num_neurons, name="decoder_dense")(dense_layer)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        reshape_layer = UpSampling1D(size=2)(reshape_layer)
        return reshape_layer

    def _add_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the first layer
        x = models.add_transpose_layers(x, reversed(self.layers[1:]), self.bn)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv1DTranspose(
            filters=self.input_shape[-1],
            kernel_size=self.layers[0][1][1],
            padding='valid',
        )
        x = conv_transpose_layer(x)
        x = tf.keras.layers.Activation('linear')(x)
        # x = BatchNormalization()(x)
        return x

    def _build_autoencoder(self):
        model_input = self._model_input
        condition = self._model_condition
        model_encoder_output = self.encoder([model_input, condition])
        model_decoder_output = self.decoder([model_encoder_output[0], condition])
        self.model = Model([model_input, condition], [model_encoder_output, model_decoder_output],
                           name="VAE")
