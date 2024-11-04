import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns

import params as p, data_loader as dl, my_tools as mt
import trainer
from beta_vae_1D import VAE
from vae_params import VAE_params


tf.compat.v1.disable_eager_execution()


def main():
    vp = VAE_params()
    vae = load_and_train(vp)


def get_data(train_prop):
    hps = p.HyperParams(p.subsets['all'])
    hps.train_prop = train_prop
    grp_data = dl.get_all_data(hps, do_preprocess=True)

    return grp_data


def load_and_train(vp):
    """ loads training and validation data and trains the VAE, saves model to global MODEL_DIR """
    X_tr, X_te, Con_tr, Con_te = get_data(vp.train_prop)[:4]
    # X_tr = X_tr.reshape(X_tr.shape + (1,)) # to align with Marcel's input shape for Conv2D
    # X_te = X_te.reshape(X_te.shape + (1,))

    vae = train(vp, X_tr, Con_tr, X_te, Con_te, vp.learning_rate,
                vp.batch_size, vp.epochs)
    mt.make_dir(vp.model_dir)
    vae.save(vp.model_dir)
    print('Model saved')
    return vae


def train(vp, x_train, c_train, x_val, c_val, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=vp.input_shape,
        layers=vp.model_shape,
        use_bn=True,
        latent_space_dim=vp.latent_dims,
        condition_dim=2,                # one-hot binary categorical encoding
        kl_weight=vp.beta,  # weight applied to KL divergence (total loss = kl_weight * KL divergence + reconstruction loss)
        condition_weight=vp.condition_weight   # use zero for normal VAE without condition
    )

    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, c_train, x_val, c_val, batch_size, epochs)

    return vae


def convert(input_folder, output_file, frame_rate):
    os.system("ffmpeg -r {} -i {}ECG %01d.svg -vcodec mpeg4 -y {}.mp4".format(frame_rate, input_folder, output_file))


if __name__ == "__main__":
    main()


