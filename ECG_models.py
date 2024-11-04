import tensorflow as tf
import ResNet
import inception_resnet_v2


def cnn(label_type, n_leads, lead_length, layers):
    """
    :param label_type: 'regr' or 'clas' - batch norm will be turned off for regression problems
    :param n_leads: number of leads in ECG
    :param lead_length: length of ECG lead
    :param layers: list of layers to apply, each list item in the form [layer_type, layer_dims] e.g. [ [Conv1D, [3,5]], ...]
        dims required for layer_dims    Conv1D: [n_kernels, kernel_size, dropout rate]
                                        Dense: [n_neurons, dropout rate]
                                        MaxPool [pool_size, stride_size]

    :return: keras model
    """
    bn = (label_type[0] == 'clas') * True + (label_type[0] == 'regr') * False  # turn off batch norm for regression
    inpt = tf.keras.layers.Input(shape=(lead_length, n_leads))
    x = inpt

    output = add_layers(x, layers, label_type, bn)

    model = tf.keras.models.Model(inputs=inpt, outputs=output)
    return model


def add_layers(x, layers, label_type, bn):
    for layer_n in range(len(layers)):
        layer = layers[layer_n]
        layer_type = layer[0]
        layer_name = f'layer_{layer_n}_{layer_type}'
        if layer_type == 'Conv1D':
            x = add_conv1D(x, layer[1], bn=bn, name=layer_name)

        elif layer_type == 'Dense':
            x = add_dense(x, layer[1], bn=bn)

        elif layer_type == 'MaxPool':
            x = tf.keras.layers.UpSampling1D(pool_size=layer[1][0], strides=layer[1][1], padding="valid")(x)

        elif layer_type == 'Flatten':
            x = tf.keras.layers.Flatten()(x)

        elif layer_type == 'Final':
            if label_type[0] == 'regr':
                x = tf.keras.layers.Dense(label_type[1], bn=False)(x)
            elif label_type[0] == 'clas':
                log_odds, x = add_softmax(x, label_type[1], bn=bn)
            elif label_type[0] == 'vae':
                pass
            else:
                print("Unrecognised output type: " + label_type)
                return
        else:
            print('Unrecognised layer type: {}'.format(layer_type))
            return

    return x


def add_conv1D(layer, kernel_dims, bn=True, name=''):
    kernels, kernel_size, dropout, padding, act_fn = kernel_dims

    x = tf.keras.layers.Conv1D(kernels, kernel_size, padding=padding, name=name)(layer)

    if act_fn is not None:
        x = tf.keras.layers.Activation(act_fn)(x)
    if bn: x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(dropout)(x)
    return x


def add_dense(layer, dims, bn=True):
    n, dropout, act_fn = dims
    y = tf.keras.layers.Dense(n)(layer)

    y = tf.keras.layers.Activation(act_fn)(y)
    if bn: y = tf.keras.layers.BatchNormalization()(y)

    y = tf.keras.layers.Dropout(dropout)(y)
    return y


def add_softmax(layer, n, bn=True):
    """ adds a Dense layer with softmax activation function and BN if bn==True
    Returns the pre-softmax values as y, and normalised values as z """
    y = tf.keras.layers.Dense(n, name='log_odds')(layer)
    if bn: y = tf.keras.layers.BatchNormalization()(y)
    z = tf.keras.layers.Activation('softmax')(y)
    return y, z


def add_transpose_layers(x, layers, bn):
    for layer in layers:
        layer_type = layer[0]
        if layer_type == 'Conv1D':
            x = add_conv1D_transpose(x, layer[1], bn=bn)

        elif layer_type == 'Dense':
            x = add_dense(x, layer[1], bn=bn)

        elif layer_type == 'MaxPool':
            x = tf.keras.layers.MaxPooling1D(pool_size=layer[1][0], strides=layer[1][1], padding="valid")(x)

        elif layer_type == 'Flatten':
            x = tf.keras.layers.Flatten()(x)

        elif layer_type == 'Final':
            pass

        else:
            print('Unrecognised layer type: {}'.format(layer_type))
            return

    return x


def add_conv1D_transpose(layer, kernel_dims, bn=True):
    kernels, kernel_size, dropout, padding, act_fn = kernel_dims
    x = tf.keras.layers.Conv1DTranspose(kernels, kernel_size, padding=padding)(layer)

    x = tf.keras.layers.Activation(act_fn)(x)
    if bn: x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(dropout)(x)
    return x


def residual_net(label_type, n_leads, lead_length, depth):
    return ResNet.resnet((lead_length, n_leads), depth[1], label_type[1], label_type[0])


def inception_v1(label_type, n_leads, ld_length, dims):
    conv_dims1, inception_dims, conv_dims2, dense, dropouts = dims
    bn = (label_type[0] == 'clas') * 'True' + (label_type[0] == 'regr') * 'False'  # turn off batch norm for regression

    # add input initial conv layers
    inpt = tf.keras.layers.Input(shape=(n_leads, ld_length))
    x = add_conv1D(inpt, conv_dims1, bn=bn)

    # add inception modules
    for i in inception_dims:
        x = add_inception(x, i)

    # add final conv, GAP, and dense layers
    x = add_conv1D(x, conv_dims2, bn=bn)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Flatten()(x)
    for i in range(len(dense)):
        x = add_dense(x, dense[i], bn=bn)

    # add final layer
    if label_type[0] == 'regr':
        output = tf.keras.layers.Dense(label_type[1])(x)
    elif label_type[0] == 'clas':
        log_odds, output = add_softmax(x, label_type[1])
    model = tf.keras.models.Model(inputs=inpt, outputs=output)
    return model


def add_inception(layer, dims, pool_size=2):
    """ adds an inception module with dims==dims and input==layer (keras layer object). Returns the output layer
    dims: 2D array of form [[n1, k1], [n2, k2], ...] where n=number of kernels, k=size of kernel """
    outputs = []
    for lyr in dims:
        lyr = tf.keras.layers.Conv1D(lyr[0], lyr[1], padding='same', activation='relu')(layer)
        lyr = tf.keras.layers.BatchNormalization()(lyr)
        outputs.append(lyr)
    # outputs.append(tf.keras.layers.MaxPool1D(pool_size)(layer))
    concat_layer = tf.keras.layers.concatenate(outputs)
    # output_layer = tf.keras.layers.MaxPool1D(pool_size=pool_size)(concat_layer)
    output_layer = concat_layer
    return output_layer


def incep_resnet_v2(label_type, n_leads, lead_length, depth):
    return inception_resnet_v2.incep_res_v2((lead_length, n_leads), label_type[1])


model_dict = {'CNN general': cnn,
              'Inception V1': inception_v1,
              'Inception ResNet v2': incep_resnet_v2,
              'ResNet': residual_net
              }
