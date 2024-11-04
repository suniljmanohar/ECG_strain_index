import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as kb
from scipy.stats import gaussian_kde
import time
import ECG_models, params as p, my_tools, data_loader as dl, ECG_plots as eplt
import my_timer
from eval_results import eval_clas, eval_regr


def main():
    i_range = range(5,6)
    hps = p.HyperParams(['all', ['combined', ['hypertension'], {}]])
    # hps = p.HyperParams(p.subsets['hcmr lv wall'])
    # hps = p.HyperParams(['hcmr', ['Echo ejection fraction (%)'], {}])  # testing regression
    return run_exp(hps, i_range)


def run_exp(hps, i_range, box_plot=True):
    timer = my_timer.Timer(start_now=True)
    all_accs, all_times, all_metrics = [], [], []

    # get ECGs, labels, and ID codes
    ecgs, labels = dl.get_all_data(hps)

    # run experiment for each parameter i
    for i in i_range:
        all_accs.append([])
        timer.start()
        hps.set_dims(i)

        # repeat experiment
        for k in range(hps.repeats):
            print('Parameter i =', i, '; Run', k + 1, 'of', hps.repeats)
            # shuffle, split and augment data
            x_train, x_test, y_train, y_test, ids_train, ids_test, hps = dl.preprocess(ecgs, labels, hps)

            # train the network
            expt_details = [my_tools.timestamp(), hps.describe(), str(k + 1)]
            test_acc, ids_test, y_pred, y_test, _, logits = train_net(x_train, x_test, y_train, y_test, ids_train,
                                                                      ids_test, hps, debug=False)
            all_accs[-1].append(test_acc)
            if hps.label_type == 'clas':
                all_metrics.append(eval_clas(ids_test, y_pred, y_test,
                                             show_ims=hps.show_ims, save_to=hps.folder + str(expt_details)))
            else:
                all_metrics = eval_regr(ids_test, y_pred, y_test,
                                        show_ims=hps.show_ims, save_to=hps.folder + str(expt_details))

            all_times.append(timer.elapsed('m'))

            #  update log with tab separated values
            log_entry = '\t'.join(expt_details + [str(round(test_acc, 3))])
            with open(hps.folder + 'results log.txt', 'a') as f:
                f.write(log_entry + '\n')

    # plots box plots to compare multiple repeats across all values of i
    if box_plot:
        boxplot_results(hps, i_range, all_accs, hps.folder + str(expt_details) + ' (boxplot).svg')
    if hps.label_type == 'clas':
        print('Average optimum threshold, average peak accuracy:', optimise_accs([m[1]['acc'] for m in all_metrics]))
    tabulate_results(i_range, all_times, all_accs)
    print('\nTotal time taken: ', timer.elapsed('m'))
    return all_accs


def build_cnn_model(hps, label_data=None):
    """ Builds a keras model using the hyperparameters in hps object, and using the integer i to generate layer
    dimensions from above fns. hps must have values label_type ('clas' or 'regr'), label_cols, and cnn_structure
     Returns the keras model """

    if hps.label_type == 'clas':
        n_classes = label_data.shape[1]
        loss_fn = (n_classes==1) * 'binary_crossentropy' + (n_classes!=1) * 'categorical_crossentropy'
        cnn_model = ECG_models.model_dict[hps.cnn_structure](['clas', n_classes], hps.cnn_n_leads, hps.lead_length, hps.dims)
        cnn_model.compile(loss=loss_fn, optimizer='adadelta', metrics=['accuracy'])
    elif hps.label_type == 'regr':
        n_vars = len(hps.datasets[0][1])  # number of dependent variables
        cnn_model = ECG_models.model_dict[hps.cnn_structure](['regr', n_vars], hps.cnn_n_leads, hps.lead_length, hps.dims)
        cnn_model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['mean_absolute_error'])
    else:
        raise ValueError("Unrecognised model type: " + hps.label_type)
    return cnn_model


def train_net(x_train, x_test, y_train, y_test, ids_train, ids_test, hps,
              save_model_to='', debug=False):
    """ Loads data from the specified file, builds, compiles, trains and evaluates the model on the data.
    x_train, x_test = np array of shape (n, n_leads, lead_length) for n-lead ECGs
    y_train, y_test = np arrays of labels ordered corresponding to ecgs
    ids_train, ids_test = list of ECG ID numbers for train and test sets
    hps = hyperparameter object
    i = any integer used to determine size of model layers based on functions above
    Returns [train accuracy, test accuracy] for classifiers, and [mean absolute error, correlation coeff] for regression
    """
    tf.autograph.set_verbosity(2)
    timer = my_timer.Timer(start_now=True)

    # Build model
    net_model = build_cnn_model(hps, y_train)

    # Create callbacks
    cb_list = create_callbacks(hps)

    # debugger
    if debug: debugger(hps, ids_train, x_train, y_train)

    # Train model
    net_model, history, train_acc, test_acc = train_model(net_model, hps, x_train, x_test,
                                                          y_train, y_test, cb_list)
    print('Training took {} ({:.1f}s per epoch)'.format(timer.elapsed('m'), timer.elapsed('s_float')/hps.epochs))

    # Save model if good enough
    if save_model_to != '':
        net_model.save(save_model_to)
    else:
        if hps.label_type == 'clas' and test_acc > hps.save_threshold:
            net_model.save(hps.folder + '_acc {:.3f}.h5'.format(test_acc))
            print('Model saved (test accuracy = {})'.format(test_acc))

    # make predictions on validation set and plot results
    validn = len(y_test) > 0
    if hps.label_type == 'clas':
        acc_fn = 'accuracy'
        if validn:
            logits, y_pred = pred_logits_probs(net_model, x_test)
            y_test = y_test[:, 1]
            histplot_results(logits, y_test, 'logits', hps.data_title, show_ims=hps.show_ims,
                             save_to=hps.save_ims * hps.folder)
            histplot_results(y_pred, y_test, 'model output', hps.data_title, show_ims=hps.show_ims,
                             save_to=hps.save_ims * hps.folder)
            density_1d([[y_pred[i] for i in range(len(y_pred)) if y_test[i]==n] for n in range(len(np.unique(y_test)))],
                       ['ground truth = {}'.format(int(i)) for i in np.unique(y_test)],
                       hps.data_title, show_ims=hps.show_ims, save_to=hps.save_ims * hps.folder)
        else:
            logits, y_pred = [], []
    else:
        acc_fn = 'mean_absolute_error'
        logits = []
        if validn:
            y_pred = net_model.predict(x_test)
        else:
            y_pred = []

    plot_train_history(history.history, [acc_fn, 'loss'], show_ims=hps.show_ims, val=validn,
                       save_to=hps.save_ims * hps.folder)

    # Return test set IDs, labels, and prediction
    kb.clear_session()
    return test_acc, ids_test, y_pred, y_test, net_model, logits


def pred_logits_probs(net_model, x_test):
    log_odds_model = tf.keras.models.Model(inputs=net_model.input,
                                           outputs=[net_model.get_layer('log_odds').output, net_model.output])
    scores = log_odds_model.predict(x_test)
    logits, y_pred = scores[0], scores[1]
    return logits[:,1] - logits[:,0], y_pred[:,1]


def debugger(hps, ids_train, x_train, y_train):
    """ to see exactly what is being fed into the network right before training (plots ECG data and prints label """
    # x_train = x_train.reshape((x_train.shape[0], hps.cnn_n_leads, x_train.shape[1] // hps.cnn_n_leads))  #?not needed
    for n in range(len(x_train)):
        print(ids_train[n], y_train[n])
        md = {'filename': ids_train[n], 'lead order': hps.cnn_lead_order,
              'r waves': [[] for j in range(hps.cnn_n_leads)]}
        eplt.plot_iso_ecg_vert(x_train[n], md, display=True)


def train_model(net_model, hps, x_train, x_test, y_train, y_test, cb_list):
    print('Fitting model with:\n{} train samples and {} test samples'.format(x_train.shape[0], x_test.shape[0]))
    history = net_model.fit(x_train, y_train, batch_size=hps.batch_size, epochs=hps.epochs, verbose=hps.verbose,
                            validation_data=(x_test, y_test), callbacks=cb_list)
    """ for dynamic loading of ECG data - needs individual ecg npy files in correct folder              #TODO
    train_gen = dl.ECG_gen(ids_train, y_train, hps)
    test_gen  = dl.ECG_gen(ids_test,  y_test,  hps)
    history = net_model.fit_generator(generator=train_gen, epochs=hps.epochs, verbose=0, validation_data=test_gen,
                                      callbacks=cb_list, steps_per_epoch=len(x_train)//hps.batch_size,
                                      validation_steps=len(x_test)//hps.batch_size)
    """
    if hps.label_type == 'clas':
        train_acc = history.history['accuracy'][-1]
        if 'val_accuracy' in history.history.keys():
            test_acc = history.history['val_accuracy'][-1]
        else:
            test_acc = 0
    else:
        train_acc = history.history['mean_absolute_error'][-1]
        if 'val_mean_absolute_error' in history.history.keys():
            test_acc = history.history['val_mean_absolute_error'][-1]
        else:
            test_acc = 0
    return net_model, history, train_acc, test_acc


class ReportCB(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch {}:'.format(epoch+1), end=' ')
        self.tstart = time.time()

    def on_epoch_end(self, epoch, logs=None):
        t = time.time() - self.tstart
        print(f'{t:.1f}s.', end=' ')
        if 'accuracy' in logs.keys():
            print(f'train acc {logs["accuracy"]:.3f}', end=' ')
            if 'val_accuracy' in logs.keys(): print(f'test acc {logs["val_accuracy"]:.3f}', end='')
        elif 'mean_absolute_error' in logs.keys():
            print(f'train mae {logs["mean_absolute_error"]:.3f}', end=' ')
            if 'val_mean_absolute_error' in logs.keys(): print(f'test mae {logs["val_mean_absolute_error"]:.3f}', end='')
        print('\n')


def create_callbacks(hps):
    # checkpoint_path = hps.folder + "checkpoints/" + "cp-{epoch:04d}.ckpt"
    # save_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=hps.checkpoint_period)
    if hps.label_type == 'clas':
        earlystop_cb = tf.keras.callbacks.EarlyStopping('val_accuracy', verbose=1, patience=hps.early_stopping)
    else:
        earlystop_cb = tf.keras.callbacks.EarlyStopping('mean_absolute_error', verbose=1, patience=hps.early_stopping)
    tensorboard_cb = TensorBoard(log_dir=hps.folder + 'logs\\', histogram_freq=1, write_images=True)
    cb_list = []
    # if hps.save_checkpoints: cb_list.append(save_cb)
    if hps.early_stopping: cb_list.append(earlystop_cb)
    if hps.use_tensorboard: cb_list.append(tensorboard_cb)
    cb_list.append(ReportCB())
    return cb_list


def boxplot_results(hps, i_range, results, save_to):
    exp_labels = ['i_range index [%d]' % i for i in i_range]
    plt.boxplot(results, labels=exp_labels)
    title = 'Labels = {}, Epochs = {}, Repeats = {}'.format(hps.data_title, hps.epochs, hps.repeats)
    plt.title(title)
    plt.savefig(save_to)
    plt.show()
    plt.close()


def plot_train_history(history, metrics, show_ims=False, val=False, save_to=''):
    """ Plot training & validation accuracy/loss values (type = 'accuracy' or 'loss') """
    for m in metrics:
        plt.plot(history[m])
        if val:
            plt.plot(history['val_' + m])
            plt.legend(['train', 'test'], loc='upper left')
        plt.title('Model ' + str(m))
        plt.ylabel(m)
        plt.xlabel('Epoch')
        if save_to != '': plt.savefig(save_to + ' ' + m + '_history.svg')
        if show_ims: plt.show()
        plt.close()


def histplot_results(y_pred, y_true, x_label, class_name, show_ims=False, save_to=''):
    data = pd.DataFrame({x_label: y_pred, class_name: y_true})
    sns.displot(data=data, x=x_label, hue=class_name, kind='kde', bw_adjust=0.5, height=8, aspect=2, fill=True)
    plt.xlabel('{} less likely   ←―――   prediction from ECG   ―――→   {} more likely'.format(class_name, class_name))
    plt.ylabel('density')
    plt.title('{} prediction from ECG: {} for each class'.format(class_name, x_label), y=0.97)
    if save_to != '': plt.savefig(save_to + x_label + '.svg')
    if show_ims: plt.show()
    plt.close()


def density_1d(series_list, name_list, x_label, bw=0.2, var_name='', normalise=False, show_ims=False, save_to=''):
    """ plot densities of multiple 1D series on one set of axes
    series_list: a list of arrays or lists of numbers in [0,1]
    name_list: a list of names for each series - should be the same length as series_list
    x_label: label for the x-axis
    normalise: if True, scales each series to have the same AUC """
    plt.figure(figsize=(12,8))
    max_series_size = max([len(s) for s in series_list])
    for i in range(len(series_list)):
        s = series_list[i]
        density = gaussian_kde(s)
        density.covariance_factor = lambda: bw
        density._compute_covariance()
        x = np.linspace(0, 1, 200)
        y = density(x)
        if normalise:
            y *= max_series_size/len(s)
        plt.plot(x, y, label=name_list[i])
        x, y = np.concatenate(([0], x, [1])), np.concatenate(([0], y, [0]))
        plt.fill(x, y, alpha=0.3)
        plt.xlabel('{} less likely   ←―――   prediction from ECG   ―――→   {} more likely'.format(x_label, x_label))
        plt.ylabel('density')
    plt.legend()
    plt.title('{} prediction from ECG: output for {}'.format(x_label, var_name))
    if save_to != '': plt.savefig(save_to + 'pred density.svg')
    if show_ims: plt.show()
    plt.close()


def tabulate_results(i_range, all_times, all_results):
    print('\t'.join(['i', 'time', '', '', 'accs']))
    for i in range(len(i_range)):
        print('\t'.join([str(i_range[i]), all_times[i]] + ['{:.2f}'.format(x) for x in all_results[i]]))


def optimise_accs(accs):
    """ accs is a list of results of different experiments. Each item is a list of accuracy values at different
    thresholds (all of same length). function finds the optimum accuracy point in each list and returns the average """
    maxes, argmaxes = [], []
    for a in accs:
        maxes.append(max(a))
        argmaxes.append(a.index(max(a)))
    av_argmax = int(sum(argmaxes)/len(argmaxes))
    av_maxes = [a[av_argmax] for a in accs]
    av_peak = sum(av_maxes)/len(av_maxes)
    return av_argmax, av_peak


if __name__ == '__main__':
    main()


def preprocess_and_train(ecgs, labels, hps, model_save_dir, debug=False):
    x_train, x_test, y_train, y_test, ids_train, ids_test, hps = dl.preprocess(ecgs, labels, hps, shuf=True)
    test_acc, ids_test, y_pred, y_test, model, logits = train_net(
        x_train, x_test, y_train, y_test, ids_train, ids_test,
        hps, save_model_to=model_save_dir, debug=debug)
