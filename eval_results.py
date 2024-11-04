import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches, matplotlib.lines as mlines
import seaborn as sns
from sklearn import metrics
from scipy import stats


def plot_predictions(y_true, y_pred, title, type, show_ims=False, save_file=''):
    """ plots a scatter of true values against predicted values
    - y_true = list of 2 1D arrays of the true values [train values, test values]
    - y_pred = list of 2 1D array of predicted values (same dim as true, also [train values, test values])
    - title = string for title of plot
    - type = 'error' for subtracted plot or 'corr' for direct correlation plot
    - save_file = string of path and filename to save to (if '' then doesn't save)
    Returns None """

    if type == 'error': y_pred -= y_true
    fig = plt.figure(figsize=(24, 12))
    plt.scatter(y_true, y_pred, marker='o', s=8)
    if type == 'error':
        plt.axhline(y=0, color='r', linewidth=0.5)
        plt.ylabel('Prediction error')
    if type == 'corr':
        plt.plot(list(range(int(max(y_pred)))), 'r-', linewidth=0.5)
        plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, y_pred, 1))(np.unique(y_true)))
        plt.ylabel('Predicted value')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.xlabel('True value')
    if save_file != '': plt.savefig(save_file + '.svg')
    if show_ims: plt.show()
    plt.close()


def calc_errors(ids, y_pred, y_true, show_ims=False, save_to='', plot_type='corr'):
    """ Compares y_true and y_pred
    - model = keras model object with inputs matching data dim in trainer.py, output is Dense(1) size
    - label_cols = column no. of labels being tested
    - x = the ECG lead data in the form [train data, test data]
    - y_true = the label data also in the form [train labels, test labels]
    - plot_type = 'corr' to plot direct correlation, 'error' to plot differences, and '' for no plot
    Returns the mean absolute error between predicted and actual values, and the correlation coefficient """

    r = round(np.corrcoef(y_true[:,0], y_pred[:,0])[0, 1], 3)
    mae = round(metrics.mean_absolute_error(y_true, y_pred), 3)
    print('Correlation coeffecient =', r)
    print('Mean absolute error =', mae)

    # Plot results
    if plot_type == 'corr':
        plt_title = 'Prediction accuracy (r = ' + str(r) + ')'
        plot_predictions(y_true[:,0], y_pred[:,0], plt_title, 'corr', show_ims=show_ims, save_file=save_to)
    elif plot_type == 'error':
        plt_title = 'Prediction error (mean absolute error = ' + str(mae) + ')'  # if plotting error
        plot_predictions(y_true[:,0], y_pred[:,0], plt_title, 'error', show_ims=show_ims, save_file=save_to)
    # regression_plot(y_true, y_pred)
    return mae, r


def conf_matrix(ids, y_pred, y_true, threshold=0.5):
    """ Calculates TP, TN, FP, FN for binary classifier predictions where y_true in [0,1]
    y_pred: array of shape (n,) containing predicted values / probabilities of label being 1
    y_true: array of shape (n,) of actual label values, same order as y_pred
    threshold: float - the value above which y_pred is interpreted as corresponding to a label value of 1
    ids:list of record IDs, in matching order to predicted and true values
    Returns tp, tn, fp and fn lists, each listing the IDs which have been classified as true positive etc. """
    assert len(y_true) == len(y_pred)
    tp, tn, fp, fn = [], [], [], []

    # convert predictions into absolute values
    y_pred_bin = [1 if x > threshold else 0 for x in y_pred]

    # classify them
    for i in range(len(y_true)):
        if y_pred_bin[i]:
            if y_true[i]:
                tp.append(ids[i])
            else:
                fp.append(ids[i])
        else:
            if y_true[i]:
                fn.append(ids[i])
            else:
                tn.append(ids[i])
    tp_n, fp_n, tn_n, fn_n = len(tp), len(fp), len(tn), len(fn)
    measures = {
        'sens': allow_div0(tp_n, tp_n + fn_n),
        'spec': allow_div0(tn_n, tn_n + fp_n),
        'ppv' : allow_div0(tp_n, tp_n + fp_n),
        'npv' : allow_div0(tn_n, tn_n + fn_n),
        'acc' : (tp_n + tn_n)/(tp_n + fp_n + tn_n + fn_n),
        'lr_pos': allow_div0(tp_n/(tp_n + fn_n), 1 - tn_n/(tn_n + fp_n)),
        'lr_neg': allow_div0(1 - tp_n/(tp_n + fn_n), tn_n/(tn_n + fp_n)),
        'f1': 2*tp_n /(2*tp_n + fp_n + fn_n),
        'youden': allow_div0(tp_n, tp_n + fn_n) + allow_div0(tn_n, tn_n + fp_n) -1,
        'bal_acc': metrics.balanced_accuracy_score(y_true, y_pred_bin, adjusted=True),
        'matthews': metrics.matthews_corrcoef(y_true, y_pred_bin)
    }

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, measures


def allow_div0(a, b):
    """ returns a/b unless b=0, in which case returns 1"""
    if b==0: return 1
    return a/b


def conf_mat_ROC(ids, y_pred, y_true, show_ims=False, save_to=''):
    """ runs conf_matrix for a range of thresholds
    returns a list of confusion matrices calculated at each incremental threshold """
    conf_mats, measures = [], []
    output_measures = {}
    for div in range(1, 200):
        th = 1 - div / 200
        x = conf_matrix(ids, y_pred, y_true, th)
        conf_mats.append(x[0])
        measures.append(x[1])
    for key in measures[0]:
        output_measures[key] = [x[key] for x in measures]

    fig = plt.figure(figsize=(20,8))
    plt.plot(output_measures['acc'])
    plt.title('Accuracy by threshold')
    if save_to != '': plt.savefig(save_to + '_acc.svg')
    if show_ims: plt.show()
    plt.close()
    return conf_mats, output_measures


def run_ROC_AUC(y_pred, y_true, show_ims=True, save_to=''):
    """ generates false positive and true positive rates based on given predictions, plots ROC and calculates AUC
    :param y_pred: array of predicted values as floats
    :param y_true: array of actual values as floats
    :param show_ims: show the curve plot
    :param save_to: save the curve plot
    :return: false positive and true positive rates
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    fig = plt.figure(figsize=(12, 12))
    plt.plot(fpr, tpr)
    plt.plot([0,1], linestyle='dotted', color='r')
    plt.title("AUC = " + str(round(auc, 3)))
    if save_to != '': plt.savefig(save_to + '_ROC_AUC.svg')
    if show_ims: plt.show()
    plt.close()
    return fpr, tpr, auc


def eval_clas(ids, y_pred, y_true, show_ims=False, save_to=''):
    """ calculates various metrics to evaluate the predictions of a keras model
    returns a list of confusion matrices at various thresholds
    and a new version of preds updated with the new predictions
    :param ids: list of ECG IDs
    :param y_pred: corresponding predicted values for each ECG ID
    :param y_true: corresponding true values for each ECG ID
    :param show_ims: show ROC curve plot
    :param save_to: pathname to save ROC plot image to
    """
    cms, measures = conf_mat_ROC(ids, y_pred, y_true, show_ims=show_ims, save_to=save_to)
    roc_results = run_ROC_AUC(y_pred, y_true, show_ims=show_ims, save_to=save_to)
    with open(save_to + 'metrics.pkl', "wb") as output_file:
        pickle.dump([cms, measures], output_file)
    # find_extremes(pd.DataFrame(index=ids, data={'y_pred': y_pred, 'y_true': y_true}), 10)
    return cms, measures, roc_results


def find_extremes(preds, n):
    """ identify the n highest scoring and n lowest scoring predicted values in preds and return the IDs
    preds: dataframe with IDs as index, column 'y_pred' as the predicted values in [0, 1] """
    asc = preds.sort_values(by='y_pred', ascending=True)
    lowest, highest = asc.head(n), asc.tail(n)
    low_names, high_names = [str(x) for x in lowest.index.tolist()], [str(x) for x in highest.index.tolist()]
    output = ['Lowest {} scores:'.format(n), '\n'.join(low_names),
              'Highest {} scores:'.format(n), '\n'.join(high_names)]
    output = '\n'.join(output)
    return low_names, high_names


def predict_and_plot(model_fname, data, x_label='y_pred', y_label='y_true', bw_adjust=1., save_to='', show_ims=False):
    """
    :param model_fname: pathname of trained model to load
    :param data: dict of form {'series 1': [ecg_array1, labels1],  ...} where
       ecg_array is a np array with shape (n_ecgs, lead_count, lead_length)
       labels1 is a dataframe with only one column of length n_ecgs, containing matching labels for each ecg in series
    :return:
    """
    plot_data = predict_multiseries(model_fname, data)
    plot_data = series_to_panda(plot_data)
    multiseries_contour(plot_data, x_label, y_label, bw_adjust, save_to=save_to, show_ims=show_ims)
    sliding_window_percentile(plot_data, x_label, y_label, save_to=save_to, show_ims=show_ims)
    return plot_data


def predict_multiseries(model_fname, data):
    """
    model_fname: pathname of trained model to load
    data: dict of form {'series 1': [ecg_array1, labels1, ids1],  ...} where
       ecg_array is a np array with shape (n_ecgs, lead_count, lead_length)
       labels1 is a dataframe with only one column of length n_ecgs, containing matching labels for each ecg in series
       ids is a list of the IDs for each array/label pair
    """
    model = tf.keras.models.load_model(model_fname)
    output = {}
    for x in data.items():
        series_name = x[0]
        ecg_array = x[1][0]
        y_true = x[1][1][:,0]

        def pred_logits_probs(net_model, x_test):
            log_odds_model = tf.keras.models.Model(inputs=net_model.input,
                                                   outputs=[net_model.get_layer('log_odds').output, net_model.output])
            scores = log_odds_model.predict(x_test)
            logits, y_pred = scores[0], scores[1]
            return logits[:, 1] - logits[:, 0], y_pred[:, 1]

        logits, y_pred = pred_logits_probs(model, ecg_array)
        output[series_name] = [y_true, y_pred, logits, x[1][2]]
    return output


def series_to_panda(data):
    """
    :param data: dict of form {'series 1': [y_true1, y_pred1, logits, ids],  ...}
    :return: pd dataframe with columns ['series name', 'y_true', 'y_pred', 'logits'] indexed by ids
    """
    df_list = []
    for i in data.items():
        series_name, series_data = i
        df = pd.DataFrame(index=series_data[3], data={'series': [series_name for x in range(len(series_data[0]))],
                                                      'y_true': series_data[0].astype(np.float16),
                                                      'y_pred': series_data[1].astype(np.float16),
                                                      'logits': series_data[2].astype(np.float16)
                                                      })
        df_list.append(df)
    return pd.concat(df_list)


def multiseries_contour(data, x_label, y_label, bw_adjust=1., contour_threshold=0.6, x_lim=None, y_lim=None,
                        show_ims=False, save_to='', order_list=None):
    """ plot multiple series of data in a scatter plot
    data: dataframe, with column headers ['series name', 'y_pred', 'y_true']
    """
    print('Calculating multi-series contour plot')
    g = sns.JointGrid(data=data, x='y_true', y='y_pred', height=12)
    g.figure.set_size_inches(24, 12)
    colours = ['green', 'red', 'yellow', 'blue', 'grey']
    cmaps = ['Greens', 'Reds', 'YlOrBr', 'Blues', 'Greys']
    series_list = list_series(data, order_list)
    patches = []
    for i in range(len(series_list)):
        subset = data[data['series'] == series_list[i]]
        sns.kdeplot(data=subset, x='y_true', y='y_pred', cmap=cmaps[i], bw_adjust=bw_adjust, fill=True, alpha=0.5,
                    ax=g.ax_joint, levels=7, thresh=contour_threshold)
        sns.kdeplot(data=subset, x='y_true', color=colours[i], bw_adjust=bw_adjust,
                    fill=True, common_norm=False, legend=False, alpha=.3, linewidth=0, ax=g.ax_marg_x)
        sns.kdeplot(data=subset, y='y_pred', color=colours[i], bw_adjust=bw_adjust,
                    fill=True, common_norm=False, legend=False, alpha=.3, linewidth=0, ax=g.ax_marg_y)
        patches.append(mpatches.Patch(color=colours[i], label=series_list[i]))

    #g.ax_joint.legend(handles=patches,)
    fix_legend(handles=patches)
    g.ax_joint.set_xlabel(x_label)
    g.ax_joint.set_ylabel(y_label)

    # axis ranges
    if y_lim is None:
        g.ax_marg_y.set_ylim([np.percentile(data['y_pred'], 1), np.percentile(data['y_pred'], 99)])
    else:
        g.ax_marg_y.set_ylim([y_lim[0], y_lim[1]])
    if x_lim is None:
        g.ax_marg_x.set_xlim([np.percentile(data['y_true'], 1), np.percentile(data['y_true'], 99)])
    else:
        g.ax_marg_x.set_xlim([x_lim[0], x_lim[1]])

    for item in g.ax_joint.get_xticklabels() + g.ax_joint.get_yticklabels():
        item.set_fontsize(30)

    if save_to != '': plt.savefig(save_to + ' contour bw=' + str(bw_adjust) + '.svg')
    if show_ims: plt.show()
    plt.close()


def list_series(data, order_list=None):
    series_list = list(np.unique(data['series'].tolist()))
    if order_list is not None:
        series_list.sort(key=lambda x: order_list.index(x))
    if 'HCMR sarcomere negative' in series_list:
        del series_list[series_list.index('HCMR sarcomere negative')]
        series_list.append('HCMR sarcomere negative')   # moves SN-HCM to the end of the list for plotting on top
    return series_list


def fix_legend(**kwargs):
    plt.subplots_adjust(right=0.7)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', **kwargs)


def sliding_window_percentile(data, x_label, y_label, show_ims=False, save_to='', width=0.25, resolution=400):
    """ 2D plot showing the mean and SD bars in a sliding window of width width
    data: dataframe with column headers ['series', 'y_pred', 'y_true']
    y_pred column with numerical data type is used to split by percentile
    y_true column with numerical data type will be plotted on the y-axis
    width: number in [0,1] defining the size of the percentiles to use to calculate mean value
    resolution: number of windows to use in total
    output: 2D plot
    """
    fig = plt.figure(figsize=(12, 12))
    colours = ['cyan', 'magenta', 'yellow', 'grey', 'pink']
    lines = []
    series_list = list(np.unique(data['series'].tolist()))
    for i in range(len(series_list)):
        x, y_lower, y_mean, y_upper = [], [], [], []
        subset = data[data['series'] == series_list[i]]
        for start in range(int(resolution * (1-width))):
            window = get_window(subset, 'y_true', start/resolution, start/resolution + width)
            y = np.mean(window['y_pred'].astype('float32'))
            y_mean.append(y)
            stderr = stats.sem(window['y_pred'].astype('float32'))
            y_lower.append(y - stderr)
            y_upper.append(y + stderr)
            x.append(np.median(window['y_true']))

        plt.plot(x, y_mean, linewidth=1, color=colours[i])
        plt.fill_between(x, y_lower, y_upper, alpha=.5, color=colours[i])
        lines.append(mlines.Line2D([], [], color=colours[i], label=series_list[i]))
    plt.legend(handles=lines)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_to != '': plt.savefig(save_to + 'percentile plot.svg')
    if show_ims: plt.show()
    plt.close()


def get_window(data, key_column, start, stop):
    """
    returns a subset from data, selecting only rows of data where the value in key_column is between start
    and stop percentiles for that column
    :param data: a dataframe
    :param key_column: name of a column in data with numerical data type (string), which is used to split by percentile
    :param start: the lowest percentile to include (float)
    :param stop: the highest percentile to include (float)
    :return: dataframe, subset of data, with all the same column headers
    """
    bottom_perc, top_perc = np.percentile(data[key_column], start*100), np.percentile(data[key_column], stop*100)
    data = data.loc[data[key_column] >= bottom_perc]
    data = data.loc[data[key_column] <= top_perc]
    return data


def eval_regr(ids, y_pred, y_true, show_ims=False, save_to=''):
    mae, r = calc_errors(ids, y_pred, y_true, show_ims=show_ims, save_to=save_to)
    return mae, r


def error_measures(model_path):
    """ test run """
    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
    calc_errors(model, 'regr')