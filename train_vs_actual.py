import os
import copy
import shutil
import ECG_plots
import all_preds
import params as p, data_loader as dl, eval_results as er
import plots_2D
import preds_analysis
import trainer as t, my_tools as mt

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def main():
    epochs = [30, 50]
    for run in range(len(epochs)):
        hps = p.HyperParams(p.subsets['all BP'])
        hps.epochs = epochs[run]
        print('Run {} of {}'.format(run+1, hps.repeats))
        run_plot(hps)


def run_plot(hps):
    train_on = 'hypertension'
    plot_on = 'DBP'
    series = {'UKBB normal BP': {'origin': 0, 'hypertension': 0},
              'UKBB high BP': {'origin': 0, 'hypertension': 1},
              'HCMR sarcomere negative': {'origin': 1, 'sarc mutation': 0},
              #'HCMR sarcomere negative high BP': {'origin': 1, 'sarc mutation': 0, 'hypertension': 1},
              'HCMR sarcomere positive': {'origin': 1, 'sarc mutation': 1},
              #'HCMR sarcomere negative normal BP': {'origin': 1, 'sarc mutation': 0, 'hypertension': 0},
              }
    train_series = {'all': ['UKBB normal BP', 'UKBB high BP', 'HCMR sarcomere negative', 'HCMR sarcomere positive'],
                    'UKBB': ['UKBB normal BP', 'UKBB high BP'],
                    'UKBB+SN': ['UKBB normal BP', 'UKBB high BP', 'HCMR sarcomere negative']}
    train_series_used = 'UKBB+SN'
    add_missing_preds = True  # merges preds from k-fold with additional preds for files with missing labels
    debug = False
    do_train = True
    do_gradcam = False
    hps.set_folder('\\\\?\\'+hps.folder+'k-fold\\'+mt.timestamp()+' '+hps.describe()+' train_on '+train_series_used+'\\')

    hps.train_prop = 1.
    if hps.k == 0:
        train_and_plot(train_on, plot_on, series, hps, do_train=do_train, debug=debug, do_gradcam=do_gradcam)
    else:
        k_fold_plot(train_on, plot_on, series, hps, train_series[train_series_used],
                    add_missing_preds=add_missing_preds, debug=debug)


def k_fold_plot(train_on, plot_on, series, hps, train_series, add_missing_preds=True, debug=False):
    # load training ecgs and labels
    groups = {x: [hps.datasets[0][0], [train_on, plot_on], series[x]] for x in series}
    train_grp_data = dl.get_data_groups(groups, hps, do_preprocess=False, dropna=False)
    train_grp_data = group_dropna(train_grp_data, [train_on])
    # train data omits any without hypertension status (req for training) or HCMR without sarc status (req for grouping)

    plot_groups = {x: [hps.datasets[0][0], [plot_on], series[x]] for x in series}
    plot_grp_data = dl.get_data_groups(plot_groups, hps, do_preprocess=False, dropna=False)
    plot_grp_data = group_dropna(plot_grp_data, [plot_on])
    # plot data omits any without plot_on & any ungroupable (UKBB no HTN status, HCMR no sarc status)

    # split each series into k-folds
    kf_dict = {grp: list(KFold(hps.k, shuffle=True).split(train_grp_data[grp][0])) for grp in train_grp_data}

    # train and predict in k-folds
    preds = []
    for n in range(hps.k):
        print('\n\nRunning fold {} of {}'.format(n+1, hps.k))
        save_to = hps.folder + 'X=trained on {} Y=actual {} Fold={}'.format(train_on, plot_on, n+1)
        preds_k, model = split_and_run_kfold(kf_dict, n, train_grp_data, hps, train_on,
                            save_to, train_series, save_models=False, debug=debug)
        preds.append(preds_k)

    preds_joined = pd.concat(preds, ignore_index=False)

    preds_joined = groups_to_df(preds_joined, train_grp_data, plot_on)
    if add_missing_preds:   # missing preds will be those in HCMR with HTN status missing, plus those ungroupable
        preds_joined = all_preds.main(old_preds=preds_joined, model=model, merge=True)  # use last k-fold's model for missing data

    group_and_plot_preds(plot_grp_data, hps, plot_on, preds_joined, train_on)
    # plot doesn't include data where hypertension status is missing but BP and ECG->preds exists - check

    # save preds
    preds_joined.to_csv(hps.folder + 'preds_logits.csv')


def split_and_run_kfold(kf_dict, n, grp_data, hps, train_on, save_to, train_series, save_models=False, debug=False):
    # combine k-folds from each group (e.g. 90% from each of 4 groups for training, 10% from each group for testing)
    test_ecgs, test_labels, train_ecgs, train_labels = combine_kfolds(grp_data, kf_dict, n, train_series)

    # train and predict
    hps.train_prop = 1.

    # load training data
    x_train, _, y_train, _, ids_train, _, hps = dl.preprocess(train_ecgs, train_labels[[train_on]], hps, shuf=True)

    # load test data
    x_test, _, y_test, _, ids_test, _, hps = dl.preprocess(test_ecgs, test_labels[[train_on]], hps, shuf=True)

    # train and predict
    if save_models:
        save_model_to = save_to+'.h5'
    else:
        save_model_to = ''
    hps_k = copy.copy(hps)
    hps_k.folder = hps.folder + 'Fold {} '.format(n+1)
    _, ids_test, y_pred, y_true, model, logits = t.train_net(x_train, x_test, y_train, y_test,
                                                         ids_train, ids_test, hps_k,
                                                         save_model_to=save_model_to, debug=debug)

    return pd.DataFrame(data={'y_pred':y_pred, 'logits':logits}, index=ids_test), model


def combine_kfolds(grp_data, kf_dict, n, train_series):
    train_ecgs, train_labels, test_ecgs, test_labels = [], [], [], []
    for grp in grp_data:
        train, test = kf_dict[grp][n]
        if grp in train_series:
            train_ecgs.append(grp_data[grp][0][train])
            train_labels.append(grp_data[grp][1].iloc[train])
        test_ecgs.append(grp_data[grp][0][test])
        test_labels.append(grp_data[grp][1].iloc[test])
    train_ecgs = np.concatenate(train_ecgs)
    test_ecgs = np.concatenate(test_ecgs)
    train_labels = pd.concat(train_labels, ignore_index=False)
    test_labels = pd.concat(test_labels, ignore_index=False)
    return test_ecgs, test_labels, train_ecgs, train_labels


def group_and_plot_preds(grp_data, hps, plot_on, preds, train_on):
    # split predictions into series groups
    data = groups_to_df(preds, grp_data, plot_on)

    # plot 2D densities
    series_list = pd.DataFrame(data=list(grp_data.keys()), columns=['series'])
    series_list = preds_analysis.sorted_series_list(series_list)
    plots_2D.multiseries_contour(data.rename(columns={'y_true':plot_on}), plot_on, 'y_pred', series_list,
                                 x_lim=[60,100], y_lim=[0,1], show_ims=False, save_to=hps.folder)
    plots_2D.multiseries_contour(data.rename(columns={'y_true':plot_on}), plot_on, 'logits', series_list,
                                 x_lim=[60,100], y_lim=[-5,5], show_ims=False, save_to=hps.folder + '_logits ')
    split_data = preds_analysis.split_subsets(data.rename(columns={'y_true':plot_on}))
    plots_2D.multiseries_scatter(split_data, plot_on, 'y_pred', show_ims=False,
                                 save_file=hps.folder + f'_scatter results.svg')
    # er.sliding_window_percentile(data, plot_on, x_axis, show_ims=False, save_to=hps.folder)
    return data


def groups_to_df(preds, grp_data, plot_on):
    plot_data = {}
    for grp in grp_data:
        y_pred = preds.loc[grp_data[grp][1].index]['y_pred']
        logits = preds.loc[grp_data[grp][1].index]['logits']
        y_true = grp_data[grp][1][plot_on]
        plot_data[grp] = [y_true, y_pred, logits, y_pred.index]
    # convert predictions to dataframe
    data = er.series_to_panda(plot_data)
    return data


def group_dropna(grps, drop_cols):
    for g in grps:
        nonnull = ~grps[g][1][drop_cols].isnull().any(axis=1)
        grps[g] = grps[g][0][nonnull], grps[g][1][nonnull]
    return grps


def train_and_plot(train_on, plot_on, series, hps, do_train=True, debug=False, do_gradcam=False):
    """ trains and then plots 2D predicted vs actual BP on 2D plot
    train_on: a column name from the labels defined in hps
    plot_on: also a column name (needs to be continuous, will be plotted on y-axis)
    hps: HyperParams object
    """
    # get training data
    ecgs, labels = dl.get_all_data(hps)

    # train and save model
    model_fname = hps.folder + train_on + ' ({} epochs).h5'.format(hps.epochs)
    if do_train:
        t.preprocess_and_train(ecgs, labels.loc[:, [train_on]], hps, model_fname, debug=debug)

    if do_gradcam:
        gc.run_ECG_gradCAM(model_fname, ecgs, save_to=hps.folder+'GradCAM\\'+mt.timestamp())

    # prepare and plot data
    groups = {x: ['combined', [plot_on], series[x]] for x in series}
    data = dl.get_data_groups(groups, hps)

    # DEBUGGING
    if debug:
        for i in range(100):
            ECG_plots.plot_iso_ecg_vert(data['HCMR sarcomere positive'][0][i],
                                        {'filename': data['HCMR sarcomere positive'][2][i],
                                         'lead order': hps.cnn_lead_order})

    plot_data = er.predict_and_plot(model_fname, data, bw_adjust=1.,
                                    x_label='{} less likely  ←――――――  prediction from ECG  ――――――→  {} more likely'.format(train_on, train_on),
                                    y_label=plot_on,
                                    save_to=hps.folder + mt.timestamp() + ' X - trained on {}, Y - actual {}'.format(train_on, plot_on),
                                    show_ims=hps.show_ims)

    get_group_extremes(plot_data, series, n=20, save_to=hps.folder + train_on + ' extremes ({} epochs).txt'.format(hps.epochs))


def get_group_extremes_names(plot_data, series, n=10, save_to=''):
    if save_to !='':
        f = open(save_to, 'w')
    for s in series:
        txt = 'Extremes for {}'.format(s) + '\n' + er.find_extremes(plot_data[plot_data['series'] == s], n) + '\n\n'
        print(txt)
        if save_to != '':
            f.write(txt)
    if save_to != '':
        f.close()


def raw_BP_plot(hps):
    # get data
    groups = {'UKBB normal BP': ['ukbb median', ['SBP', 'DBP'], {'hypertension':0, 'HCM':0}],
              'UKBB high BP': ['ukbb median', ['SBP', 'DBP'], {'hypertension':1, 'HCM':0}],
              'HCMR sarc neg': ['hcmr', ['Systolic BP at time of CMR', 'Diastolic BP at time of CMR'], {'sarc mutation': 0}],
              'HCMR sarc pos': ['hcmr', ['Systolic BP at time of CMR', 'Diastolic BP at time of CMR'], {'sarc mutation': 1}]}

    hps.train_prop = 1.
    data = {}
    for g in groups:
        _, labels = dl.get_data(groups[g][0], groups[g][1], groups[g][2], hps.cnn_lead_order)
        renaming = {'SBP': 'y_true', 'DBP': 'y_pred', 'Systolic BP at time of CMR': 'y_true', 'Diastolic BP at time of CMR': 'y_pred'}
        data[g] = [labels.rename(columns=renaming)['y_pred'], labels.rename(columns=renaming)['y_true']]
    data = er.series_to_panda(data)
    er.multiseries_contour(data, 'Diastolic BP', 'Systolic BP')


def report_data(labels):
    for x in ['SBP', 'DBP']:
        print('normotensive: mean {} = {:.1f}'.format(x, np.mean(labels[labels['hypertension'] == 0][x])))
    for x in ['SBP', 'DBP']:
        print('hypertensive: mean {} = {:.1f}'.format(x, np.mean(labels[labels['hypertension'] == 1][x])))


def get_group_extremes(data, groups, n, save_to):
    for g in groups:
        # create folders
        for x in ['\\highest\\', '\\lowest\\']:
            if not os.path.exists(save_to+g+x):
                os.makedirs(save_to+g+x)
        lows, highs = er.find_extremes(data[data['series'] == g], n)
        for fname in lows:
            copy_ecg_image(fname, save_to+g+'\\lowest\\')
        for fname in highs:
            copy_ecg_image(fname, save_to+g+'\\highest\\')


def copy_ecg_image(fname, copy_to):
    """
    find ECG image corresponding to fname in one of the folders for UKBB or HCMR ECG images
    :param fname: file name (id only, no path or extension) to be found
    :param copy_to: folder to copy image file to
    :return: None
    """
    if fname.find('-') > -1:
        fname_list = os.listdir(p.image_files['hcmr'])
        search_list = [x[:8] for x in fname_list]
        i = search_list.index(fname)
        copy_from = p.image_files['hcmr'] + fname_list[i]
    else:
        fname_list = os.listdir(p.image_files['ukbb median'])
        search_list = [x[:7] for x in fname_list]
        i = search_list.index(fname)
        copy_from = p.image_files['ukbb median'] + fname_list[i]

    shutil.copy(copy_from, copy_to + fname_list[i])


if __name__ == '__main__':
    main()