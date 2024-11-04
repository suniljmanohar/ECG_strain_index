import params as p, trainer as t, augmentation as aug, data_loader as dl, my_tools
from eval_results import eval_clas, conf_matrix
# from source.ECG_plots import plot_iso_ecg, plot_iso_ecg_vert

import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold


def main():
    hps = p.HyperParams(p.subsets['hcmr all sarc'])
    k = 2              # number of folds to use for k-fold validation
    runs = 1            # repeats the whole k-fold validation process this many times
    threshold = 0.5     # value above which predictions are considered positive

    hps.set_folder(hps.folder + 'K-fold\\' + my_tools.timestamp() + ' ' + hps.describe() + '\\')
    # Path(output_folder).mkdir(parents=True, exist_ok=True)
    run_kfolds(hps, k, runs, threshold)


def run_kfolds(hps, k, runs, threshold=0.5, analyse=True):
    """ run k-fold evaluation using parameters in hps
    outputs a numpy array of shape (n, runs+2) where the first column is ECG IDs, second column is actual label values
    and each subsequent column contains the predictions (range [0-1]) from a single run of k-fold validations """

    # set params
    save_aug_n = hps.aug_n  # save augmentation for later during K-fold process
    hps.train_prop, hps.aug_n = 1, 0  # don't split and don't augment

    # get data and preprocess
    ecgs, labels = dl.get_all_data(hps)
    all_preds = []
    ecgs, _, labels, _, ids, _, hps = dl.preprocess(ecgs, labels, hps, shuf=True)
    hps.aug_n = save_aug_n  # reinstate augmentation after data load

    # run k-fold validation
    for r in range(runs):
        print('Run {} of {}'.format(r+1, runs))
        preds = (k_fold(k, ids, ecgs, labels, hps, hps.folder + 'Run {}_'.format(r)))
        all_preds.append(preds)

    # merge and analyse all preds
    all_preds, all_preds_long = merge_dfs(all_preds)
    if analyse:
        all_preds, cms, measures, roc_results = analyse_kfold_results(all_preds, all_preds_long, threshold, hps)

    # save as csv
    all_preds.to_csv(hps.folder + 'all_preds.csv')
    return all_preds, all_preds_long


def k_fold(k, ids, ecgs, labels, hps, output_folder):
    """ Runs 1 complete k-fold validation and returns an array of true and predicted values for each ECG ID
    preds: dict with IDs as keys and list of true and predicted labels as values
    k: number of folds
    ids: list of ECG IDs in order
    ecgs: np array of n ECGs in shape (n, n_leads, lead_length) in same order
    labels: np array of corresponding labels for each ECG ID in same order
    hps: instance of hyperparameter class defined in trainer.py """
    ids = np.array(ids)
    kf = list(KFold(k, shuffle=True).split(labels))
    preds = pd.DataFrame(data={'y_true': labels[:, 1], 'y_pred': [None for i in range(len(ids))]}, index=ids)
    for n in range(len(kf)):
        print('Fold %d of %d' % (n+1,k))

        train, test = kf[n]

        # apply augmentation after splitting into train and test sets (?don't augment test set)
        x_train, y_train, ids_train = aug.augment(ecgs[train],
                                                  pd.DataFrame(data=labels[train], index=ids[train]),
                                                  hps.aug_n)
        x_test,  y_test, ids_test = ecgs[test], labels[test], ids[test]

        # build, fit and evaluate new model for each fold
        _, ids_test, y_pred, y_true, _, _ = t.train_net(x_train, x_test, y_train, y_test, ids_train, ids_test, hps)
        preds = update_preds(preds, ids_test, y_pred)
        cms, measures, roc_results = eval_clas(ids_test, y_pred, y_true, show_ims=False,
                                                      save_to=output_folder + 'fold {}'.format(n))
    return preds.astype('float32')


def analyse_kfold_results(preds, long_preds, threshold, hps):
    """ takes a dataframe with ECG IDs as the index column and actual binary labels as column 'y_true'
     Remaining columns are all predicted values from the CNN in range [0-1] for a binary label labelled 'y_predX'
     Values > (threshold) are treated as True whilst values <= (threshold) are treated as false
     Calculates sens, spec, PPV, NPV, LR+ and LR- for the CNN's predictions """

    # separate IDs, true values
    ids, y_true, y_pred = preds.index.to_numpy(), preds['y_true'].to_numpy(), preds.drop(columns='y_true').to_numpy()
    preds['y_pred_mean'] = preds.drop(columns='y_true').mean(axis=1)
    y_pred_mean = preds['y_pred_mean'].to_numpy()

    # histogram plot
    t.histplot_results(long_preds['y_pred'], long_preds['y_true'], 'model output', hps.data_title,
                       show_ims=hps.show_ims, save_to=hps.folder + '_mean_preds')

    # calculate sens, spec, etc. for mean predictions
    cms, measures, roc_results = eval_clas(ids, y_pred_mean, y_true)
    metrics = pd.DataFrame(measures)
    pd.options.display.float_format = '{:,.3f}'.format
    print(metrics.loc[int(threshold*200)])

    # calculate average proportion of correct predictions for each ID
    preds['mean_acc'] = preds.apply(lambda row: mean_acc(row, threshold), axis=1)

    # sort and return
    preds = preds.sort_values(by=['mean_acc'])
    return preds, cms[int(threshold*200)], measures, roc_results


def mean_acc(row, threshold):
    y_true = row['y_true']
    y_correct = row.drop('y_true') > threshold
    score = np.sum(y_correct == y_true)/len(y_correct)
    return score


def merge_kfold_scores(folder):
    """ Takes all npy array files in folder and merges the IDs and averages the accuracy scores across all files
    Each file should be a np array of shape (n, 2 + experiment_runs) where n is the number of cases and columns are
    [ECG IDs, true values, expt1_values, expt2_values, ...]
    Returns a DataFrame with shape (n, 3) with columns [ECG IDs, true values, average_expt_values] """
    # load arrays from file_list
    arrays = []
    print('Merging files in {}: '.format(folder), end='')
    for fname in os.listdir(folder):
        if fname[-3:] == 'npy':
            print(fname, end=' ')
            arrays.append(np.load(folder + fname))
    print('\n')

    # convert to DataFrames and set indices and column titles
    counter = 0  # number of acc columns found
    dfs = []
    for f in arrays:
        dfs.append(pd.DataFrame(data=f, index=f[:, 0],
                                columns=['id', 'true'] +
                                        ['acc'+str(n) for n in range(counter+1, counter + f.shape[1]-1)]))
        counter += f.shape[1] - 2

    # merge all dfs
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.combine_first(df)
    stacked = dfs[0]
    for df in dfs[1:]:
        df.columns = df.columns[:2].tolist() + ['acc1'] * (len(df.columns) -2)
        print(df.columns.tolist())
        stacked = pd.concat((stacked, df))

    # calculate averages and drop other columns
    acc_cols = ['acc'+str(n) for n in range(1, counter+1)]
    merged['av acc'] = merged[acc_cols].astype('float32').mean(axis=1)
    merged = merged.drop(columns=acc_cols)
    merged = merged.sort_values(by='av acc', axis=0)
    np.savetxt(fname[:-4] + '_merged.csv', merged, fmt='%s', delimiter=",", newline='\n')


def misclas_dist(preds):
    """
    preds is a df with columns y_true as true values and y_pred1, y_pred2, ... as predicted values
    plots a bar chart with the most misclassified files at the left, and the least at the right
    """
    preds['misclas_err'] = preds.drop(columns='y_true') - preds['y_true']


def update_preds(preds, ids, y_pred):
    """
    :param preds: dict as {ECG ID: [true_label, predicted_label1, predicted_label2, ...], ...}
    :param ids: list of ECG IDs
    :param y_pred: corresponding predicted values for each ECG ID
    :return: updated dict with new predicted values added to the value list for each key
    """
    for i in range(len(y_pred)):
        preds.at[ids[i], 'y_pred'] = y_pred[i]
    return preds


def merge_dfs(dfs):
    """ takes a list of dataframes as dfs and produces a single dataframe that includes indexes and columns from
    all the dataframes. If any duplicate index/column pairs are found, the value from the first list in which that
    pairing is found will be prioritised """
    if len(dfs) == 1: return dfs[0], dfs[0]
    for i in range(len(dfs)):
        if i == 0:
            long_result = dfs[0]
            result = dfs[0].rename(columns={'y_pred': 'y_pred0'})
        else:
            long_result = pd.concat([long_result, dfs[i]])
            dfs[i] = dfs[i].rename(columns={'y_pred': 'y_pred' + str(i)})
            result = result.combine_first(dfs[i])
    return result, long_result


if __name__ == '__main__':
    main()




