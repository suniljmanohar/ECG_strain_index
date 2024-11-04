import ECG_plots
import params as p, data_loader as dl, my_timer, trainer as t
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn import metrics


def main():
    """train network to classify HCMR vs UKBB (excluding HCM cases)
    Then apply network to both paper and digital original ECGs
    If digital originals are classified as not HCMR then this indicates bias """

    training = True
    outliers_cutoff = 0.3

    """ bias test 1 - HCMR vs normal """
    save_dir = p.folder['results'] + 'bias test 1\\'
    train_data_spec = ['HCMR vs UKBB bias test', ['combined', ['hypertension'], {'ox': 0}]]
    ox_paper_spec = ['Oxford paper ECGs', ['combined', ['ox'], {'ox': 1}]]
    ox_dig_spec = ['Oxford digital ECGs', ['hcmr ox', ['ox'], {'ox': 1}]]

    """ bias test 2 - sarc pos vs sarc neg 
    save_dir = p.folder['results'] + 'bias test 2\\'
    train_data_spec = ['sarc +- bias test',
                       ['hcmr', ['sarc mutation'], {'sarc mutation':0, 'ox':0}],
                       ['hcmr', ['sarc mutation'], {'sarc mutation':1, 'ox':0}]]
    ox_paper_spec = ['Oxford paper ECGs',
                     ['hcmr', ['sarc mutation'], {'ox': 1}]]
    ox_dig_spec = ['Oxford digital ECGs',
                   ['hcmr ox', ['sarc mutation'], {'ox': 1}]]
    """

    x_train, x_test, y_train, y_test, ids_train, ids_test, hps = fetch_data(train_data_spec, shuf=True)

    hps.folder = save_dir
    if training:
        model, _ = train(x_train, x_test, y_train, y_test, ids_train, ids_test, hps)
    else:
        hps.folder += "2024-01-26 14.18.53 HCMR vs UKBB bias test 2 30 50 0 32x7-32x7-64x5-64x3-128x3-128x3-256x3-256x3-128-32-\\"
        model = tf.keras.models.load_model(hps.folder + 'bias test model.h5')
        print(model.summary())

    preds = apply_model(model, ox_paper_spec, ox_dig_spec, hps, outliers_cutoff)
    preds.to_csv(hps.folder + '_merged predictions.csv')


def train(x_train, x_test, y_train, y_test, ids_train, ids_test, hps):
    hps.folder = hps.folder + my_timer.timestamp() + ' ' + hps.describe() + '\\'
    test_acc, ids_test, y_pred, y_test, model, logits = t.train_net(
        x_train, x_test, y_train, y_test, ids_train, ids_test,
        hps, save_model_to=hps.folder + 'bias test model.h5', debug=False)
    return model, hps


def fetch_data(data_spec, train_prop=1., shuf=False):
    # data_spec formatted as [data_title, [ecg_src1, [label_cols1], selection1], [ecg_src2, ...], ...]
    hps = p.HyperParams(data_spec)
    ecgs, labels = dl.get_all_data(hps)
    hps.train_prop = train_prop
    return dl.preprocess(ecgs, labels, hps, shuf=shuf)


def apply_model(model, ox_paper_spec, ox_dig_spec, hps, outliers_cutoff):
    paper_data = fetch_data(ox_paper_spec, 1, shuf=False)
    paper_pred = model.predict(paper_data[0])

    dig_data = fetch_data(ox_dig_spec, 1, shuf=False)
    dig_pred = model.predict(dig_data[0])

    # match IDs
    paper_df = pd.DataFrame(paper_pred[:,0], index=paper_data[4], columns=['paper'])
    dig_df = pd.DataFrame(dig_pred[:, 0], index=dig_data[4], columns=['digital'])
    merged = paper_df.merge(dig_df, 'inner', left_index=True, right_index=True)

    # analysis
    merged, outliers = find_outliers(merged, outliers_cutoff)
    print('Outliers and difference in predictions:\n', outliers['diff'])

    # histograms
    # histogram_comparisons(dig_data, hps, outliers, paper_data)

    # direct comparison plots
    scatter_preds(merged['paper'], merged['digital'], save_file=hps.folder)
    density_1d([paper_pred[:,0], dig_pred[:,0]], ['paper', 'digital'], 'model output value',
               save_file=hps.folder)

    # outliers plots
    diffs = outliers_plots(paper_data, dig_data, outliers, merged, hps, only_outliers=True)
    diffs.to_csv(hps.folder + '_digital-paper subtractions.csv')

    return merged


def find_outliers(data, cutoff=0.4):
    data['diff'] = data['digital'] - data['paper']
    outliers = data[abs(data['diff'])>cutoff]
    return data, outliers


def split_outliers(paper_data, dig_data, outliers):
    hi = [paper_data[0][np.isin(paper_data[4], (outliers[outliers['diff'] > 0].index))],
          dig_data[0][np.isin(dig_data[4], (outliers[outliers['diff'] > 0].index))]]
    lo = [paper_data[0][np.isin(paper_data[4], (outliers[outliers['diff'] < 0].index))],
          dig_data[0][np.isin(dig_data[4], (outliers[outliers['diff'] < 0].index))]]
    return hi, lo


def scatter_preds(preds1, preds2, save_file='', show_ims=True):
    res = pearsonr(preds1, preds2)
    pear_r, pear_p, pear_ci = res.statistic, res.pvalue, res.confidence_interval
    res = spearmanr(preds1, preds2)
    spear_r, spear_p = res.statistic, res.pvalue
    mae = round(metrics.mean_absolute_error(preds1, preds2), 3)
    print(f'Pearson correlation coeffecient = {pear_r} (p = {pear_p})')
    print(f'Spearman correlation coeffecient = {spear_r} (p = {spear_p})')
    print('Mean absolute error =', mae)
    fig = plt.figure(figsize=(12, 12))
    plt.scatter(preds1, preds2, s=80, facecolors='none', edgecolors='b')
    plt.title(f'Correlation between paper and digitised ECG predictions\nPearson r={pear_r:.3f} (p = {pear_p:.3f})\n'
              f'Spearman ro={spear_r:.3f} (p = {spear_p:.3f})')
    plt.plot(np.unique(preds1), np.poly1d(np.polyfit(preds1, preds2, 1))(np.unique(preds1)))
    plt.plot((0,1), (0,1), 'r-')
    plt.xlabel('Paper ECG prediction')
    plt.ylabel('Digital ECG prediction')
    if save_file != '': plt.savefig(save_file + '_paper vs digital scatter plot.svg')
    if show_ims: plt.show()
    plt.close()


def density_1d(series_list, name_list, x_label, bw=0.2, normalise=True, save_file=''):
    """ plot densities of multiple 1D series on one set of axes
    series_list: a list of arrays or lists of numbers in [0,1]
    name_list: a list of names for each series - should be the same length as series_list
    x_label: label for the x-axis
    normalise: if True, scales each series to have the same AUC """
    max_series_size = max([len(s) for s in series_list])
    handles = []
    fig = plt.figure(figsize=(12, 12))
    for i in range(len(series_list)):
        s = series_list[i]
        density = gaussian_kde(s)
        x = np.linspace(0, 1, 200)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        y = density(x)
        if normalise:
            y *= max_series_size/len(s)
        handles.append(plt.plot(x, y, label=name_list[i]))
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.legend()
    if save_file != '': plt.savefig(save_file + '_paper vs digital density plots.svg')
    plt.show()


def outliers_plots(d1, d2, outliers, preds, hps, only_outliers=False):    # d1=paper, d2=digital
    means, variances, eids = [], [], []
    for i in range(len(d1[4])):
        eid = d1[4][i]
        if eid in d2[4]:
            j = d2[4].index(eid)
            ecg1 = d1[0][i]
            ecg2 = d2[0][j]
            if (not only_outliers) or (eid in outliers.index):
                # compare_plots(np.transpose(ecg1), np.transpose(ecg2), outliers.loc[eid], eid,
                #               save_to=hps.folder + eid + ' comparison plot', show_ims=hps.show_ims)
                overlay_plot(np.transpose(ecg1), np.transpose(ecg2), preds.loc[eid], eid, hps,
                              save_to=hps.folder + eid + ' overlay plot', show_ims=hps.show_ims)
                mean, var = subtract_signals(ecg1, ecg2)
                means.append(mean)
                variances.append(var)
                eids.append(eid)
        else:
            print('{} not in digital data index'.format(eid))

    output = pd.DataFrame(data={'diff mean': means, 'diff variance': variances}, index=eids)
    return output


def subtract_signals(ecg1, ecg2):
    diff = ecg1 - ecg2
    return diff.mean(), diff.var()


def compare_plots(ecg1, ecg2, preds, eid, save_to='', show_ims=True):
    n_leads = len(ecg1)
    fig = plt.figure(figsize=(10, 20))
    grid = plt.GridSpec(n_leads, 2, hspace=0.)
    for i in range(2):
        axs = [None] * n_leads
        for x in range(n_leads):
            axs[x] = fig.add_subplot(grid[x, i])
            axs[x].plot([ecg1, ecg2][i][x], 'k-')
            ECG_plots.format_ecg_plot(axs[x], v_scale=0.8)
            if x==0:
                axs[x].set_title([f'Paper (ECG HTN score = {preds["paper"]:.3f})',
                                  f'Digital (ECG HTN score = {preds["digital"]:.3f})'][i])
    plt.suptitle(f'{eid}: Difference in preds = {preds["diff"]:.2f}')
    if save_to != '': plt.savefig(save_to + '.svg')
    if show_ims: plt.show()
    plt.close()


def overlay_plot(ecg1, ecg2, preds, eid, hps, save_to='', show_ims=True):
    n_leads = len(ecg1)
    fig = plt.figure(figsize=(6, 20))
    grid = plt.GridSpec(n_leads, 1, hspace=0.5)
    axs = [None] * n_leads
    for x in range(n_leads):
        axs[x] = fig.add_subplot(grid[x, 0])
        hnd1, = axs[x].plot(ecg1[x], 'b-', label='paper')
        hnd2, = axs[x].plot(ecg2[x], 'g-', label='digital')
        axs[x].set_title(hps.cnn_lead_order[x])
        ECG_plots.format_ecg_plot(axs[x])
        if x==0: handles = [hnd1, hnd2]
    fig.suptitle(f'{eid}: Difference in preds = {preds["diff"]:.2f}\n'
                 f'Paper (ECG HTN score = {preds["paper"]:.3f})\n'
                 f'Digital (ECG HTN score = {preds["digital"]:.3f})')
    fig.legend(handles=handles)
    if save_to != '': plt.savefig(save_to + '.svg')
    if show_ims: plt.show()
    plt.close()


def compare_hists(paper, dig, save_to='', show_ims=True):
    paper = paper.flatten()
    dig = dig.flatten()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches((20,10))
    ax1.hist(paper, bins=30)
    ax1.set_xlim(-400, 400)
    ax2.hist(dig, bins=30)
    ax2.set_xlim(-400, 400)
    ax1.set_title(f'Paper (mean={paper.mean():.1f}, std={paper.std():.1f})')
    ax2.set_title(f'Digital (mean={dig.mean():.1f}, std={dig.std():.1f})')
    plt.suptitle('Histogram of ECG trace values')

    if save_to != '': plt.savefig(save_to + '.svg')
    if show_ims: plt.show()
    plt.close()


def histogram_comparisons(dig_data, hps, outliers, paper_data):
    compare_hists(paper_data[0], dig_data[0], show_ims=hps.show_ims, save_to=hps.folder + 'histogram all data')
    hi, lo = split_outliers(paper_data, dig_data, outliers)
    compare_hists(*lo, show_ims=hps.show_ims,
                  save_to=hps.folder + 'histogram low outliers')  # digital score < paper score
    compare_hists(*hi, show_ims=hps.show_ims,
                  save_to=hps.folder + 'histogram high outliers')  # digital score > paper score


if __name__ == '__main__':
    main()