import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

import my_tools
import params as p


def main():
    # check for variation in these between missing and observed subsets
    test_vars = ['age1', 'BMI1', 'sex', 'SBP', 'DBP', 'hypertension']

    combos = [
        ['age1'],
        ['BMI1'],
        ['sex'],
        ['DBP'],
        ['SBP'],
        ['hypertension'],
        ['max_LVOT_grad'],
        ['NTproBNP'],
        ['TnTStat'],
        ['glsmean'],
        ['log_lge_total6'],
        ['wallthkmax'],
        ['age1', 'BMI1', 'sex', ],
        ['age1', 'BMI1', 'sex', 'DBP'],
        ['age1', 'BMI1', 'sex', 'SBP'],
        ['age1', 'BMI1', 'sex', 'max_LVOT_grad', 'glsmean', 'NTproBNP', 'log_lge_total6', 'wallthkmax'],
    ]

    subgroups = {'UKBB': {'origin': 0},
                 'HCMR': {'origin': 1},
                 'UKBB-N': {'origin': 0, 'hypertension': 0}, 'UKBB-H': {'origin': 0, 'hypertension': 1},
                 'SN-HCM': {'origin': 1, 'sarc mutation': 0}, 'SP-HCM': {'origin': 1, 'sarc mutation': 1}
                 }

    preds_file = 'G:\\results\\Significant results\\2024-03-05 final\\preds_logits.csv'
    folder = 'g:\\missing data analysis\\'
    my_tools.make_dir(folder)

    check_labels(p.label_files['combined'], preds_file, subgroups, combos, test_vars, folder)
    check_ECG(p.ecg_files['combined'][0])
    check_metadata(p.ecg_files['combined'][1])


def check_ECG(filename):
    # are missing ECGs associated with age, sex, BMI, SBP, DBP, hypertension diagnosis?
    # and for excluded ECGs
    ecgs = np.load(filename, allow_pickle=True)


def check_metadata(filename):
    md = np.load(filename, allow_pickle=True)


def check_labels(labels_file, preds_file, subgroups, combos, test_vars, folder):
    labels = pd.read_csv(labels_file, index_col=0)
    preds = pd.read_csv(preds_file, index_col=0)
    labels.index = labels.index.map(str)
    preds.index = preds.index.map(str)

    # remove those without ECG
    merged = preds.join(labels)

    check_diff_by_missing(merged, {'HCMR': {'origin': 1}}, combos, ['sarc mutation'],
                          folder + 'sarc status check.csv')
    check_diff_by_missing(merged, subgroups, combos, test_vars,
                          folder + 'label checks.csv')

    follow_up_BP(labels)


def check_diff_by_missing(labels, subgroups, combos, test_vars, folder):
    # are missing data associated with age, sex, BMI, SBP, DBP, hypertension diagnosis
    # e.g. t-test between complete data and missing data (those who did not attend follow up
    all_results = []

    for comb in combos:
        for var in test_vars:
            print(f'Testing {var} where at least one of {comb} is missing')
            if var in comb:
                pass
            else:
                sub_results = difference_by_missing(labels, subgroups, comb, var)
                all_results += sub_results
    results = pd.DataFrame(data=all_results,
                           columns=['subgroup', 'label missing', 'variable tested', 'n missing', 'n observed',
                                    'missing mean', 'observed mean',
                                    'test', 'test statistic', 'test p value', 'test CI low', 'test CI high'])
    results.to_csv(folder)


def difference_by_missing(labels, subgroups, comb, var):
    results = []
    for series in subgroups:
        filters = subgroups[series]
        labels_grp = labels.loc[(labels[list(filters)] == pd.Series(filters)).all(axis=1)]
        observed = labels_grp.dropna(subset=comb)
        missing = labels_grp[labels_grp[comb].isna().any(axis=1)]
        missing_mu, observed_mu = missing[var].mean(), observed[var].mean()
        if len(missing) > 0 and len(observed) > 0:
            if len(set(observed[var])) > 2:     # assume continuous if not binary
                res = stats.ttest_ind(observed[var], missing[var], nan_policy='omit')
                ci_low, ci_high = res.confidence_interval(0.95).low, res.confidence_interval(0.95).high
                test_type = 't-test'
                # box_plots([observed[var].dropna(), missing[var].dropna()],
                #           ['observed', 'missing'],
                #           title=f'{var} where {label} is observed or missing in {data} (p = {res.pvalue})',
                #           save_to=FOLDER + f'{var} where {label} is observed or missing in {dict_to_str(data)}.svg',
                #           display=False
                #           )

            else:
                res = my_binom_test(missing[var], observed[var])
                ci_low, ci_high = res.proportion_ci(0.95).low, res.proportion_ci(0.95).high
                test_type = 'binomial'

            results.append([series, comb, var, len(missing), len(observed), missing_mu, observed_mu,
                            test_type, res.statistic, res.pvalue, ci_low, ci_high])
        else:
            results.append([series, comb, var, len(missing), len(observed), missing_mu, observed_mu,
                            None, None, None, None, None])
    return results


def dict_to_str(d):
    """ converts dict to string safe to use in filenames """
    d = str(d)
    return d.replace(':','-')


def box_plots(series, series_names, title='', display=True, save_to=''):
    plt.boxplot(series, labels=series_names)
    plt.suptitle(title)
    if display: plt.show()
    if save_to != '': plt.savefig(save_to)
    plt.close()


def my_binom_test(missing, observed):
    missing = missing.dropna()
    observed = observed.dropna()
    res = stats.binomtest(int(missing.sum()), len(missing), observed.sum() / len(observed), alternative='two-sided')
    return res


def follow_up_BP(labels):
    sbp_cols = [x for x in labels.columns if (x[:5]=='4080-' or x[:3]=='93-')]
    dbp_cols = [x for x in labels.columns if (x[:5]=='4079-' or x[:3]=='94-')]
    for field_cols in [sbp_cols, dbp_cols]:
        cols = labels[field_cols]
        nan_count = cols.isna().sum(axis=1)
        # how many missing, and are they sequential or non-sequential


if __name__ == '__main__':
    main()
