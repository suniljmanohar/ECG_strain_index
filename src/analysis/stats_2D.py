import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import (pearsonr, spearmanr, mannwhitneyu, ttest_ind, ks_2samp, chi2_contingency, contingency,
                         fisher_exact, shapiro, kruskal, f_oneway)
from pycircstat import watson_williams


DTYPES = {0: 'binary',
          1: 'categorical',
          2: 'ordinal',
          3: 'continuous non-normal',
          4: 'continuous normal'}


def tests():
    return {0: chisq,
            2: spearman_r,
            3: pearson_r,
            4: anova_1w,
            5: kruskal_wallis,
            6: t_test,
            7: mann_whitney}


def stat_test(x1, x2, user_choice=True):
    """ Take 2 columns of data and determine the data type of each column
    Determine appropriate statistical test including testing for normal distribution of continuous vars
    Asks for user decision if normal distribution uncertain
    Where one binary or categorical variable is used, it should be used as x1
    Returns name of statistical test used, statistic value, p-value """
    x1_type = get_type(x1)
    x2_type = get_type(x2)

    test_choice = choose_test(x1_type, x2_type)
    if user_choice:
        x1_type, x2_type, test_choice = ask_user(x1, x2, x1_type, x2_type, test_choice)

    result = tests()[test_choice](x1, x2)


def get_type(x):
    unique_vals = np.unique(x)
    if unique_vals == 2:            # binary
        return 0
    if all_num(x):                  # numeric
        if unique_vals > 10:        # continuous
            if shapiro:             # normal
                return 4
            else:                   # non-normal
                return 3
        else:                       # ordinal
            return 2
    else:                           # categorical
        return 1


def all_num(x):
    is_numeric=True
    i = 0
    while is_numeric and i<len(x):
        if isinstance(x[i], int) or isinstance(x[i], float):
            i+=1
        else:
            is_numeric = False
    return is_numeric


def choose_test(t1, t2):
    if t1 == 0:
        if t2==0 or t2==1: return 0
        if t2==2: return 7
        if t2==3: return 7
        if t2==4: return 6
    if t1 == 1:
        if t2==0 or t2==1: return 0
        if t2==2: pass
        if t2==3: return 5
        if t2==4: return 4
    if t1 == 2:
        if t2==0: return 7
        if t2==1: pass
        if t2==2: return 2
        if t2==3: return 2
        if t2==4: return 2
    if t1 == 3:
        if t2==0: return 7
        if t2==1: return 5
        if t2==2: return 2
        if t2==3: return 3
        if t2==4: return 3
    if t1==4:
        if t2==0: return 6
        if t2==1: return 4
        if t2==2: return 2
        if t2==3: return 3
        if t2==4: return 3


def ask_user(x1, x2, x1_type, x2_type, test_choice):
    fig, axs = plt.subplots(2)
    axs[0].hist(x1)
    axs[1].hist(x2)
    plt.show()
    plt.close()

    make_changes = True
    change = ''
    while make_changes:
        while change not in ['x1', 'x2', 'test', 'none']:
            change = input(f'Types determined as x1: {DTYPES[x1_type]} and x2: {DTYPES[x2_type]}. Test chosen: {test_choice}.'
                           f' Make changes [x1, x2, test, none]? ')
        if change == 'x1':
            change2 = ''
            while change2 not in [0, 1, 2, 3, 4]:
                change2 = input('Enter new type for x1 [0: binary, 1: categorical, 2: ordinal, 3: continuous non-normal, '
                                 '4: continuous normal]: ')
            x1_type = change2
        elif change == 'x2':
            change2 = ''
            while change2 not in [0, 1, 2, 3, 4]:
                change2 = input('Enter new type for x2 [0: binary, 1: categorical, 2: ordinal, 3: continuous non-normal, '
                                 '4: continuous normal]: ')
            x2_type = change2
        elif change == 'test':
            change2 = ''
            while change2 not in [0, 1, 2, 3, 4, 5, 6, 7]:
                change2 = input('Enter new test choice [0: chi2, 1: Fisher exact, 2: Spearman r, 3: Pearson r,'
                                 ' 4: 1-way ANOVA, 5: Kruskal-Wallis, 6: t-test, 7: Mann-Whitney U]: ')
            test_choice = change2
        else:
            make_changes = False

    return x1_type, x2_type, test_choice


def chisq(x1, x2):
    ct = contingency_table(x1, x2)
    if is_ct_valid(ct):
        res = chi2_contingency(ct)
        t, p = res[0], res[1]
        return ['chi-squared', t, p]
    else:
        if np.array(ct).shape == (2,2):
            res = fisher_exact(ct)
            t, p = res.statistic, res.pvalue
            return ['Fisher exact', t, p]
        else:
            return ['cell values too small for chi-square', np.nan, np.nan]


def contingency_table(x1, x2, normalised=False):
    df = pd.DataFrame({'x1':x1, 'x2':x2})
    cats1, cats2 = sorted(list(set(x1))), sorted(list(set(x2)))
    ct = [[((df['x1'] == c1) & (df['x2']==c2)).sum() for c1 in cats1] for c2 in cats2]
    if normalised:
        ct = [[x/sum(row) for x in row] for row in ct]
    return ct


def is_ct_valid(ct):
    """ check contingency table to ensure frequencies and totals are not too low
    Prints warning if they are too low"""
    is_valid = True
    try:
        ct_expected = contingency.expected_freq(ct)
    except ValueError as e:
        print(e)
        return False
    if (ct_expected > 5).sum() < 0.8*ct_expected.size: is_valid = False     # freq_expected > 5 for 80% of cells (only 4 cells)
    if (np.array(ct) == 0).sum() > 0: is_valid = False                      # all values in ct must be > 0
    return is_valid


def spearman_r(x1, x2):
    res = spearmanr(x1, x2)
    return 'Spearman r', res.statistic, res.pvalue


def pearson_r(x1, x2):
    res = pearsonr(x1, x2)
    return 'Spearman r', res.statistic, res.pvalue


def anova_1w(x1, x2):
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    cats1 = sorted(list(set(x1)))
    samples = [(df['x1']==c)['x2'] for c in cats1]
    statistic, pvalue = f_oneway(*samples)
    return 'ANOVA', statistic, pvalue

def kruskal_wallis(x1, x2):
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    cats1 = sorted(list(set(x1)))
    samples = [(df['x1']==c)['x2'] for c in cats1]
    statistic, pvalue = kruskal(*samples)
    return 'Kruskal-Willis', statistic, pvalue

def t_test(x1, x2):
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    cats1 = sorted(list(set(x1)))
    samples = [(df['x1']==c)['x2'] for c in cats1]
    res = ttest_ind(samples[0], samples[1])
    return 't test', res.statistic, res.pvalue

def mann_whitney(x1, x2):
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    cats1 = sorted(list(set(x1)))
    samples = [(df['x1']==c)['x2'] for c in cats1]
    res = mannwhitneyu(samples[0], samples[1])
    return 'Mann-Whitney', res.statistic, res.pvalue
