import math
import numpy as np
import pandas as pd

import matplotlib

import plots_2D
from baseline_descriptive_stats import get_summ_stats, get_fields, chi2test, contingency_table, hist_qq
from params import abbreviate

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, mannwhitneyu, ks_2samp, kruskal
from scipy.stats import ttest_ind, zscore
from pycircstat import watson_williams
import statsmodels.formula.api as smf

import my_tools
import params as p
from label_dtypes import label_dtypes
import eval_results
from plots_2D import multiseries_scatter, multiseries_contour, radial_plot, violin_plot, box_plot, bar_plot, fix_legend
from plots_2D import kaplan_meier, sliding_window_percentile, COLOURS


ECG_METRICS = ['SL_criteria_value', 'RR_mean', 'Pdur_mean', 'PR_mean', 'JTc', # 'QTp_mean',
               'JTp_mean', 'TpTe_mean', 'QTcB_mean', 'T_axis', 'P_axis', 'QRS_axis',
               'QRSdur_mean', 'TWI_presence', 'Giant_TWI', 'LBBB', 'RBBB', 'Fragmented_QRS', 'AF_ECG',]
BASELINE_CHARS = ['SBP', 'DBP', 'age1', 'BMI1', 'sex', 'Morphology_1_2', 'Morphology', 'obesity', 'cad',
                  'hypertension', 'ethnicity_white', 'hxhcm1st', 'hxscd1st', 'hxsync', 'blcp', 'bldysp', 'BP_meds',
                  'heart_failure', 'hxvt', 'hxscd', 'hxafib', 'hxstrk', 'hf_any', 'af_any', 'vt_any', 'stroke_any',
                  'diabetes', 'fhx_hcm_any', 'NTproBNP', 'log_NTproBNP', 'TnTStat', 'log_TnTStat', 'log_LVOT_delta',
                  'ESC_SCD_risk_score', 'glsmean', 'glsendo', 'glsepi', 'origin', 'M_isolated_basal_septal',
                  'M_asymmetric_septal_reverse_curve', 'M_apical', 'M_concentric', 'M_apical_aneurysm', 'M_other',
                  'blbeta', 'blcalc', 'blacearb', 'bldisop', 'blaldo', 'blsota', 'blamio', 'av_hr', 'HR', 'anti_HTN',
                  ]
IMAGING_CHARS = ['lvsvi', 'log_lge_total6', 'wallthkmax', 'lge_pres', 'ecvfwhole', 'log_ecvfwhole', 'lv_ef',
                 'low_lvef', 'BNP_group', 'lvmassi', 'echoprv', 'log_echoprv', 'echoprv_group', 'echorest', 'log_echorest',
                 'echorest_group', 'max_LVOT_grad', 'log_max_LVOT_grad', 'max_LVOT_grad_group', 'LVOT_delta', 'rvef',
                 'lavi', 'log_lavi', 'lav', 'log_lav', 'MR_severity', 'echosam', 'lvedvi', 'lvesvi', 'LVOT_any_50',
                 'LVOT_any_30', 'LVOT_both_30', 'cardiac_index']
OUTCOMES = ['hf_hosp', 'death', 'death_hf', 'death_cardiac', 'death_stroke', 'SCD', 'afib', 'vt',
            'vt_shock', 'transplant', 'stroke']  # 'death_noncardiac'

CIRC_VARS = ['T axis', 'P axis', 'QRS axis']

SERIES = {1: 'UKBB normal BP',
          2: 'UKBB high BP',
          4: 'HCMR sarcomere positive',
          3: 'HCMR sarcomere negative',
          5: 'HCMR sarcomere negative hypertension',
          6: 'HCMR sarcomere negative no hypertension',
          7: 'HCMR sarcomere positive hypertension',
          8: 'HCMR sarcomere positive no hypertension',
          }

CATEGORICAL_CUTOFF = 10


def main():
    subsets = [[5],[6],[7],[8]]       # list of lists; members of an inner list will be grouped, outer lists will be compared
    use_imputed_LVOT = False     # if True use datafile where blank LVOT values are imputed as (mean LVOT where LVOT<=30)
    lvot_type = 'log_max_LVOT_grad'   # any of ['log_max_LVOT_grad', 'log_echoprv', 'log_echomax']
    val_type = 'y_pred'         # 'y_pred': use transformed HTN scores, 'logits': use raw logit values
    logits_cap = 5, False       # number of standard deviations to cap at (int), include capped outliers (T/F)
    show_ims = False
    group_byscore = False         # into Group 1 and Group 2
    merge_SN_BP = False         # merges SN-HCM hypertension and SN-HCM no hypertension (if saved data is split)
    split_by_BP = True           # splits SN-HCM hypertension and SN-HCM no hypertension
    adjust_BP = 0               # adds this amount to systolic and diastolic BP for patients taking antihypertensives
    bw_adjust = 1.             # bandwidth adjustment for kde contour plot. Set to 0 to skip contour plot
    contour_threshold = 0.6     # threshold for lowest contour on contour plot

    # NEW FORMAT
    preds_folder = 'G:\\results\\Significant results\\'
    preds_subfolder = "2024-03-05 final\\"
    # preds_folder = '\\\\?\\G:\\results\\CNN general\hypertension\\k-fold\\'
    # preds_subfolder = "2024-04-06 19.13.13 hypertension 5 30 50 0 32x7-32x7-64x5-64x3-128x3-128x3-256x3-256x3-128-32- train_on all\\"
    # preds_subfolder = '2024-04-25 00.34.55 hypertension 5 30 50 0 32x7-32x7-64x5-64x3-128x3-128x3-256x3-256x3-128-32- train_on UKBB\\'
    # preds_subfolder = '2024-04-04 22.04.41 hypertension 5 150 50 0 32x7-32x7-64x5-64x3-128x3-128x3-256x3-256x3-128-32- train_on UKBB\\'
    # preds_subfolder = '2024-09-26 02.19.03 hypertension 5 50 50 0 32x7-32x7-64x5-64x3-128x3-128x3-256x3-256x3-128-32- train_on UKBB+SN\\'
    preds_file = 'preds_logits.csv'

    test_vars = BASELINE_CHARS + IMAGING_CHARS + OUTCOMES + ECG_METRICS
    folder = preds_folder + preds_subfolder
    output_folder = (preds_folder + preds_subfolder + f'{val_type} analysis BP adjust={adjust_BP}'
                     + split_by_BP*' split_by_BP' + merge_SN_BP*' merge_SNBP' + use_imputed_LVOT*' impute_LVOT' + '\\')
    my_tools.make_dir(output_folder)

    analyse_results(folder, preds_file, test_vars, val_type, logits_cap,
                    output_folder, bw_adjust, contour_threshold,
                    show_ims=show_ims, subsets=subsets, adjust_BP=adjust_BP, group_by_score=group_byscore,
                    merge_SN_BP=merge_SN_BP, split_by_BP=split_by_BP, use_imputed_LVOT=use_imputed_LVOT,
                    lvot_type=lvot_type)


def analyse_results(folder, preds_file, test_vars, val_type, cap, output_folder, bw_adjust=1., contour_threshold=0.5,
                    adjust_BP=0, show_ims=False, subsets=(), group_by_score=False, merge_SN_BP=False, split_by_BP=False,
                    use_imputed_LVOT=False, lvot_type='log_max_LVOT_grad'):
    # load predictions and labels
    joined_data = load_all(folder, output_folder, preds_file, test_vars, cap,
                           adjust_BP, use_imputed_LVOT=use_imputed_LVOT)

    # combine/split SN-HTN and SN-normal
    joined_data = preprocess_SN_BP(joined_data, merge_SN_BP, split_by_BP)

    # Group distributions
    box_plot(joined_data, val_type, palette=COLOURS, x='series', rotate_labels=40, xfontsize=10,
             save_to=output_folder + f'_{val_type}_all groups box plot.svg')
    violin_plot(joined_data, [SERIES[i] for i in [1,2,4,3]], 'series', val_type, palette=COLOURS,
                save_to=output_folder + f'_{val_type}_all groups violin plot.svg')

    # plot BP_meds status on BP vs HTN score plot
    # bp_meds_plot(joined_data, output_folder)

    # histograms and QQ of preds, specific linear models, ROC curves
    if not split_by_BP:
        linear_BP_models(joined_data, val_type, output_folder + f'_linear model_')
        covs = ['age1', 'sex', 'BMI1', 'SBP', lvot_type, 'log_NTproBNP', 'glsmean', 'wallthkmax', 'log_lge_total6'] # 'lvmassi',
        for series_name in subsets:
            series_list = [SERIES[x] for x in series_name]
            ser = joined_data[joined_data['series'].isin(series_list)]
            hist_qq(ser[val_type], save_to=output_folder + f'_{val_type}, subset={abbreviate([series_list])} - ', show_ims=show_ims)
            htn_score_ols(ser, val_type, covs, normalise=True,
                          save_to=output_folder + f'_ECGSI OLS {covs}, subset={abbreviate([series_list])}.txt')

        # ROC analysis
        # roc(joined_data, val_type, output_folder + 'roc analysis\\')

        special_tests(joined_data, val_type, output_folder, lvot_type)

    # analyse each variable
    all_results, all_split_results = [], []
    for v in test_vars:
        print(f'Analysing {v}')

        # Main comparisons
        res = analyse_var(joined_data, val_type, v, 'ECG_SI', bw_adjust,
                          contour_threshold, subsets=subsets, group_byscore=group_by_score, show_ims=show_ims,
                          save_to=output_folder)
        all_results += res

        # Group 1 - Group 2 analyses
        series_to_group = 'SN'
        other_series_to_compare = []    #   ['HCMR sarcomere positive']
        if group_by_score:
            split_res_series = [series_to_group + '_Group1', series_to_group + '_Group2'] + other_series_to_compare
            split_res = analyse_split_grps(joined_data, v, split_res_series, series_to_group,
                                           show_ims=show_ims,
                                           save_to=output_folder + f'{v}, ')
            all_split_results += split_res

        # HCMR HTN - no HTN analyses
        if split_by_BP:
            split_res = analyse_split_BP(joined_data, v, subsets,
                                           show_ims=show_ims, save_to=output_folder + f'{v}, ')
            all_split_results += split_res

    abbr_ser = [[abbreviate([[SERIES[x]]])[0][0] for x in ser_name] for ser_name in subsets]
    save_results(all_results, output_folder + f'_ECGSI correlations, subset={abbr_ser}.csv')
    split_fname = (split_by_BP*f'HTN-noHTN split correlations, subset={abbr_ser}.csv' +
                   group_by_score*f'_{series_to_group} Group 1-2 split correlations, subset={abbr_ser}.csv')
    if all_split_results: save_results(all_split_results, split_fname)


def analyse_var(data, val_type, x_var, y_var, bw_adjust=1., contour_threshold=0.6,
                subsets=(), group_byscore=False, show_ims=False, save_to=''):
    if save_to !='': my_tools.make_dir(save_to)

    # select columns, rename, drop NaNs, split into subsets
    data_var = data[list({'series', val_type, x_var, 'age1', 'BMI1', 'sex'})].dropna()

    # OLS
    res = []
    for s in subsets:
        ser = [SERIES[x] for x in s]
        for covs in [[x_var], [x_var, 'BMI1', 'age1'], [x_var, 'BMI1', 'age1', 'sex']]:
            res1 = htn_score_ols(data_var, val_type, covs, series_list=ser,
                                 save_to=save_to + f'{x_var}, subsets={abbreviate_list([[ser]])}, covs={covs} - MV regression.txt')
            if res1 is not None:
                res.append([x_var, ser, f'LinR {covs}', res1.params[1], res1.pvalues[1], f'grad = {res1.params[1]:.5f}'])

    # analysis and plotting
    data_var = data_var.rename(columns={val_type: y_var})
    subset_data = split_subsets(data_var, subsets)
    data_len = len(set(data_var[x_var]))

    # continuous/ordinal variables
    if data_len > CATEGORICAL_CUTOFF:
        results = analyse_continuous(data_var, group_byscore, save_to, show_ims,
                                     subset_data, x_var, y_var,
                                     bw_adjust=bw_adjust, contour_threshold=contour_threshold)
        results += res

    # categorical variables
    elif 2 < data_len <= CATEGORICAL_CUTOFF:
        results = analyse_categorical(subset_data, x_var, y_var, show_ims,
                                      save_to=save_to)

    # binary variables
    else:
        results = analyse_binary(subset_data, x_var, x_label=y_var, show_ims=show_ims,
                                 save_to=save_to)

    return results


def analyse_continuous(data_var, group_byscore, save_to, show_ims, subset_data, x_var, y_var, bw_adjust=1., contour_threshold=0.5):
    stack_plots = False

    df = data_var.dropna(subset=[x_var, y_var])
    include = [x[0][0] for x in subset_data]
    series_list = sorted_series_list(df, include)
    if bw_adjust > 0:
        multiseries_contour(df, x_var, y_var, series_list,
                               bw_adjust=bw_adjust, contour_threshold=contour_threshold,
                               y_lim=[0,1],
                               show_ims=show_ims,
                               save_to=save_to + f'{x_var}, subsets={abbreviate_list(subset_data)}')
    sliding_window_percentile(subset_data, x_var, y_var, width=0.5, show_ims=show_ims,
                              y_lim=(0.35, 0.6), stack_plots=stack_plots,
                              save_to=save_to + f'{x_var}, subsets={abbreviate_list(subset_data)}'
                                                f' - {"stacked "*stack_plots}sliding window plot.svg')
    results = multiseries_scatter(subset_data, x_var, y_var, show_ims=show_ims,
                                  save_file=save_to + f'{x_var}, subsets={abbreviate_list(subset_data)} - scatter plot.svg')

    if group_byscore:
        plot_data = df[df['series'].isin(['HCMR sarcomere negative', 'HCMR sarcomere negative hypertension',
                                                      'HCMR sarcomere negative no hypertension'])]
        plot_split(plot_data, x_var, y_var, show_ims,
                   save_to=save_to + f'{x_var}, Group 1-Group 2 split scatter plot.svg')
    return results


def analyse_categorical(subset_data, x_var, y_var, show_ims=True, save_to=''):
    results = []
    for series_name, data in subset_data:
        df = data.dropna(subset=x_var)
        cat_list = sorted(list(set(df[x_var])))
        cat_data = [df[df[x_var]==cat][y_var] for cat in cat_list]
        means = '\n'.join([f'{int(cat_list[i])}: {cat_data[i].median():.3f}' for i in range(len(cat_list))])
        # means = [cat_data[i].mean() for i in range(len(cat_list))]
        stat, pval = kruskal(*cat_data)
        results.append([x_var, series_name, 'Kruskal-Wallis', stat, pval, means])

        # preds known to be non-normal so omit test
        # stat, pval = f_oneway(*cat_data)
        # results.append([x_var, series_name, '1-way ANOVA', stat, pval, means])

        violin_plot(df, cat_list, x_var, y_var, show_ims=show_ims, palette=COLOURS,
                    save_to=save_to + f'{x_var}, subset={abbreviate([series_name])} - violin plot.svg')
        box_plot(df, y_var, stat_test=None, show_ims=show_ims, palette=COLOURS,
                 save_to=save_to + f'{x_var}, subset={abbreviate([series_name])} - box plot.svg', x=x_var, hue='series')
    return results


def analyse_binary(data, var, x_label, show_ims=False, save_to=''):
    result_list = []
    for d in data:
        series_name, var_vals = d[0], sorted(list(set(d[1][var])))
        if len(var_vals) == 2:
            violin_plot(d[1], var_vals, var, x_label, stat_test=None, show_ims=show_ims, palette=COLOURS,
                        save_to=save_to + f'{var}, subset={abbreviate([series_name])} - violin plot.svg')
            box_plot(d[1], x_label, x=var, stat_test=None, show_ims=show_ims,
                     save_to=save_to + f'{var}, subset={abbreviate([series_name])} - box plot.svg')
            d[1] = d[1].drop(columns=['series'])
            groups = [d[1][d[1][var]==i][x_label] for i in var_vals]
            if len(groups[0]) > 1 and len(groups[1]) > 1:
                # trainer.density_1d(groups, [f'{var} = {int(i)}' for i in range(2)],
                #                    x_label, normalise=False, var_name=f'{var} in {series_name}',
                #                    show_ims=show_ims, save_to=save_to + f'{var}, subset={abbreviate([series_name])}')

                means = f'0: {groups[0].median():.3f}\n1: {groups[1].median():.3f}'
                # means = [groups[0].mean(), groups[1].mean()]
                mw = mannwhitneyu(groups[0], groups[1])
                result_list.append([var, series_name, 'Mann Whitney U', mw.statistic, mw.pvalue, means])

                # preds known to be non-normal so omit these tests
                # tt = ttest_ind(groups[0], groups[1])
                # result_list.append([var, series_name, 't-test', tt.statistic, tt.pvalue, means])
                # shap_stat, shap_p = shapiro(pd.concat(groups, axis=0))
                # result_list.append([var, series_name, 'Shapiro-Wilk', shap_stat, shap_p, ''])
    return result_list


def analyse_split_grps(data, v, series, series_to_group, show_ims=True, save_to=''):
    output = []
    df = data.copy()
    df = group_by_score(df, series_to_group)
    df = df[df['series'].isin(series)]
    df = df.dropna(subset=v)
    df[v] = df[v].astype(float)

    two_group_analysis(df, v, series, output, show_ims, save_to)
    return output


def analyse_split_BP(data, v, subsets, show_ims=False, save_to=''):
    for s in subsets:
        ser = [SERIES[x] for x in s]
        output = []
        df = data.copy()
        df = df[df['series'].isin(ser)]
        df = df.dropna(subset=[v, 'hypertension'])
        df['series'] = df['hypertension'].replace({0:'No hypertension', 1:'Hypertension'})
        df[v] = df[v].astype(float)

        two_group_analysis(df, v, ['No hypertension', 'Hypertension'], output,
                           show_ims, save_to)
        return output


def two_group_analysis(df, v, series, output, show_ims, save_to):
    # continuous/ordinal variables
    box_plot(df, v, stat_test=None,
             show_ims=show_ims, save_to=save_to + 'box plot.svg', x='series', order=series)
    violin_plot(df, series, 'series', v, save_to=save_to + 'vln plot.svg', show_ims=show_ims)
    plots_2D.histo_plot(df, 'series', v, save_to=save_to + f'{series} histogram.svg', show_ims=show_ims)
    for i in range(len(series) - 1):
        for j in range(i + 1, len(series)):  # pairwise comparisons
            # logistic regression model adjusted for age, BMI, sex
            if len(set(df[v])) > 1:
                df['group'] = df['series'].apply(grp_to_number(series[i], series[j], grp1_val=0))
                formula = f"group ~ {v} + sex + age1 + BMI1"
                caption = f'LogR {formula} ({series[i]}=0 {series[j]}=+1)'
                log_reg = smf.logit(formula, data=df).fit()
                if save_to != '':
                    with open(save_to + caption + '.txt', 'w') as f:
                        print(log_reg.summary2(), file=f)

                # get summary stats
                summ_stats = ''
                d_type = get_fields()[v]
                for idx in [i, j]:
                    ss = get_summ_stats(df[df['series'] == series[idx]][v], d_type)
                    summ_stats += f'{series[idx]}: {[f"{s:.2f}" for s in ss]}\n'

                output.append([v, f'{series[i]}, {series[j]}', 'Logistic regression', log_reg.params[1],
                               log_reg.pvalues[1], f'beta = {log_reg.params[1]:.3f}\n{summ_stats}'])
    if v in OUTCOMES and sum(df[v]) > 0:
        res = kaplan_meier(df, v, series, save_to=save_to, show_ims=show_ims)
        output.append(res)


def plot_split(data, x_label, y_label, show_ims, save_to=''):
    groups = [data[data[y_label]>0.5], data[data[y_label]<=0.5]]
    fig = plt.figure(figsize=(24, 12))
    cols = ['r', 'b']
    handles, rs, ps = [], [], []
    for i in range(2):
        x, y, = groups[i][x_label], groups[i][y_label]
        r, p = pearsonr(x, y)
        rs.append(r)
        ps.append(p)
        plt.scatter(x, y, marker='o', s=8, c=cols[i])
        h = plt.plot(np.unique(x),
                     np.poly1d(np.polyfit(x.astype(float), y.astype(float), 1))(np.unique(x)),
                     cols[i]+'-', label='Group'+str(i+1))
        handles.append(h[0])
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(x_label)
    fix_legend(handles=handles)
    if show_ims: plt.show()
    if save_to != '': plt.savefig(save_to)
    plt.close()

    return rs, ps


def circ_ww_test(data, var, show_ims=False, save_to=''):
    series = list(set(data['series']))
    series_data = [data[data['series']==s][var]*2*math.pi/360 for s in series]  # convert to radians
    radial_plot(series_data, series, show_ims=show_ims, save_to=save_to)
    w = [len(s) for s in series_data]
    pval, table = watson_williams(*series_data, w=w, axis=0)
    results = [var, data, 'Watson Williams F-test', table, pval]
    return results


def linear_BP_models(df, values, folder, normalise=True):
    """ model BP on hypertension score and grouping (UKBB-no-HTN vs SN-HCM) """
    df['DBP'] = df['DBP'].astype(float)

    data = df.copy()
    if normalise:
        for c in ['DBP', 'age1', 'BMI1', 'log_max_LVOT_grad']:
            data[c] = data[[c]].dropna().astype(float).apply(zscore)

    tests = [
        [['UKBB normal BP'], ['UKBB high BP'], f'{values} ~ group + age1 + BMI1 + sex', 'UKBB-N vs UKBB-H'],
        [['UKBB high BP'], ['HCMR sarcomere negative'], f'DBP ~ group + age1 + BMI1 + sex', 'DBP in UKBB-H vs SN'],
        [['UKBB high BP'], ['HCMR sarcomere negative'], f'SBP ~ group + age1 + BMI1 + sex', 'SBP in UKBB-H vs SN'],
        [['UKBB normal BP'], ['HCMR sarcomere positive'], f'{values} ~ group + age1 + BMI1 + sex', 'UKBB-N vs SP'],
        [['UKBB normal BP'], ['HCMR sarcomere negative'], f'{values} ~ group + age1 + BMI1 + sex', 'UKBB-N vs SN'],
        [['UKBB high BP'], ['HCMR sarcomere negative'], f'{values} ~ group + age1 + BMI1 + sex', 'UKBB-H vs SN'],
        [['UKBB normal BP', 'UKBB high BP'], ['HCMR sarcomere negative'], f'{values} ~ DBP * group + age1 + sex+ BMI1', 'UKBB vs SN'],
        [['UKBB normal BP', 'UKBB high BP'], ['HCMR sarcomere positive'], f'{values} ~ DBP * group + age1 + sex + BMI1', 'UKBB vs SP'],
        [['UKBB normal BP', 'UKBB high BP'], ['HCMR sarcomere positive', 'HCMR sarcomere negative'], f'{values} ~ DBP * group + age1 + BMI1 + sex', 'UKBB vs all HCM'],
        [['HCMR sarcomere negative'], ['HCMR sarcomere positive'], f'{values} ~ log_max_LVOT_grad * group', 'SN vs SP'],
        [['HCMR sarcomere negative'], ['HCMR sarcomere positive'], f'{values} ~ log_max_LVOT_grad + age1 + BMI1 + sex + log_TnTStat * group', 'SN vs SP']
    ]
    for t in tests:
        res = group_mod(data, folder, *t)
        print(f'{t}\nbeta = {res.params[1]}, p = {res.pvalues[1]:.6f}')

    # compare UKBB and HCMR
    data['UKBBvsHCMR'] = data['series'].apply(grp_to_number(['UKBB normal BP', 'UKBB high BP'], ['HCMR sarcomere positive', 'HCMR sarcomere negative']))
    formula = f'{values} ~ UKBBvsHCMR*sex + UKBBvsHCMR*age1 + UKBBvsHCMR*BMI1 + UKBBvsHCMR*DBP'
    mod = smf.ols(formula=formula, data=data)
    result = mod.fit()
    with open('_'.join([folder, formula.replace('*', 'x'), 'UKBB vs HCMR.txt']), 'w') as f:
        print(result.summary2(), file=f)


def grp_to_number(grp1, grp2, grp1_val=-1, grp2_val=1):
    def convert(x):
        if x in grp1:
            return grp1_val
        if x in grp2:
            return grp2_val
    return convert


def group_mod(data, folder, grp1, grp2, formula, caption, grp_field='series', vars_to_zscore=()):
    data_sub = data[data[grp_field].isin(grp1 + grp2)]
    data_sub['group'] = data_sub[grp_field].apply(grp_to_number(grp1, grp2))
    for c in vars_to_zscore:
        data_sub[c] = data_sub[[c]].dropna().astype(float).apply(zscore)
    mod = smf.ols(formula=formula, data=data_sub)
    result = mod.fit()
    with open('_'.join([folder, caption + '.txt']), 'w') as f:  # formula.replace('*', 'x'),
        print(result.summary2(), file=f)
    return result


def htn_score_ols(data, values, covs, series_list=None, normalise=True, save_to=''):
    data_sub = data.copy()
    if series_list is not None:
        data_sub = data[data['series'].isin(series_list)]
    data_sub = data_sub[[values]+covs]
    if (len(set(data_sub[covs[0]])) > 1) and not data_sub[covs].isnull().all(axis=0).any():
        if normalise:
            for c in covs:
                data_sub[c] = data_sub[[c]].dropna().astype(float).apply(zscore)
        mod = smf.ols(formula=f'{values} ~ '+"+".join(covs), data=data_sub)
        result = mod.fit()
        if save_to!='':
            with open(save_to, 'w') as f:
                print(result.summary2(), file=f)
        return result


def split_subsets(data, subsets=((1,), (2,), (3,), (4,), (5,), (6,))):
    output = []
    for s in subsets:
        selection = [SERIES[x] for x in s]
        df = data[data['series'].isin(selection)]
        if len(df) != 0:
            output.append([selection, df])
    return output


def load_all(folder, output_folder, preds_file, test_vars, cap, adjust_BP=0, use_imputed_LVOT=False):
    fname = p.label_files['combined'][:-4] + use_imputed_LVOT*' (LVOT imputed)' + '.csv'
    labels = pd.read_csv(fname, index_col=0, dtype=label_dtypes)
    labels = labels.loc[:, test_vars]
    labels = fix_labels(labels, adjust_BP)
    data = pd.read_csv(folder + preds_file, index_col=0)

    # set data types and merge
    data.index, labels.index = data.index.map(str), labels.index.map(str)
    joined_data = data.join(labels)

    # cap logits
    joined_data['logits'] = cap_logits(joined_data, cap, output_folder)

    return joined_data


def cap_logits(df, cap, folder):
    data = df.copy()
    val, inc = cap
    logits = data['logits']
    print(inc*f'Capping logits at +/- {val}' + ~inc*f'Excluding logits +/- {val}')
    hi = pd.DataFrame(data[data['logits'] > val][['series', 'logits']], index=data[data['logits'] > val].index)
    lo = pd.DataFrame(data[data['logits'] < -val][['series', 'logits']], index=data[data['logits'] < -val].index)
    hi.sort_values(by='logits', ascending=False).to_csv(folder + f'_Logits greater than {val}.csv')
    lo.sort_values(by='logits', ascending=False).to_csv(folder + f'_Logits less than {-val}.csv')
    if inc:
        data['logits'] = np.where((logits <-val), -val, logits)
        data['logits'] = np.where((logits > val), val, logits)
    else:
        data['logits'] = np.where((logits<-val) | (logits>val), np.nan, logits)
    return data['logits']


def merge_htn(joined_data):
    joined_data = joined_data.replace(
        ['HCMR sarcomere negative no hypertension', 'HCMR sarcomere negative hypertension'], 'HCMR sarcomere negative')
    return joined_data


def detect_htn(preds):
    """ splits "HCMR sarcomere negative" data into hypertensive and normotensive
    requires preds is a df with column called series and a column called hypertension
    if hypertension == 1 then renames series value from 'HCMR sarcomere negative' to 'HCMR sarcomere negative high BP'
    and similar if hypertension == 0 to 'HCMR sarcomere negative normal BP'
    """
    def substitute(row):
        s = row['series']   # e.g. 'HCMR sarcomere negative'
        if s[:4] == 'HCMR':
            if row['hypertension']==1:
                return s + ' hypertension'
            else:
                return s + ' no hypertension'
        else:
            return row['series']

    preds['series'] = preds.apply(substitute, axis=1)
    return preds


def save_results(all_results, output_file):
    all_results = pd.DataFrame(all_results, columns=['var', 'series', 'test', 'stat', 'p value', 'effect size'])
    all_results['series'] = all_results['series'].astype(str)
    all_results = all_results.pivot_table(index=['var', 'test'], columns='series', values=['effect size', 'p value'], aggfunc='first')
    all_results.to_csv(output_file)


def fix_labels(data, adjust_BP=0):
    # adjust all angles to 0-360
    def f(angle):
        if pd.notnull(angle):
            try:
                a = float(angle) % 360
                if a >= 270: a -= 360
                return a
            except:
                return np.nan
        else:
            return angle
    for ax in ['P axis', 'QRS axis', 'T axis']:
        if ax in data.columns:
            data[ax] = data[ax].apply(f)

    # adjust BP measurements
    if adjust_BP != 0:
        data['SBP'] = np.where(data['origin']==1, data['SBP'] + data['BP_meds'] * adjust_BP, data['SBP'])
        data['DBP'] = np.where(data['origin']==1, data['DBP'] + data['BP_meds'] * adjust_BP, data['DBP'])
    return data


def group_by_score(data, series_to_group, cutoff=0.5, verbose=False):
    """ splits SN-HCM subgroup into Group 1 and Group 2 based on hypertension score
    data: dataframe with column 'series' containing series names of which some will be "HCMR sarcomere negative"
    and another column 'y_true' containing hypertension scores
    cutoff: the value for the hypertension score used to split into Group 1 and Group 2
    :returns data, but with series name changed to Group 1 or Group 2 for all SN-HCM
    """
    s = {'SP': 'HCMR sarcomere positive', 'SN': 'HCMR sarcomere negative',}[series_to_group]
    df = data.copy()
    df['series'] = np.where((df['series']==s) & (df['y_pred']>cutoff),
                              series_to_group + '_Group1', df['series'])
    df['series'] = np.where((df['series']==s) & (df['y_pred'] <= cutoff),
                              series_to_group + '_Group2', df['series'])
    if verbose:
        print(f"{sum(df['series']==(series_to_group + '_Group1'))} in Group 1")
        print(f"{sum(df['series']==(series_to_group + '_Group2'))} in Group 2")
    return df


def roc(df, val_type, output_folder):
    data = df.dropna(subset=[val_type, 'hypertension'])
    for s in [['UKBB normal BP', 'UKBB high BP'], ['HCMR sarcomere negative', 'HCMR sarcomere positive'],
              ['HCMR sarcomere negative'], ['HCMR sarcomere positive']]:
        subset = data[data['series'].isin(s)]
        y_true, y_pred = subset['hypertension'], subset[val_type]
        eval_results.eval_clas(subset.index.tolist(), y_pred, y_true,
                               save_to=output_folder + str(s) + '_')


def preprocess_SN_BP(joined_data, merge_SN_BP, split_by_BP):
    if split_by_BP:
        joined_data = detect_htn(joined_data)
    if merge_SN_BP:
        joined_data = merge_htn(joined_data)
    return joined_data


def sorted_series_list(data, series_list=None):
    if series_list is None:
        series_list = list(np.unique(data['series'].tolist()))

    series_lookup = [SERIES[i] for i in range(1, 9)]
    series_list.sort(key=lambda x: series_lookup.index(x))
    return series_list


def abbreviate_list(split_data):
    group_list = [d[0] for d in split_data]
    l = abbreviate(group_list)
    return list(l)


def debug_plot(data):
    data = data[data['series'] == 'HCMR sarcomere negative']
    sns.kdeplot(data=data, x='y_true', bw_adjust=1., fill=True, legend=False, alpha=.3, linewidth=0)
    plt.xlim(0,1)


def test_true_hcm(df):
    test = df.copy()
    test['high_ss'] = test['y_pred'] > 0.5
    for grp, ss in [['HCMR sarcomere positive', 0], ['HCMR sarcomere positive', 1],
                    ['HCMR sarcomere negative', 0], ['HCMR sarcomere negative', 1]]:
        sub = test[(test['series']==grp) & (test['high_ss']==ss)]
        print(f'{grp}, high ECG-SS={ss}: Not true HCM: {len(sub[sub["true_hcm"]==0])}, '
              f'Apical Morphology: {len(sub[sub["Morphology"]==3])}, '
              f'Not True HCM and Apical Morphology: {len(sub[(sub["Morphology"]==3) & (sub["true_hcm"]==0)])}')


def special_tests(df, val_type, folder, lvot_type):
    # plot troponin SP-HCM
    data = df[list({'series', val_type, 'DBP', 'sex', 'age1', 'log_lge_total6', 'log_TnTStat',
                    'BMI1', 'log_NTproBNP', lvot_type, 'wallthkmax'})].dropna()
    spm = data[(data['series']=='HCMR sarcomere positive') & (data['sex']==1)]
    spf = data[(data['series'] == 'HCMR sarcomere positive') & (data['sex'] == 0)]
    subset_data = [[[0], spf], [[1], spm]]
    sliding_window_percentile(subset_data, 'log_TnTStat', val_type, width=0.5, show_ims=False,
                              y_lim=(0.35, 0.6), stack_plots=False,
                              save_to=f"{folder}_log_TnTStat, SP-HCM by sex, sliding window plot.svg")

    # model sex:Troponin interaction effect
    for ser in [['HCMR sarcomere positive'], ['HCMR sarcomere negative'], ['HCMR sarcomere positive', 'HCMR sarcomere negative']]:
        sub_data = data[data['series'].isin(ser)]
        group_mod(sub_data,
                  folder,
                  [0], [1],
                  f'{val_type}~group*log_TnTStat + age1 + BMI1 + log_NTproBNP + {lvot_type} + wallthkmax'
                  f' + log_lge_total6',
                  f' {plots_2D.abbreviate(ser)} F vs M',
                  grp_field='sex',
                  vars_to_zscore=('age1', 'BMI1', 'log_NTproBNP', lvot_type, 'wallthkmax', 'log_lge_total6',
                                  'log_TnTStat'))

    # MODEL 1: multivariable with SN-SP interaction term
    for bp in ['DBP', 'SBP']:
        covs = ['age1', 'sex', 'BMI1']
        interests = [bp, lvot_type, 'log_NTproBNP', 'glsmean', 'wallthkmax', 'log_lge_total6'] # 'lvmassi',
        var_subset = covs + interests
        f = (f'{val_type}~group*{bp} + age1 + sex + BMI1 + group*{lvot_type} + group*log_NTproBNP + group*glsmean'
             f'+ group*wallthkmax + group*log_lge_total6')  # + group*lvmassi
        data = df[['series', val_type] + var_subset].dropna()
        res = group_mod(data, folder,
                  ['HCMR sarcomere negative'], ['HCMR sarcomere positive'],
                  f,
                  f' {var_subset} SN vs SP',
                  vars_to_zscore=var_subset)
        run_contrasts(interests, res, 'SN-HCM', 'SP-HCM',
                    f'{folder}_ {var_subset} SN vs SP contrasts.txt')
        plots_2D.visualise_lr(interests, res, save_to=folder+'_LR coefs+interactions bar plot.svg')

    # MODEL 2: ECGSI ~ (UKBB vs HCMR) * BP(+ age + sex + BMI)
    data = df[['series', val_type, 'DBP', 'age1', 'sex', 'BMI1', 'hypertension']].dropna()
    res = group_mod(data, folder,
                    ['UKBB high BP', 'UKBB normal BP'], ['HCMR sarcomere negative', 'HCMR sarcomere positive'],
                    f'{val_type}~group*DBP +age1+sex+BMI1',
                    f' (UKBB vs HCMR) x DBP',
                    vars_to_zscore=('DBP', 'age1', 'BMI1'))
    run_contrasts(['DBP'], res, 'UKBB', 'HCMR',
                f'{folder}_ (UKBB vs HCMR) x DBP contrasts.txt')

    # MODEL 3: ECGSI ~ (SN vs BBHTN) * BP(+ age + sex + BMI)
    res = group_mod(data, folder,
                    ['HCMR sarcomere negative'], ['UKBB high BP'],
                    f'{val_type}~group*DBP +age1+sex+BMI1',
                    f' (SN vs UKB-H) x DBP',
                    vars_to_zscore=('DBP', 'age1', 'BMI1'))
    # MODEL 4: ECGSI ~ (SP vs BB-N) * BP (+ age +sex + BMI)
    res = group_mod(data, folder,
                    ['HCMR sarcomere positive'], ['UKBB normal BP'],
                    f'{val_type}~group*DBP +age1+sex+BMI1',
                    f' (SP vs UKB-N) x DBP',
                    vars_to_zscore=('DBP', 'age1', 'BMI1'))

    res = group_mod(data, folder,
                    ['HCMR sarcomere negative'], ['HCMR sarcomere positive'],
                    f'{val_type}~group*DBP + group*hypertension +age1+sex+BMI1',
                    f' (SN vs SP) x DBP + (SN vs SP) x hypertension',
                    vars_to_zscore=('DBP', 'age1', 'BMI1'))

    # MODEL 5: ECG-SI ~ HTN x BP (+ LVOT x BP)
    for s in [['HCMR sarcomere negative'], ['HCMR sarcomere positive'],
              ['HCMR sarcomere negative', 'HCMR sarcomere positive'],
              #['UKBB high BP', 'UKBB normal BP'],
              ]:
        for bp in ['DBP', 'SBP']:
            covs = ['age1', 'BMI1', 'sex', 'hypertension', bp, lvot_type]
            data = df[['series', val_type] + covs]
            data = data[data['series'].isin(s)]
            for c in ['age1', 'BMI1', bp,]: # 'log_max_LVOT_grad']:
                data[c] = data[[c]].dropna().astype(float).apply(zscore)
            for c in [[], ['age1', 'sex', 'BMI1']]:
                c_str = (c!=[])*' + ' + ' + '.join(c)
                data_sub = data[[val_type, 'hypertension', bp, lvot_type] + c].dropna()# modify when needed
                f = f'{val_type}~hypertension*{bp} + {lvot_type}' + c_str                  # modify when needed
                mod = smf.ols(formula=f, data=data_sub)
                result = mod.fit()
                s_name = abbreviate([s])[0]
                with open(folder + f'_ ({s_name}) HTN x {bp} + {lvot_type}{c_str}.txt', 'w') as fo:   # modify when needed
                    print(result.summary2(), file=fo)

    # lvmassi*wallthkmax interaction
    for bp in ['DBP', 'SBP']:
        covs = [bp, lvot_type, 'log_NTproBNP', 'glsmean', 'lvmassi', 'wallthkmax', 'log_lge_total6', 'age1', 'BMI1', 'sex',]
        data = df[['series', val_type] + covs].dropna()
        data = data[data['series'].isin(['HCMR sarcomere positive', 'HCMR sarcomere negative'])]
        for c in ['age1', 'BMI1', bp, lvot_type, 'log_NTproBNP', 'glsmean', 'lvmassi', 'wallthkmax', 'log_lge_total6']:
            data[c] = data[[c]].dropna().astype(float).apply(zscore)
        f = f'{val_type}~lvmassi*wallthkmax + ' + '+'.join(covs)
        mod = smf.ols(formula=f, data=data)
        result = mod.fit()
        with open(folder + f'_ECGSI OLS lvmassi x wallthkmax.txt', 'w') as fo:
            print(result.summary2(), file=fo)

    # ECG-SI ~ (UKBB vs HCMR) * ECG indices
    ecg_inds = ['RR_mean', 'Pdur_mean', 'PR_mean', 'QRSdur_mean', 'QTcB_mean'] # 'JTc', 'TpTe_mean']
    data = df[['series', val_type, 'age1', 'sex', 'BMI1'] + ecg_inds].dropna()
    f = (f'{val_type}~' + '+'.join([f'group*{x}' for x in ecg_inds]) + '+age1+sex+BMI1')
    res = group_mod(data, folder,
                    ['UKBB high BP', 'UKBB normal BP'], ['HCMR sarcomere negative', 'HCMR sarcomere positive'],
                    f,
                    f' (UKBB vs HCMR) x ECG indices',
                    vars_to_zscore=ecg_inds + ['age1', 'sex', 'BMI1'])
    run_contrasts(ecg_inds, res, 'UKBB', 'HCMR',
                  f'{folder}_ (UKBB vs HCMR) x ECG indices contrasts.txt')
    plots_2D.visualise_lr(ecg_inds, res, show_ims=False,
                          save_to=folder+'_ECG indices+interactions bar plot.svg')

    # histograms of GLS by group
    data = df[df['series'].isin(['HCMR sarcomere negative', 'HCMR sarcomere positive'])]
    sns.histplot(data, x='glsmean', hue='series', bins=30)
    plt.savefig('g:\\GLSmean histograms.svg')
    plt.close()

    data = data[['series', 'glsmean', 'log_NTproBNP']].dropna()
    subset_data = split_subsets(data, [[3], [4]])
    sliding_window_percentile(subset_data, 'glsmean', 'log_NTproBNP', width=0.3, save_to='g:\\GLS vs BNP.svg')


def run_contrasts(interests, res, grp1_name, grp2_name, output_file):
    contrasts = ''
    for v in interests:
        contrast_minus = res.t_test(f'{v} - group:{v} = 0')
        contrast_plus = res.t_test(f'{v} + group:{v} = 0')
        beta_min, t_min, p_min = contrast_minus.effect, contrast_minus.statistic, contrast_minus.pvalue
        beta_plus, t_plus, p_plus = contrast_plus.effect, contrast_plus.statistic, contrast_plus.pvalue
        contrasts += f'\n\n{v}\n'
        contrasts += f'{grp1_name}: effect = {beta_min[0]:.4f}, t = {t_min[0][0]:.4f}, p = {p_min.min():.4f}\n' \
                     f'{grp2_name}: effect = {beta_plus[0]:.4f}, t = {t_plus[0][0]:.4f}, p = {p_plus.min():.4f}'
    str_to_file(contrasts, output_file)


def str_to_file(s, f):
    fo = open(f, "w")
    fo.write(s)
    fo.close()


def analyse_split_grps_OLD(data, v, series, series_to_group, show_ims=True, save_to=''):
    output = []
    df = data.copy()
    df = group_by_score(df, series_to_group)
    df = df[df['series'].isin(series)]
    df = df.dropna(subset=v)
    df[v] = df[v].astype(float)

    # continuous/ordinal variables
    if len(set(df[v])) > CATEGORICAL_CUTOFF:
        box_plot(df, v, stat_test=None,
                 show_ims=show_ims, save_to=save_to + 'box plot.svg', x='series', order=series)
        violin_plot(df, series, 'series', v, save_to=save_to + 'vln plot.svg', show_ims=show_ims)
        for i in range(len(series)-1):
            for j in range(i+1, len(series)):   # pairwise comparisons
                mw = mannwhitneyu(df[df['series']==series[i]][v], df[df['series']==series[j]][v])
                ks = ks_2samp(df[df['series']==series[i]][v], df[df['series']==series[j]][v])
                tt = ttest_ind(df[df['series'] == series[i]][v], df[df['series'] == series[j]][v])
                medians = '\n'.join([f"{series[k]}: {df[df['series'] == series[k]][v].median():.3f}" for k in [i,j]])
                output.append([v, str([series[i], series[j]]), 'Mann Whitney U', mw.statistic, mw.pvalue, medians])
                output.append([v, str([series[i], series[j]]), 'KS test', ks.statistic, ks.pvalue, medians])
                output.append([v, str([series[i], series[j]]), 't-test', tt.statistic, tt.pvalue, medians])

                # linear model adjusted for age, BMI, sex
                res = group_mod(df, save_to+f' LR ', [series[0]], [series[1]],
                                f'{v} ~ group + age1 + BMI1 + sex', f'{v} ~ {series[0]} vs {series[1]}',
                                vars_to_zscore=['age1', 'BMI1', 'sex'])
                output.append([v, str(series), f'LR {v} ~ group + age1 + BMI1 + sex', res.params[1], res.pvalues[1], ''])
                # too many data - shap_p always tiny
                # shap_stat, shap_p = shapiro(df[df['series'].isin([series[i], series[j]])][v])
                # output.append([v, str([series[i], series[j]]), 'Shapiro-Wilk', shap_stat, shap_p, ''])

    # categorical variables
    else:
        res = chi2test([df[df['series'] == s][v] for s in series])
        ct = contingency_table([df[df['series'] == s][v] for s in series], return_dict=True)
        bar_plot(ct, series, p=None, show_ims=show_ims, save_to=save_to + 'bar chart.svg')  # p = res[1]
        ct = contingency_table([df[df['series'] == s][v] for s in series], return_dict=True, normalised=True)
        bar_plot(ct, series, p=None, show_ims=show_ims, save_to=save_to + 'normalised bar chart.svg')
        output.append([v, series, 'Chi-square test', res[2], res[1], ''])

    # for all variables - logistic regression model adjusted for age, BMI, sex
    if len(set(df[v])) > 1:
        df['group'] = df['series'].apply(grp_to_number('SN_Group2', 'SN_Group1', grp1_val=0))
        log_reg = smf.logit(f"group ~ {v} + sex + age1 + BMI1", data=df).fit()
        output.append([v, str(series), f'LogR group ~ {v} + age1 + BMI1 + sex', log_reg.params[1],
                       log_reg.pvalues[1], f'beta = {log_reg.params[1]}'])

    if v in OUTCOMES and sum(df[v]) > 0:
        res = kaplan_meier(df, v, series, save_to=save_to, show_ims=show_ims)
        output.append(res)

    return output


if __name__ == '__main__':
    main()