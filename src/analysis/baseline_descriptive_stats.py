import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, kstest, mannwhitneyu, shapiro, chi2_contingency, fisher_exact, contingency
import pandas as pd
from statsmodels import api as sm

import params
import plots_2D

groups = {'all': {'included': 1},
          'UKBB': {'origin': 0, 'included': 1},
          'HCMR': {'origin': 1, 'included': 1},
          # 'HCMR excl':  {'origin':1, 'included':0},
          'UKBB no HTN': {'origin': 0, 'hypertension': 0, 'included': 1},
          'UKBB HTN': {'origin': 0, 'hypertension': 1, 'included': 1},
          'HCMR SN': {'origin': 1, 'sarc mutation': 0, 'included': 1},
          'HCMR SP': {'origin': 1, 'sarc mutation': 1, 'included': 1},
          'HCMR SN Gp1': {'origin': 1, 'sarc mutation': 0, 'SNHCM_Group': 1, 'included': 1},
          'HCMR SN Gp2': {'origin': 1, 'sarc mutation': 0, 'SNHCM_Group': 2, 'included': 1},
          'HCMR SP Gp1': {'origin': 1, 'sarc mutation': 1, 'SNHCM_Group': 1, 'included': 1},
          'HCMR SP Gp2': {'origin': 1, 'sarc mutation': 1, 'SNHCM_Group': 2, 'included': 1},
          'HCMR SN No HTN': {'origin': 1, 'sarc mutation': 0, 'hypertension':0},
          'HCMR SN HTN': {'origin': 1, 'sarc mutation': 0, 'hypertension':1},
          'HCMR HTN': {'origin':1, 'hypertension':1}
          }

test_pairs = [['UKBB', 'HCMR'],
              ['UKBB no HTN', 'UKBB HTN'],
              ['HCMR SP', 'HCMR SN'],
              ['UKBB HTN', 'HCMR SN'],
              ['UKBB HTN', 'HCMR SN Gp1'],
              ['UKBB HTN', 'HCMR SN Gp2'],
              ['UKBB no HTN', 'HCMR SP'],
              ['UKBB HTN', 'HCMR SP'],
              ['UKBB no HTN', 'HCMR SN'],
              ['HCMR SN Gp1', 'HCMR SN Gp2'],
              ['HCMR SP Gp1', 'HCMR SP Gp2'],
              ['UKBB HTN', 'HCMR SN HTN'],
              ['HCMR SN No HTN', 'HCMR SN HTN'],
              ['UKBB HTN', 'HCMR HTN'],
              # ['HCMR', 'HCMR excl']
              ]

series = {'UKBB vs HCMR': [{'origin': 0}, {'origin': 1}],
          'UKBB no HTN vs HTN': [{'origin': 0, 'hypertension': 0}, {'origin': 0, 'hypertension': 1}],
          'SNHCM Group 1 vs Group 2': [{'origin': 1, 'sarc mutation': 0, 'SNHCM_Group': 1},
                                       {'origin': 1, 'sarc mutation': 0, 'SNHCM_Group': 2}]
          }

HTN_SCORES = ('G:\\results\\significant results\\2024-03-05 final\\preds_logits.csv')


def main():
    f = get_fields()  # full set
    test_vars = list(f.keys())
    load_vars = test_vars + ['origin']
    output_folder = 'g:\\results\\baseline data analysis\\'
    # test_vars = ['sex', 'age1', 'BMI1', 'SBP', 'DBP', 'hypertension', 'diabetes']   # test set
    # load_vars = test_vars + ['origin', 'sarc mutation']

    data = load_data(load_vars, exclude_no_preds=True)

    # specific analyses
    # bp_bmi(data)
    htn_lvot(data)


    # intialise output df with totals row
    output = {g: [len(get_subset(data, ['origin'], groups[g]))] for g in groups}
    output['[shapiro p value]'] = [np.nan]
    for p in test_pairs:
        col_name = str(p)
        output[col_name] = [np.nan]

    for var in test_vars:
        print('Processing', var)
        # get data type
        d_type = f[var]

        # normality test
        if d_type > 0:
            output['[shapiro p value]'].append([np.nan, np.nan])
        else:
            shap_stat, shap_p = shapiro(data[var].dropna())
            output['[shapiro p value]'].append(['Shapiro-Wilk', shap_p])
            hist_qq(data[var].dropna(), save_to=output_folder + f'{var}')

        # calculate descriptive stats for each data group
        for g in groups:
            subset = get_subset(data, var, groups[g])
            stats = get_summ_stats(subset, d_type)
            output[g].append(stats)

        # compare test pairs
        for tp in test_pairs:
            # load main label data for each sub series and merge with hypertension scores
            d = []  # test data
            for selection in tp:
                d.append(get_subset(data, var, groups[selection]))

            # statistical test
            if len(d[0]) < 3 or len(d[1]) < 3:
                test_result = [np.nan, np.nan]
            else:
                if d_type == 0:     # continuous normal variable
                    test_result = do_ttest(d[0], d[1])
                elif d_type == -1:  # continuous non-normal variable
                    test_result = do_mwtest(d[0], d[1])
                else:               # categorical variable
                    test_result = chi2test([d[0], d[1]])[:2]

            output[str(tp)].append(test_result)

    output = finalise_output(output, list(f.values()))
    output = pd.DataFrame(data=output, index=['Totals'] + test_vars)
    output.to_csv(output_folder + '_baseline_comparison.csv', float_format='%.3f')


def bp_bmi(data):
    df = data[['BMI1', 'DBP', 'SBP']]
    df = df.dropna()
    series = [['DBP', pd.DataFrame({'BMI': df['BMI1'], 'BP': df['DBP']})],
              ['SBP', pd.DataFrame({'BMI': df['BMI1'], 'BP': df['SBP']})]]
    results = plots_2D.multiseries_scatter(series, 'BMI', 'BP', show_ims=True)


def get_summ_stats(data, d_type):
    if len(data) == 0:
        if d_type > 0:
            return np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan
    if d_type == 2:
        n = (data == 1).sum()
        pc = n / len(data)
        return n, pc
    elif d_type == 0:
        mu, std = np.mean(data), np.std(data)
        return mu, std
    elif d_type == -1:
        med, lower_q, upper_q = np.median(data), np.percentile(data, 25), np.percentile(data, 75)
        return med, lower_q, upper_q
    else:
        print('PROBLEM: uncertain data type, takes {} distinct values'.format(d_type))
        return np.nan, np.nan, np.nan


def do_ttest(d1, d2):
    t, p = ttest_ind(d1, d2)
    return ['t-test', p]


def do_mwtest(d1, d2):
    # t, p = kstest(d1, d2)
    mw = mannwhitneyu(d1, d2)
    return ['M-W test', mw.pvalue]


def load_data(load_vars, exclude_no_preds):
    # load label_data
    labels = pd.read_csv(params.label_files['combined'], index_col=0)
    labels.index = labels.index.map(str)

    # load hypertension scores
    preds = pd.read_csv(HTN_SCORES, index_col=0)
    preds.index = preds.index.astype(str)

    # merge with hypertension scores and field for whether included in analysis
    data = labels.join(preds)
    data = data.loc[:, load_vars]
    data['included'] = data.index.isin(preds.index)

    data = split_SNHCM(data)
    # data = data.drop(columns=['series'])

    if exclude_no_preds: data = data[data['included'] == 1]

    return data


def get_subset(data, var, sel):
    d = data.copy()
    for s in sel:
        d = d[d[s] == sel[s]]
    d = d[var]
    d = d.dropna()
    return d


def split_SNHCM(data, cutoff=0.5):
    """ takes a dataframe with column 'y_true'
    adds a column 'Group 1' which is True if y_true > cutoff and False otherwise
    returns dataframe with this column added """
    data['SNHCM_Group'] = (data['y_pred'] < cutoff) + 1
    return data


def finalise_output(output, dtypes):
    """ convert lists of stats in output to strings for publishing """
    for col in output:
        for i in range(1, len(output[col])):  # ignore first row of totals
            if col[0] == '[':
                # comparison column with test-type and p-value
                output[col][i] = 'p = {:.6f} ({})'.format(output[col][i][1], output[col][i][0])
            elif dtypes[i-1] == 2:
                # categorical var with count and percentage
                output[col][i] = '{} ({:.3f})'.format(output[col][i][0], 100 * output[col][i][1])
            elif dtypes[i-1] == 0:
                # continuous normal variable with mean and SD
                output[col][i] = '{:.3f} ± {:.3f}'.format(output[col][i][0], output[col][i][1])
            elif dtypes[i-1] == -1:
                # continuous non-normal variable with median, lower and upper quartiles
                output[col][i] = '{:.3f} ({:.3f}–{:.3f})'.format(output[col][i][0], output[col][i][1],
                                                                 output[col][i][2])
    return output


def get_fields():
    # DEMOGRAPHICS
    # UKBB - check if code numbers included
    fields = {
        'y_pred': -1,
        'sex': 2,
        'age1': 0,
        'ethnicity_white': 2,
        'wt': 0,
        'BMI1': 0,
        'obesity': 2,
        'cursmoke': 2,
        'pstsmoke': 2,
        'DBP': 0,
        'SBP': 0,
        'av_hr': 0,     # clinical heart rate
        'HR': 0,        # ECG heart rate

        # PMH - get stroke, AF, MR from UKBB
        'hypertension': 2,
        'diabetes': 2,
        'cad': 2,
        'hxstrk': 2,  # 'History of stroke'
        'heart_failure': 2,  # same as UKBB
        'hxafib': 3,  # 'History of atrial fibrillation', 1=Persistent, 2=Paroxysmal, 3=None
        'holtafib': 2,  # 'Holter or other monitoring - Atrial fibrillation at time of monitoring',
        'echograd': 5,  # 'Echo mitral regurgitation grade'
        'MR_severity': 2,  # 'Echo mitral regurgitation grade' grouped into 0:[0,1], 1:[2,3]
        'af_any': 2,
        'hf_any': 2,
        'vt_any': 2,
        'stroke_any': 2,
        'fhx_hcm_any': 2,

        # SYMPTOMS
        'blcp': 2,  # 'History of chest pain',
        'bldysp': 2,  # 'History of dyspnea',
        'hxsync': 2,  # 'History of unexplained syncope'
        'hxscd': 2,

        # SCD RISK
        'hxhcm1st': 5,
        'hxhcm2nd': 5,
        'hxscd1st': 5,  # 'Number of first degree relatives with SCD'
        'hxscd2nd': 7,  # 'Number of second degree relatives with SCD'
        'hxvt': 3,  # 'History of ventricular tachycardia',
        'echorest': -1,  # resting LVOT gradient
        'log_echorest': -1,
        'echorest_group': 3,
        'echoprv': -1,  # provoked LVOT gradient
        'log_echoprv': -1,
        'echoprv_group': 2,
        'max_LVOT_grad': -1,
        'log_max_LVOT_grad': -1,
        'LVOT_delta': -1,
        'log_LVOT_delta': -1,
        'max_LVOT_grad_group': 3,  # Max LVOTO gradient grouped by <30, 30-50, >=50
        'LVOT_any_50': 2,
        'LVOT_any_30': 2,
        'LVOT_both_30': 2,
        'echosam': 2,
        'wallthkmax_>_30': 2,
        'ESC_SCD_risk_score': -1,

        # MEDICATIONS
        'blbeta': 2,  # 'Beta blocker',
        'blcalc': 2,  # 'Calcium channel blocker',
        'blacearb': 2,  # 'ACE or ARB',
        'bldisop': 2,  # 'disopyramide'
        'blaldo': 2,  # 'Aldosterone blocker',
        'blsota': 2,  # 'Sotalol',
        'blamio': 2,  # 'Amiodarone',
        'BP_meds': 2,  # CCB or ACE or ARB,
        'anti_HTN': 2, # antihypertensive treatment

        # BIOMARKERS
        'sarc mutation': 2,
        'NTproBNP': -1,  # 'Biomarker NTproBNP result',
        'log_NTproBNP': -1,
        'BNP_group': 3,  # BNP grouped <400, 400-2000, >=2000
        'TnTStat': -1,  # 'Biomarker TnTStat result',
        'log_TnTStat': -1,
        'glsmean': -1,
        'glsepi': -1,
        'glsendo': -1,

        # CMR
        'Morphology': 6,    # 1=isolated basal septal; 2=asymmetric septal (reverse curve); 3=apical; 4=concentric; 5=mid cavity obstruction with apical aneurysm; 6=other
        'Morphology_1_2': 2,
        'M_isolated_basal_septal': 2,
        'M_asymmetric_septal_reverse_curve': 2,
        'M_apical': 2,
        'M_concentric': 2,
        'M_apical_aneurysm': 2,
        'M_other': 2,
        'wallthkmax': -1,
        'lv_edv': 0,
        'lvedvi': 0,
        'lvesvi': 0,
        'lv_ef': 0,
        'lv_esv': 0,
        'lvsv': 0,
        'lvsvi': 0,
        'cardiac_index': 0,
        'low_lvef': 2,
        'lvmass': 0,
        'lvmassi': 0,
        'rvedvi': 0,
        'rvesvi': 0,
        'rvef': 0,
        'reservoirpct': 0,
        'contractilepct': 0,
        'lge_pres': 2,
        'lge_totalvis': -1,
        'lge_total6': -1,
        'log_lge_total6': -1,
        't1prewhole': -1,
        'FieldStrength': 2,
        'ecvfwhole': -1,
        'log_ecvfwhole': -1,
        'lavi': -1,
        'log_lavi': -1,
        'lav': -1,
        'log_lav': -1,

        # ECG
        'Sokolow-Lyon_criteria': 2,
        'Cornell_criteria': 2,
        'SL_criteria_value': 0,
        'RR_mean': 0,
        'Pdur_mean': -1,
        'QRSdur_mean': -1,
        'QTcB_mean': 0,
        'PR_mean': -1,
        'JT_mean': 0,
        'JTc': 0,
        'JTp_mean': 0,
        'TpTe_mean': -1,
        'P_axis': 0,
        'QRS_axis': -1,
        'T_axis': -1,
        'Q_waves': 2,
        'TWI_presence': 2,
        'Giant_TWI': 2,
        'LBBB': 2,
        'RBBB': 2,
        'Fragmented_QRS': 2,
        'AF_ECG': 2,

        # FOLLOW UP
        # 'HFHOSP+HFDEATH',
        # 'fuafib',
        # 'fuvtvf',
        # 'VT+SHOCK',
        # 'fualive',
        # 'SCD',
        # 'HF DEATH',
        # 'NonCardiac Death',
        # 'STROKE DEATH',
        # 'CARDIAC DEATH',
        # 'futxp',
        # 'fuhosphf',
        # 'funostrk',
        # 'fuicd',
        # 'fuicdrea',
        # 'APPROPRIATE SHOCK',
        # 'fu ef < 50',
        'hf_hosp': 2,
        'death': 2,
        'death_hf': 2,
        'death_cardiac': 2,
        'death_noncardiac': 2,
        'death_stroke': 2,
        'SCD': 2,
        'afib': 2,
        'vt': 2,
        'vt_shock': 2,
        'transplant': 2,
        'stroke': 2
    }
    return fields


def chi2test(dfs):
    ct = contingency_table(dfs)
    if is_ct_valid(ct):
        res = chi2_contingency(ct)
        t, p = res[0], res[1]
        return ['chi-squared', p, t]
    else:
        if np.array(ct).shape == (2,2):
            res = fisher_exact(ct)
            t, p = res.statistic, res.pvalue
            return ['Fisher exact', p, t]
        else:
            return ['cell values too small for chi-square', np.nan, np.nan]


def contingency_table(dfs, return_dict=False, normalised=False):
    categories = sorted(list(set().union(*dfs)))
    ct = [[(d == c).sum() for c in categories] for d in (dfs)]
    if normalised:
        ct = [[x/sum(row) for x in row] for row in ct]
    if return_dict:
        ct = {str(categories[i]): [ct[j][i] for j in range(len(ct))] for i in range(len(categories))}
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
    if (ct_expected > 5).sum() < 0.8*ct_expected.size: is_valid = False    # freq_expected > 5 for 80% of cells (only 4 cells)
    if (np.array(ct) == 0).sum() > 0: is_valid = False            # all values in ct must be > 0
    return is_valid


def hist_qq(data, n_bins=30, show_ims=False, save_to=''):
    fig, ax = plt.subplots()
    plt.hist(data.dropna(), bins=n_bins)
    if save_to != '': plt.savefig(save_to + '-histogram.svg')
    if show_ims: plt.show()
    plt.close()

    sm.qqplot(data, line='s')
    plt.tight_layout()
    if save_to != '': plt.savefig(save_to + '-QQ plot.svg')
    if show_ims: plt.show()
    plt.close()


def htn_lvot(data):
    order = [0,1]   #['HCMR sarcomere negative', 'HCMR sarcomere positive']
    data['series'] = data['sarc mutation'].map({0: 'HCMR sarcomere negative', 1: 'HCMR sarcomere positive'})
    df = data.copy().dropna(subset=['series', 'hypertension', 'log_max_LVOT_grad'])
    # plots_2D.violin_plot(df, order, 'hypertension', 'log_max_LVOT_grad',
    #                      show_ims=False, save_to='g:\\results\\LVOT vs HTN.svg')


if __name__ == '__main__':
    main()
