import math

import pandas as pd
import numpy as np

VALIDATION = {'DBP': [10,200],
              'SBP': [40,300],
              'av_hr': [20, 300],
              'age1': [18,120],
              'max_LVOT_grad': [0.1, 200]
              # 'P axis': [-180, 270],
              # 'QRS axis': [-180, 270],
              # 'T axis': [-180, 270]#
}


def main():
    merge_labels(do_lvot_impute=False)
    # merge_ecgs()


def merge_ecgs():
    hcmr = np.load('g:\\hcmr\\HCMR ECGs.npy', allow_pickle=True)
    hcmr_md = np.load('g:\\hcmr\\HCMR metadata.npy', allow_pickle=True)
    #ukbb = np.load('g:\\ukbb\\ECG\\processed ECGs\\all_rest_ECGs_iso.npy', allow_pickle=True)
    #ukbb = np.load('g:\\ukbb\\ECG\\processed ECGs\\all_median_ECGs.npy', allow_pickle=True)
    ukbb = np.load('g:\\ukbb\\ECG\\processed ECGs\\all_rest_ECGs_iso_cols.npy', allow_pickle=True)
    ukbb_md = np.load('g:\\ukbb\\ECG\\processed ECGs\\metadata.npy', allow_pickle=True)
    all_md = np.concatenate((hcmr_md, ukbb_md), axis=0)
    all_ecg = np.concatenate((hcmr, ukbb), axis=0)
    np.save('g:\\Combined\\all_ecgs.npy', all_ecg, allow_pickle=True)
    np.save('g:\\Combined\\all_metadata.npy', all_md, allow_pickle=True)


def merge_labels(do_validation=True, do_lvot_impute=True):
    """
    Load hcmr labels csv and ukbb labels csv, merge them together adding a column 'origin' with value 0 for ukbb and 1
    for hcmr; map other columns to match e.g. age, systolic BP. Save the output as a new csv
    :return: None
    """

    hcmr_mapping = {
        'cmrsysbp': 'SBP',
        'cmrdiabp': 'DBP',
        'age': 'age1',
        'gender': 'sex',
        'bmi': 'BMI1',
        'hxhf': 'heart_failure',
        'hxhtn': 'hypertension',
        'cmrhr': 'av_hr',
        'lvedv': 'lv_edv',
        'lvesv': 'lv_esv',
        'lvef': 'lv_ef',
        'ci': 'cardiac_index',
        'co': 'cardiac_output',
        'sarcomere': 'sarc mutation'
    }

    # load hcmr data
    hcmr_label_file = 'g:\\hcmr\\label data\\HCMR labels.csv'
    hcmr_measures_file = 'g:\\hcmr\\label data\\measures.csv'
    hcmr_lvh_file = 'g:\\hcmr\\label data\\lvh criteria.csv'
    hcmr_features_file = 'g:\\hcmr\\label data\\HCMR ECG features.csv'
    hcmr_gls_file = 'g:\\hcmr\\gls\\strain.csv'

    hcmr = pd.read_csv(hcmr_label_file, index_col=0)
    hcmr_meas = pd.read_csv(hcmr_measures_file, index_col=0)
    hcmr_lvh = pd.read_csv(hcmr_lvh_file, index_col=0)
    hcmr_features = pd.read_csv(hcmr_features_file, index_col=0)
    hcmr_gls = pd.read_csv(hcmr_gls_file, index_col=0)

    # preprocess hcmr data
    hcmr = preprocess_hcmr(hcmr, hcmr_features, hcmr_lvh, hcmr_meas,
                           hcmr_gls, do_lvot_impute)

    # load ukbb data
    ukbb_label_file = 'g:\\ukbb\\labels\\processed\\ukbb labels.csv'
    ukbb_metrics_file = 'g:\\ukbb\\labels\\UKBB ECG metrics.csv'

    ukbb = pd.read_csv(ukbb_label_file, index_col=0)
    ukbb_metrics = pd.read_csv(ukbb_metrics_file, index_col=0)

    # preprocess UKBB data
    ukbb = ukb_preprocess(ukbb, ukbb_metrics)

    hcmr.rename(columns=hcmr_mapping, inplace=True)
    hcmr['origin'], ukbb['origin'] = [1]*len(hcmr), [0]*len(ukbb)
    comb = pd.concat([ukbb, hcmr])
    comb.index = comb.index.astype(str)
    comb['obesity'] = comb['BMI1'] >=30

    if do_validation:
        comb = validate(comb)

    if do_lvot_impute:
        f_out = 'all labels (LVOT imputed).csv'
    else:
        f_out = 'all labels (no LVOT imputed).csv'
    comb.to_csv('g:\\Combined\\'+f_out, index=True, header=True)


def preprocess_hcmr(hcmr, hcmr_features, hcmr_lvh, hcmr_meas, hcmr_gls, do_lvot_impute):
    # merge extra data into hcmr labels
    hcmr = hcmr.join(hcmr_meas)
    hcmr = hcmr.join(hcmr_lvh)
    hcmr = hcmr.join(hcmr_features)
    hcmr = hcmr.join(hcmr_gls[['glsmean', 'glsendo', 'glsepi']])
    hcmr = get_hcmr_fu2(hcmr)

    # split morphology into binary labels
    hcmr = split_morphology(hcmr)

    # fix diabetes labels
    hcmr['diabetes'] = hcmr[['diatype1', 'diatyp2']].any(axis=1).astype(int)

    # fix gender labels (female=0, male=1; as per UKBB)
    hcmr = hcmr.replace({'gender': {2: 0}})

    # remove zeros
    for var in ['echorest', 'echoprv']:
        hcmr[var] = hcmr[var].replace({0: np.nan})

    # fix echoprv and echorest
    subset = hcmr['echorest'] > hcmr['echoprv']
    hcmr.loc[subset, ['echorest', 'echoprv']] = hcmr.loc[subset, ['echoprv', 'echorest']].values

    # calculated variables
    hcmr['max_LVOT_grad'] = hcmr[['echorest', 'echoprv']].max(axis=1)
    if do_lvot_impute: hcmr = impute_lvot(hcmr)
    hcmr['LVOT_delta'] = hcmr['echoprv'] - hcmr['echorest']
    hcmr['LVOT_both_30'] = (hcmr['echorest']>=30) & (hcmr['echoprv']>=30)
    hcmr['LVOT_any_50'] = (hcmr['echorest'] >= 50) | (hcmr['echoprv'] >= 50)
    hcmr['LVOT_any_30'] = (hcmr['echorest'] >= 30) | (hcmr['echoprv'] >= 30)
    hcmr['low_lvef'] = hcmr['lvef'] < 50
    for var in ['max_LVOT_grad', 'echorest', 'echoprv']:
        hcmr[var + '_group'] = hcmr[var].apply(lvot_grp)
    hcmr['BNP_group'] = hcmr['NTproBNP'].apply(bnp_grp)
    hcmr['ethnicity_white'] = hcmr['race']==1
    hcmr['cad'] = hcmr['hxMI']
    hcmr['Morphology_1_2'] = hcmr['Morphology'].map({1:1, 2:2, 3:np.nan, 4:np.nan, 5:np.nan, 6:np.nan})
    hcmr['anti_HTN'] = (hcmr['blacearb']==1) | (hcmr['blbeta']==1) | (hcmr['blcalc']==1) | (hcmr['blaldo']==1)

    hcmr['ESC_SCD_risk_score'] = get_scd_risk(hcmr)
    hcmr = get_lavi(hcmr)
    hcmr['MR_severity'] = hcmr['echograd'] > 1
    hcmr['BP_meds'] = (hcmr['blcalc']==1) | (hcmr['blacearb']==1) | (hcmr['blsota']==1)
    hcmr['fhx_1st'] = (hcmr['hxhcm1st'] > 0) | (hcmr['hxscd1st'] > 0)
    hcmr['fhx_hcm_any'] = (hcmr['hxhcm1st']>0) | (hcmr['hxhcm2nd']>0)
    hcmr['wallthkmax > 30'] = hcmr['wallthkmax'] > 30
    hcmr['true_hcm'] = (hcmr['wallthkmax'] > 15) | ((hcmr['wallthkmax'] > 13) &
                                                    ((hcmr['sarcomere']==1) | (hcmr['fhx_1st']==1)))

    # combined baseline and incident morbidity
    hcmr['hf_any'] = (hcmr['hxhf']==1) | (hcmr['hf_hosp']==1) | (hcmr['death_hf']==1)
    hcmr['af_any'] = (hcmr['hxafib']==1) | (hcmr['hxafib']==2) | (hcmr['holtafib']==1) | (hcmr['ettafib']==1) | (hcmr['afib']==1)
    hcmr['vt_any'] = ((hcmr['hxvt']==1) | (hcmr['hxvt']==2) | (hcmr['vt']==1) | (hcmr['holtnsvt']==1) |
                      (hcmr['ettvt']==1) | (hcmr['ettvf']==1)) | (hcmr['SCD']==1) | (hcmr['vt_shock']==1)
    hcmr['stroke_any'] = (hcmr['hxstrk']==1) | (hcmr['stroke']==1)

    # add log variables
    for var in ['max_LVOT_grad', 'NTproBNP', 'TnTStat', 'echorest', 'echoprv', 'lavi', 'lav', 'ecvfwhole',
                'lvmassi']:
        hcmr['log_'+var] = hcmr[var].apply(np.log10)
    hcmr['log_lge_total6'] = (hcmr['lge_total6']+0.1).apply(np.log10)
    hcmr['log_LVOT_delta'] = (hcmr['LVOT_delta'] + 1).apply(np.log10)

    # fix spaces in names
    hcmr = hcmr.rename(columns={c: c.replace(' ', '_') for c in hcmr.columns})

    return hcmr


def get_hcmr_fu2(hcmr):
    raw = pd.read_csv('g:\\HCMR\\label data\\Nov 2023\\HCMRAdjDataset_2023-11-13.csv')
    df = pd.DataFrame(data=hcmr_fu_mapping(raw))
    condensed = condense_rows(df, max)
    condensed = condensed.drop(columns=['sitePatID'])
    condensed = condensed.astype(int)
    hcmr = hcmr.join(condensed)

    # assume records not included in follow up data have no events - fill False for all these
    cols = condensed.columns.tolist()
    cols.remove('daystoevent')
    hcmr = hcmr.fillna(value={c:0 for c in cols})

    return hcmr


def condense_rows(df, agg_fn):
    output = []
    for id in sorted(list(set(df['sitePatID']))):
        subset = df[df['sitePatID'] == id]  # select all rows with same ID
        subset = subset.apply(agg_fn, axis=0)  # take max value for each field -
        output.append(subset.tolist())
    output = pd.DataFrame(output, columns=df.keys())
    output.index = output['sitePatID']
    return output


def get_lavi(df):
    a1 = 'area4chamsys'
    a2 = 'area2chamsys'
    l1 = 'length2chamsys'
    l2 = 'length4chamsys'
    df['lav'] = 0.85 * (df[a1] * df[a2] / df[[l1, l2]].max(axis=1))
    df['lavi'] = df['lav']/df['bsa']
    return df


def get_scd_risk(df):
    # Probability of Sudden Cardiac Death at 5 years = 1 - 0.998 exp(PrognosticIndex)
    fhx = df['hxscd1st'] > 0
    nsvt = (df['holtnsvt']==1) | (df['holtvt']==1) | (df['ettvt']==1) | (df['hxvt']==1)
    prog_ind = 0.15939858*df['wallthkmax'] - 0.00294271*df['wallthkmax']**2 + 0.0259082*df['echomaxl']*10 + 0.00446131*df['max_LVOT_grad']+ 0.4583082*fhx + 0.82639195*nsvt + 0.71650361*df['hxsync'] - 0.01799934*df['age']
    risk = 1 - 0.998**(math.e**prog_ind)    # and check this
    return risk


def lvot_grp(lvot):
    if lvot < 30:
        return 0
    if 30 <= lvot < 50:
        return 1
    else:
        return 2


def bnp_grp(lvot):
    if lvot < 400:
        return 0
    if 400 <= lvot < 2000:
        return 1
    else:
        return 2


def split_morphology(hcmr):
    morphs = {1: 'M isolated basal septal',
              2: 'M asymmetric septal reverse curve',
              3: 'M apical',
              4: 'M concentric',
              5: 'M apical aneurysm',
              6: 'M other'}
    for i in range(1,7):
        hcmr[morphs[i]] = hcmr['Morphology'] == i
    return hcmr


def impute_lvot(hcmr):
    mean_normal_lvot = hcmr[hcmr['max_LVOT_grad'] <= 30]['max_LVOT_grad'].mean()
    hcmr['max_LVOT_grad'] = hcmr['max_LVOT_grad'].replace({np.nan: mean_normal_lvot})
    return hcmr


def validate(data):
    data = data.replace('#DIV/0!', np.nan)
    for col in VALIDATION:
        data[col] = data[col].astype(float)
        data.loc[data[col] < VALIDATION[col][0], col] = np.nan
        data.loc[data[col] > VALIDATION[col][1], col] = np.nan

    # add ox labels
    data['ox'] = [y in ['021'] for y in [x[:3] for x in data.index]]

    return data


def hcmr_fu_mapping(raw):
    # fields = ['hf',         # heart failure episode requiring hospitalisation
    #           'afib',       # atrial fibrillation event
    #           'death',      # did they die
    #           'death_cause',# 1=cardiac, 2=non-cardiac, 3=unclassified
    #           'deathCardio',# 1=sudden arhythmic, 2=non-sudden arrhythmic, 3=HF, 4=ischaemic, 5=procedural,
    #                         # 6=haemorrhage, 7=other, 8=stroke, 9=vascular
    #           'transplant', # did they undergo transplant surgery
    #           'stroke',     # did they have a fatal or non-fatal stroke
    #           'ICDShockAF', # ICD shock for atrial arrhythmia
    #           'vtach',      # VT event
    #           'vtachtype',  # 1=VF/VT arrest, 2=sustained VT, 3=ICD shock of VT/VF, 4=ICD ATP termination of VT/VF
    #           ]
    df = {
        'sitePatID': raw['sitePatID'],
        'hf_hosp': raw['hf'],
        'death': raw['death']==1,
        'death_hf': raw['deathCardio']==3,
        'death_cardiac': raw['death_cause']==1,
        'death_noncardiac': raw['death_cause']==0,
        'death_stroke': raw['deathCardio']==8,
        'SCD': raw['deathCardio']==1,
        'afib': raw['afib'],
        'vt': raw['vtach'],
        'vt_shock': raw['vtachtype']>2,
        'transplant': raw['transplant'],
        'stroke': raw['stroke'],
        'daystoevent': raw['daystoevent']
    }
    return df


def ukb_preprocess(ukbb, metrics):
    # average all SBP and DBP
    sbp_cols = [x for x in ukbb.columns if (x[:5]=='4080-' or x[:3]=='93-')]
    dbp_cols = [x for x in ukbb.columns if (x[:5]=='4079-' or x[:3]=='94-')]
    ukbb['SBP'] = ukbb[sbp_cols].mean(axis='columns')
    ukbb['DBP'] = ukbb[dbp_cols].mean(axis='columns')

    # remove HCM positive
    ukbb = ukbb[ukbb['HCM']==0]

    # merge with metrics
    ukbb = ukbb.join(metrics)
    ukbb['PR mean'] = ukbb['PQ mean']
    ukbb['cad'] = ukbb['MI'] | ukbb['CABG']

    # rename cols
    col_dict = {'af': 'af_any', 'stroke': 'hxstrk',
                '21000-0.0': 'ethnicity1', '21000-1.0': 'ethnicity2', '21000-2.0': 'ethnicity3',
                '21001-0.0': 'BMI1', '21001-1.0': 'BMI2', '21001-2.0': 'BMI3', '21001-3.0': 'BMI4',
                '21003-0.0': 'age1', '21003-1.0': 'age2', '21003-2.0': 'age3', '21003-3.0': 'age4',
                '22420-2.0': 'lv_ef', '22421-2.0': 'lv_edv', '22422-2.0': 'lv_esv', '22423-2.0': 'lv_sv',
                '22424-2.0': 'cardiac_output', '22425-2.0': 'cardiac_index', '22426-2.0': 'av_hr', '22427-2.0': 'body_sa',
                '31-0.0': 'sex'}

    ukbb = ukbb.rename(columns=col_dict)

    # convert ethnicity coding
    mapping = {1001:1, 1002:1, 1003:1, 2001:7, 2002:7, 2003:7, 2004:7, 3001:4, 3002:4, 3003:4, 3004:4,
               4001:2, 4002:2, 4003:2, 5:3, 6:8, -1:-1, -3:-3}
    ukbb['race'] = ukbb['ethnicity1'].map(mapping)
    ukbb['ethnicity_white'] = ukbb['race'] == 1
# HCM:  1 White
#       2 Black/African Am
#       3 East Asian
#       4 South Asian
#       5 Pacific Islander
#       6 American Indian/Alaskan Native

# UKB:  1 White: 1001 British, 1002	Irish, 1003	Any other white background,
#       2 Mixed: 2001 White and Black Caribbean, 2002 White and Black African, 2003 White and Asian, 2004 Any other mixed background
#       3 Asian or Asian British: 3001 Indian, 3002 Pakistani, 3003 Bangladeshi, 3004 Any other Asian background
#       4 Black or Black British: 4001 Caribbean, 4002 African, 4003 Any other Black background
#       5 Chinese
#       6 Other ethnic group
#       -1 Do not know
#       -3 Prefer not to answer

#   output ethnicity codes  1: White 2: Black/African 3: East Asian 4: South Asian 5: Pacific 6: American Indian/Alaskan
#                           7: Mixed 8: Other -1: Do not know -3: Prefer not to answer

    ukbb = ukbb.rename(columns={c: c.replace(' ', '_') for c in ukbb.columns})
    return ukbb


def merge_HCMRs():
    folder = 'g:\\hcmr\\label data\\June 2022\\'
    data = pd.read_csv(folder + 'HCMRBaselineDataset_2022-04-13.csv', index_col=0)
    datasets = [#'HCMRAdjDataset_2022-04-13.csv',  # contains duplicate rows (multiple events per person)
                'HCMRFUPDataset_2022-04-13.csv',
                'HCMRImageDataset_2020-02-27.csv']
    for d in datasets:
        df = pd.read_csv(folder + d, index_col=0)
        df = df[df.columns.difference(data.columns, sort=False)]
        data = data.join(df)
    data.to_csv('g:\\hcmr\\label data\\all labels.csv')


if __name__ == '__main__':
    main()
