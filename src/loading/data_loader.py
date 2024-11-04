import numpy as np
import pandas as pd
import math
from keras.utils import to_categorical
# from keras.utils import np_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import params as p, augmentation as aug, signal_analysis as sig


def get_all_data(hps, combine=False, do_preprocess=False, shuf=True, categorical=True, describe=False, dropna=True):
    if combine:
        data = combine_datasets(hps.datasets, hps.cnn_lead_order)
    else:
        data = get_data(hps.datasets[0][0], hps.datasets[0][1], hps.datasets[0][2], hps.cnn_lead_order, dropna=dropna)
    if describe: describe_data(data)
    if do_preprocess:
        return preprocess(data[0], data[1], hps, shuf=shuf, categorical=categorical)
    else:
        return data


def get_data(ecg_source, label_cols, selection, cnn_lead_order, dropna=False):
    """ Loads ECGs from source(s) specified in hps, and gets matching labels, discarding any NaNs
    Remaps the lead order if needed; concatenates the leads for the CNN sep leads model if needed
    Returns a list of matching ECGs (np array), and labels (unshuffled panda with IDs as index)
    selection is a dict of selection parameters e.g. {'hypertension': 0} to exclude labels with hypertension = 1 """
    ecgs = np.load(p.ecg_files[ecg_source][0], allow_pickle=True)
    md = np.load(p.ecg_files[ecg_source][1], allow_pickle=True)
    ids = [x['filename'][:7 + (x['filename'][3] == '-')] for x in md]
    ecgs, labels = match_labels(ecg_source, ecgs, ids, label_cols, selection,
                                verbose=False, dropna=dropna)

    # remap leads
    ecgs, md = remap(ecgs, md, cnn_lead_order, invert_AVR=False)

    return ecgs, labels


def match_labels(ecg_source, ecgs, ids, label_cols, selection, verbose=False, dropna=True):
    """ takes ECG lead data and metadata (used for filenames) and returns all ECGs where matching label data exists
    in label_cols, along with the matching labels and metadata.
    source_type should be 'ukbb' for UK Biobank data, or 'hcmr' for hcmr dataset
    any rows with non-numerical data will be dropped
    selection is a dict of selection parameters e.g. {'hypertension': 0} to exclude labels with hypertension = 1 """
    if verbose: print("Reading label data...")

    # import all labels data
    selection_cols = [col for col in selection]
    all_labels = pd.read_csv(p.label_files[ecg_source], index_col=0).loc[:, list(set(label_cols + selection_cols))]
    all_labels.index = all_labels.index.map(str)

    # select labels with matching ECG IDs
    labels = all_labels.reindex(ids)

    # exclude non-numerical data
    if dropna: labels = labels.dropna()
    non_num = len(ids) - len(labels)

    # apply selection filters
    filtered_out = ''
    for field, value in selection.items():
        tot = len(labels)
        labels = labels[labels[field] == value]
        filtered_out = filtered_out + str(tot - len(labels)) + ' excluded by ' + field + ' = ' + str(value) + '\n'

    # drop selection_cols and keep only label_cols
    drop_cols = [col for col in selection_cols if col not in label_cols]
    labels = labels.drop(columns=drop_cols)

    # remove non-matching ECGs
    included = labels.index.tolist()
    ecgs = ecgs[[x in included for x in ids]]

    print("Label matching:\n{} Total labels found\n{} non-numerical excluded\n{}{} labels returned".format(
        len(ids), non_num, filtered_out, len(labels)))
    return ecgs, labels


def preprocess(ecgs, labels, hps, ac_filter=(50, 60), shuf=True, categorical=True):
    """ scale, shuffle, split, and augment data; get label type and convert labels to categorical format if needed """
    # check if labels are for classification or regression
    hps.label_type = get_label_type(labels)

    # cast to float16
    ecgs = ecgs.astype(np.float16)
    labels.astype(np.float16)

    # apply electrical filter
    for freq in ac_filter:
        ecgs = np.array([sig.ac_filter(ecg, freq) for ecg in list(ecgs)])

    # down sample and transpose
    # ecgs = ecgs[:, :, ::hps.t_scale_fac] # resample by skip
    ecgs = resample_mean(ecgs, hps.t_scale_fac) # resample by average
    # ecgs = signal.resample(ecgs, ecgs.shape[2]//hps.t_scale_fac, axis=2) # resample by Fourier - needs testing
    ecgs = ecgs.transpose(0, 2, 1)
    hps.lead_length = ecgs.shape[1]

    # concatenate leads for sep leads model
    if hps.cnn_structure == 'CNN sep leads':
        ecgs = np.reshape(ecgs, (ecgs.shape[0], hps.lead_length * hps.cnn_n_leads, 1))

    # shuffle ecgs and labels
    if shuf:
        ecgs, labels = shuffle(ecgs, labels)

    # separate out training and test sets
    x_train, x_test = train_split(ecgs, hps.train_prop)
    y_train, y_test = train_split(labels, hps.train_prop)

    # data augmentation
    x_train, y_train, ids_train = aug.augment(x_train, y_train, hps.aug_n)
    x_test, y_test, ids_test = aug.augment(x_test, y_test, 0)

    # Convert y vectors to categorical format
    if (hps.label_type == 'clas') and categorical:
        y_train = make_categorical(np.array(y_train))
        y_test = make_categorical(np.array(y_test))

    return np.array(x_train), np.array(x_test), y_train, y_test, ids_train, ids_test, hps


def resample_mean(ecgs, fac):
    """ takes s data as array of shape (n_ecgs, n_leads, lead_length) and scaling factor fac (must be integer)
    Resamples ECG along axis 1 by factor """
    pad_size = math.ceil(ecgs.shape[-1] / fac) * fac - ecgs.shape[-1]
    ecgs = np.append(ecgs, np.full([ecgs.shape[0], ecgs.shape[1], pad_size], np.nan), axis=2)
    ecgs = ecgs.reshape((ecgs.shape[0], ecgs.shape[1], ecgs.shape[2]//fac, fac))
    resampled = np.nanmean(ecgs, axis=3)
    return resampled


def get_data_groups(groups, hps, do_preprocess=True, dropna=False):
    """
    groups: dict of {series name : [ecg_source, label_cols, selection], ...}
    dropna: list of columns to drop nan from
    :return: dict of all 4 main data groups with series name as key and label list as value
    """
    hps.train_prop = 1.
    output = {}
    for g in groups:
        ecgs, labels = get_data(groups[g][0], groups[g][1], groups[g][2],
                                hps.cnn_lead_order, dropna=dropna)
        if do_preprocess:
            ecgs, _, labels, _, ids, _, _ = preprocess(ecgs, labels, hps, categorical=False)
            output[g] = [ecgs, labels, ids]
        else:
            output[g] = [ecgs, labels]
    return output


def remap(ecgs, md, lead_order, invert_AVR=False):
    """ changes the order of leads for all ECGs in ecgs from to lead_order (given as lead names) using the
    lead names given in md[0]['lead order'] - i.e. assumes all ECGs have the same lead order as the first one """

    # reorder leads
    output_ecgs = np.zeros((ecgs.shape[0], len(lead_order), ecgs.shape[2]))
    for i in range(len(ecgs)):
        old_order = md[i]['lead order']
        if invert_AVR:
            l_avr = old_order.index('AVR')
            ecgs[i][l_avr] = - ecgs[i][l_avr]
        mapping = [old_order.index(x) for x in lead_order]
        output_ecgs[i] = [ecgs[i][mapping[j]] for j in range(len(lead_order))]
        md[i]['lead order'] = lead_order
    return output_ecgs, md


def train_split(data, train_prop):
    """ splits data into training and test sets by train_prop and returns the two subsets """
    train_number = int(train_prop * len(data))
    train, test = data[:train_number], data[train_number:]
    return train, test


def make_categorical(arr):
    """ converts np arrays of label integers into categorical format for classification networks
    returns a np array of integer labels """
    if arr.size != 0:
        arr = arr.astype(np.int8)
        arr = to_categorical(arr)
    return arr


def get_label_type(labels, max_categories=5, verbose=True):
    """ takes a dataframe of labels and determines whether each column should be classified or regressed
    returns a list with length labels.shape[1] where each item is 'clas' or 'regr'
    min_length is the minimum number of values needed to return a result per label (e.g. ~1000 for deep learning) """
    # any field taking more discrete values than max_categories will be treated as continuous/for regression
    types = []
    for index, data in labels.items():
        vals = set(data.dropna().tolist())
        if len(vals) <= max_categories:
            types.append('clas')
            if verbose:
                print('Using classifier model for label {}. Proportion in each class: {}'.format(
                    index, {x: round(sum(labels[index] == x) / len(data), 3) for x in vals}))
                print('Number in each class: {}'.format({x: sum(labels[index] == x) for x in vals}))
        else:
            types.append('regr')

    # check all types match
    if len(set(types)) > 1:
        # raise ValueError('Inconsistent label types found: {}'.format(types))
        pass
    else:
        return types[0]


def combine_datasets(datasets, cnn_lead_order, new_label_title='origin'):
    """ takes a list of datasets, each of which is of the form [ecg_source, label_col, selection] where:
    ecg_source (string) e.g. 'hcmr' or 'ukbb' such that all ECGs have the same dimensions except along axis 0,
    label_col (string) the column header from the corresponding label source file to extract
    selection (dict) with keys = column headings from labels files; values = value for each column key to include
    then concatenates the ecg arrays along axis 0, and
    attaches a label to each item based on its index in datasets e.g. each ECG in datasets[1] will be labelled 1
    title of this new label column will be new_label_title
    returns concatenated list and labels list"""

    all_ecgs = []
    output_labels = []
    for i in range(len(datasets)):
        ecg_source, label_col, selection = datasets[i]
        ecgs, labels = get_data(ecg_source, label_col, selection, cnn_lead_order)
        all_ecgs.append(ecgs)
        cols_to_drop = labels.columns
        labels[new_label_title] = i
        labels = labels.drop(columns=cols_to_drop)
        output_labels.append(labels)
    merged_ecgs = np.concatenate(all_ecgs)
    merged_labels = pd.concat(output_labels)

    return merged_ecgs, merged_labels


def describe_data(data):
    all_ecgs, all_labels = data
    for label in np.unique(all_labels['origin']):
        print('label =', label)
        ecgs = all_ecgs[all_labels['origin'] == label]
        labels = all_labels[all_labels['origin'] == label]
        flat = ecgs.flatten()
        print('Mean ECG value for label {} = {}'.format(label, np.mean(flat)))
        print('Variance of ECG values for label {} = {}'.format(label, np.var(flat)))
        plt.hist(flat, bins=2000, range=(-1000, 1000))
        plt.title('Histogram of origin = ' + str(label))
        # plt.xlim(-100, 100)
        plt.show()
        plt.close()


class ECG_gen(Sequence):
    def __init__(self, ECG_IDs, labels, hps):
        self.batch_size = hps.batch_size
        self.ECG_IDs = ECG_IDs
        self.labels = labels
        self.folder = hps.folder
        self.ecg_sources = hps.ecg_sources

    def __len__(self):
        return (np.ceil(len(self.ECG_IDs) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.ECG_IDs[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        if self.ecg_source == 'hcmr':
            fname_ext = '.xml'
        elif self.ecg_source == 'ukbb':
            fname_ext = '_20205_2_0.xml'
        return np.array([np.load(self.folder + fname + fname_ext) for fname in batch_x]), np.array(batch_y)


# lookup for meaning of each column label - taken directly from csv datafile
labels_dict = {
    0: 'BMI',
    1: 'Number of first degree relatives with HCM',
    2: 'Number of second degree relatives with HCM',
    3: 'Family members previously included in HCMR study',
    4: 'Number of family members previously included in HCMR study (4 max)',
    5: 'Number of first degree relatives with SCD',
    6: 'Number of second degree relatives with SCD',
    7: 'History of unexplained syncope',
    8: 'History of heart failure (NCDR definitions)',
    9: 'History of hospitalization for heart failure',
    10: 'Current NYHA class',
    11: 'History of stroke',
    12: 'History of ventricular tachycardia',
    13: 'History of atrial fibrillation',
    14: 'History of SCD',
    15: 'History of hypertension (NCDR definition)',
    16: 'Current smoking (NCDR definition)',
    17: 'Past smoking (NCDR definition)',
    18: 'Diabetes mellitus type 1',
    19: 'Diabetes mellitus type 2',
    20: 'Treated dyslipidemia',
    21: 'History of chest pain',
    22: 'History of dyspnea',
    23: 'Echo septal diastolic wall thickness (millimeters)',
    24: 'Echo septal diastolic wall - Not measured',
    25: 'Echo maximal diastolic wall thickness (millimeters)',
    26: 'Echo maximal diastolic wall - Not measured',
    27: 'Echo Systolic Anterior Motion',
    28: 'Echo gradient at rest  (mm Hg.)',
    29: 'Echo gradient at rest - Not performed',
    30: 'Echo provocable gradient (mm Hg.)',
    31: 'Echo provocable gradient - Not performed',
    32: 'Echo site of gradient',
    33: 'Echo ejection fraction (%)',
    34: 'Echo ejection fraction - Unknown',
    35: 'Echo mitral regurgitation grade',
    36: 'Echo pulmonary artery systolic pressure (mm Hg)',
    37: 'Echo pulmonary artery systolic pressure - Not measured',
    38: 'Echo maximum LA dimension (cm)',
    39: 'Echo maximum LA dimension - Not measured',
    40: 'Holter or other monitoring - Atrial fibrillation at time of monitoring',
    41: 'Holter or other monitoring - NSVT (>120bpm, 3beats) at time of monitoring',
    42: 'Holter or other monitoring - NSVT # of episodes',
    43: 'Holter or other monitoring - NSVT max HR',
    44: 'Holter or other monitoring - VT at time of monitoring',
    45: 'Holter or other monitoring - VT # of runs',
    46: 'Holter or other monitoring - VT max HR',
    47: 'Holter or other monitoring - VT longest run',
    48: 'Exercise Treadmill Test done',
    49: 'ETT - METS',
    50: 'ETT - METS - Not available',
    51: 'ETT - peak systolic blood pressure',
    52: 'ETT - peak systolic blood pressure - Not available',
    53: 'ETT - peak diastolic blood pressure',
    54: 'ETT - peak diastolic blood pressure - Not available',
    55: 'ETT - hypotension or failure to increase >20mm Hg',
    56: 'ETT - atrial fibrillation',
    57: 'ETT - VT',
    58: 'ETT - VF',
    59: 'Beta blocker',
    60: 'Calcium channel blocker',
    61: 'ACE or ARB',
    62: 'Aldosterone blocker',
    63: 'Amiodarone',
    64: 'Disopyramide',
    65: 'Sotalol',
    66: 'Other antiarrhythmic',
    67: 'Diuretic',
    68: 'Oral anticoagulant',
    69: 'Oral antiplatelet agents',
    70: 'Statin',
    71: 'HR at time of CMR',
    72: 'Systolic BP at time of CMR',
    73: 'Diastolic BP at time of CMR',
    74: 'Rhythm at time of CMR',
    75: 'Biomarker NTproBNP result',
    76: 'Biomarker TnTStat result',
    77: 'Biomarker galectin-3 (GAL3) result (pg/mL)',
    78: 'Biomarker ST2 result (pg/mL)',
    79: 'Biomarker matrix metalloproteinase-1 (MMP1) result (pg/mL)',
    80: 'Biomarker tissue inhibitor metalloproteinase-1 (TIMP1) result (pg/mL)',
    81: 'Biomarker C-terminal propeptide of type 1 procollagen (CICP) result (ng/mL)',
    82: 'Biomarker bone alkaline phosphatase (BAP) (U/mL)',
    83: 'sarcomere mutation results',
    84: 'lge_pres',
    85: 'lge_totalvis',
    86: 'lge_total6',
    87: 'lge_total4',
    88: 'greyzone',
    89: 'rvinsert_ant',
    90: 'rvinsert_post',
    91: 'septum',
    92: 'latwall',
    93: 'apex',
    94: 'vis_basal1',
    95: 'vis_basal2',
    96: 'vis_basal3',
    97: 'vis_basal4',
    98: 'vis_basal5',
    99: 'vis_basal6',
    100: 'vis_mid7',
    101: 'vis_mid8',
    102: 'vis_mid9',
    103: 'vis_mid10',
    104: 'vis_mid11',
    105: 'vis_mid12',
    106: 'vis_ap13',
    107: 'vis_ap14',
    108: 'vis_ap15',
    109: 'vis_ap16',
    110: '_6sd_bas1',
    111: '_6sd_bas2',
    112: '_6sd_bas3',
    113: '_6sd_bas4',
    114: '_6sd_bas5',
    115: '_6sd_bas6',
    116: '_6sd_mid7',
    117: '_6sd_mid8',
    118: '_6sd_mid9',
    119: '_6sd_mid10',
    120: '_6sd_mid11',
    121: '_6sd_mid12',
    122: '_6sd_ap13',
    123: '_6sd_ap14',
    124: '_6sd_ap15',
    125: '_6sd_ap16',
    126: '_4sd_bas1',
    127: '_4sd_bas2',
    128: '_4sd_bas3',
    129: '_4sd_bas4',
    130: '_4sd_bas5',
    131: '_4sd_bas6',
    132: '_4sd_mid7',
    133: '_4sd_mid8',
    134: '_4sd_mid9',
    135: '_4sd_mid10',
    136: '_4sd_mid11',
    137: '_4sd_mid12',
    138: '_4sd_ap13',
    139: '_4sd_ap14',
    140: '_4sd_ap15',
    141: '_4sd_ap16',
    142: 'sd_bas1',
    143: 'sd_bas2',
    144: 'sd_bas3',
    145: 'sd_bas4',
    146: 'sd_bas5',
    147: 'sd_bas6',
    148: 'sd_mid7',
    149: 'sd_mid8',
    150: 'sd_mid9',
    151: 'sd_mid10',
    152: 'sd_mid11',
    153: 'sd_mid12',
    154: 'sd_ap13',
    155: 'sd_ap14',
    156: 'sd_ap15',
    157: 'sd_ap16',
    158: 'sd_mean',
    159: 'sd_global',
    160: 'perqa',
    161: 'bsa',
    162: 'hr',
    163: 'lvrvmass',
    164: 'lvmass',
    165: 'lvmassi',
    166: 'lvedv',
    167: 'lvedvi',
    168: 'lvesv',
    169: 'lvesvi',
    170: 'lvsv',
    171: 'lvsvi',
    172: 'lvef',
    173: 'co',
    174: 'ci',
    175: 'lvmass2vol',
    176: 'wallthkbasal1',
    177: 'wallthkbasal2',
    178: 'wallthkbasal3',
    179: 'wallthkbasal4',
    180: 'wallthkbasal5',
    181: 'wallthkbasal6',
    182: 'wallthkmid7',
    183: 'wallthkmid8',
    184: 'wallthkmid9',
    185: 'wallthkmid10',
    186: 'wallthkmid11',
    187: 'wallthkmid12',
    188: 'wallthkapical13',
    189: 'wallthkapical14',
    190: 'wallthkapical15',
    191: 'wallthkapical16',
    192: 'rvmass',
    193: 'rvmassi',
    194: 'rvedv',
    195: 'rvedvi',
    196: 'rvesv',
    197: 'rvesvi',
    198: 'rvsv',
    199: 'rvsvi',
    200: 'rvef',
    201: 'Is patient still alive? ',
    202: 'Cause of death ',
    203: 'Did patient have a heart transplant? ',
    204: 'Did patient have a left-ventricular assist device placed? ',
    205: 'Was patient hospitalized for heart failure (or had a hospitalization prolonged due to HF)?',
    206: 'Did patient experience new onset atrial fibrillation? ',
    207: 'Did patient experience clinically documented VT/VF? ',
    208: 'Did patient experience a nonfatal stroke? ',
    209: 'Did patient have an ICD placed? ',
    210: 'Reason for ICD placement',
    211: 'Did patient experience ICD shock? ',
    212: 'Type of ICD shock',
    213: 'Did patient have a septal myectomy?',
    214: 'Did patient have an alcohol septal ablation? ',
    215: 'Did patient have a mitral valve replacement? ',
    216: 'Did patient have a mitral valve repair? ',
    217: 'Did patient have a pacemaker placement? ',
    218: 'Did patient have EF<50% on imaging? ',
    219: '',
    220: '',
    221: '',
    222: '',
    223: '',
    224: '',
    225: '',
    226: '',
    227: '',
    228: '',
    229: '',
    230: '',
    231: '',
    232: '',
    233: '',
    234: '',
    235: '',
    236: '',
    237: '',
    238: '',
    239: '',
    240: '',
    241: '',
    242: '',
    243: '',
    244: '',
    245: '',
    246: '',
    247: '',
    248: '',
    249: '',
    250: '',
    251: '',
    252: '',
    253: '',
    254: '',
    255: '',
    256: '',
    257: '',
    258: '',
    259: '',
    260: '',
    261: '',
    262: '',
    263: '',
    264: '',
    265: '',
    266: '',
    267: '',
    268: '',
    269: '',
    270: '',
    271: '',
    272: '',
    273: '',
    274: '',
    275: '',
    276: '',
    277: '',
    278: '',
    279: '',
    280: '',
    281: '',
    282: '',
    283: '',
    284: '',
    285: '',
    286: '',
    287: '',
    288: '',
    289: '',
    290: 'Thick var',
    291: 'Thin var'
}
