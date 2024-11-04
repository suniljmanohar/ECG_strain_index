import pandas as pd
import numpy as np
import os


def codes():
    field_codes = {'CMR vals': list(range(22420, 22428)),
                   'FH': [20107, 20110, 20111],
                   'self dx': [20002],
                   'meds': [6177],  #[20003],
                   'base chars': [31, 21003, 21000, 21001, 23104, 93, 4080, 94, 4079, 30750],
                   'ICD10': [41270, 41202, 41204],
                   'ICD9': [41271, 41203, 41205]}

    self_coded = {'self_hypertension': [1065, 1072, 1073],
                  'self_diabetes_type1': [1222],
                  'self_diabetes_type2': [1223],
                  'self_diabetes_any': [1220, 1221, 1222, 1223],
                  'self_heart failure': [1076],
                  'self_af': [1471, 1483],           # includes flutter
                  'self_stroke': [1081, 1082],       # includes TIA
                  'self_MI': [1075],
                  'self_HCM': [1588],
                  'self_sarcoid': [1371]}


    icd_coded = {'icd_MI': ('I21', 'I22', 'I23', 'I200', 'I252'),   #includes unstable angina I200
                 'icd_CABG': ('T822', 'Z951', 'Z955'),
                 # 'icd_all htn'      :('I10', 'I11', 'I12', 'I13', 'I15', 'O10', 'O11', 'O13', 'O14', 'O16'),
                 'icd_hypertension': ('I10', 'I11', 'I12', 'I13', 'I15', 'O10', 'O11'),
                 'icd_diabetes': ('E10', 'E11', 'E12', 'E13', 'E14', 'O24'),
                 'icd_heart failure': ('I110', 'I130', 'I132', 'I50'),
                 'icd_af': ('I48',),
                 'icd_stroke': ('I64', 'G463', 'G464', 'I694'),
                 'icd_HCM': ('I421', 'I422'),
                 'icd_DCM': ('I420', 'I426'),
                 'icd_sarcoid': ('D86',)}

    # need to convert meds to code numbers
    medications = {
        'hypercholesterolemia': ['simvastatin', 'atorvastatin', 'fluvastatin', 'pravastatin', 'rosuvastatin', 'lipitor',
                                 'lipid lowering', 'lescol XL', 'lovastatin', 'altoprev', 'pitavastatin', 'livalo',
                                 'pravachol', 'crestor', 'zocor', 'cholestyramine', 'prevalite', 'colesevelam',
                                 'welchol',
                                 'colestipol', 'colestid', 'alirocumab', 'praluent', 'evolocumab', 'repatha'],
        'hypertension': ['ramipril', 'propranolol', 'spirinolacton', 'trandolapril', 'nifedipine', 'losartan',
                         'bisoprolol', 'enalapril', 'valsartan', 'perindopril', 'amlodipine', 'metoprolol',
                         'lecanidipine', 'lisinopril'],
        'diabetes': ['insulin', 'Insulin', 'metformin', 'Metformin', 'actos', 'amaryl', 'avandia', 'bydureon',
                     'byetta', 'eylea', 'forxiga', 'galvus', 'glucobay', 'insulin', 'invokana', 'januvia', 'jardiance',
                     'lantus', 'lucentis', 'lyxumia',
                     'metformin', 'onglyza', 'prandin', 'symlin', 'tresiba', 'trulicity', 'victoza']}

    medications = {'anti_HTN': [2]}

    return {'self': self_coded, 'icd': icd_coded, 'field': field_codes, 'meds': medications}


def main():
    ecg_folder = 'g:\\UKBB\\ECG\\'
    label_folder = 'g:\\UKBB\\labels\\raw\\'
    output_folder = 'g:\\UKBB\\labels\\processed\\'
    label_files = ['ukb33927.csv', 'ukb39925 fixed.csv', 'ukb39926.csv', 'ukb41160.csv', 'ukb46186.csv']
    field_codes = codes()['field']

    # records with null values in any of these fields will be discarded
    mandatory_fields = ['rest ECG']

    # records with null values in all of these fields will be discarded
    nonempty_fields = ['base chars', 'CMR vals']

    # records will be included regardless of whether these fields are null or non-null
    always_fields = ['ICD10', 'self dx', 'meds']

    # read data
    nonempty_codes = list(set().union(*[field_codes[x] for x in nonempty_fields]))
    always_codes = list(set().union(*[field_codes[x] for x in always_fields]))
    subset = get_all(ecg_folder, label_folder, label_files, mandatory_fields, nonempty_codes, always_codes)

    # select records by specific field value
    criteria = {}  # {'HCM':0, 'heart failure':0, 'DCM':0, 'MI':0}
    subset = select_field_value(subset, criteria)

    # save data
    all_fields = str(mandatory_fields + nonempty_fields + always_fields) + \
                 ','.join(['='.join([k, str(v)]) for k, v in criteria.items()])
    subset.to_csv(r'{}UKBB labels extract {}.csv'.format(output_folder, str(all_fields)),
                  index=True, header=True)


def get_all(ecg_folder, label_folder, label_files, mand_fields, nonempty_fields, always_fields):
    data = get_from_csv(ecg_folder, [label_folder + f for f in label_files], mand_fields, nonempty_fields,
                        always_fields)
    all_fields = mand_fields + nonempty_fields + always_fields
    if not data.empty:
        # convert self-reported diagnosis data to categorical format
        if 20002 in all_fields: data = selfdx_to_categorical(data)

        # convert ICD diagnosis data to categorical format
        if set(codes()['field']['ICD10']) & set(all_fields): data = icd_to_categorical(data)

        # merge self reported dx and ICD dx fields
        if (set(codes()['field']['ICD10']) & set(all_fields)) and 20002 in all_fields: data = merge_dxs(data)

        # merge medications into categorical fields
        if set(codes()['field']['meds']) & set(all_fields): data = meds_to_categorical(data)

    return data


def get_from_csv(ecg_folder, source_files, mand_fields, nonempty_fields, always_fields):
    """ read all the data from csv files in list source_files and return a panda df containing the requested fields
    mand_fields is a list of mandatory fields; records with null data in any one of these fields will not be returned
    nonempty_fields is a list of all other optional fields; records where these are ALL blank will not be returned
    always_fields is a list of all other fields; records will be returned irrespective of values in these fields
    """

    # read in data
    data_list = []
    all_fields = mand_fields + nonempty_fields + always_fields
    for sf in source_files:
        data_list.append(pd.read_csv(sf, index_col=0,
                                     usecols=lambda h: parse_header(h)[0] in ['eid'] + all_fields))

    # merge all files
    data = merge_dfs(data_list)

    # add 2 columns to indicate if has rest ECG, stress ECG
    data = get_has_ECG(ecg_folder, data)

    # split out mandatory and other headers
    headers = data.columns
    mand_headers = [h for h in headers if parse_header(h)[0] in mand_fields]
    nonempty_fields = [h for h in headers if parse_header(h)[0] in nonempty_fields]
    if mand_fields != [] and mand_headers == []:
        print('Could not find any of {} in headers in all files'.format(mand_fields))
        return pd.DataFrame([[]])

    # exclude lines with null values in any mandatory field
    data = data[data[mand_headers].notnull().all(1)]

    # exclude lines with null values in all optional fields
    if nonempty_fields:
        data = data[data[nonempty_fields].notnull().any(1)]

    return data


def get_has_ECG(ecg_folder, data):
    """ adds 2 columns onto the pandas dataframe data, True or False for whether has stress ECG or rest ECG """
    ids = get_ECG_IDs(ecg_folder)
    for id_list in ids:
        lookup = set(ids[id_list])
        data[id_list] = data.index.isin(lookup)
        data.loc[data[id_list] == 0, id_list] = np.nan
    return data


def get_ECG_IDs(ecg_folder, verbose=False):
    """ iterate over all ECG xml files. Returns list of all 7-digit file IDs, list of all resting ECG IDs, and
    list of all stress ECG IDs """
    fnames, rest_fnames, stress_fnames = [], [], []
    for fname in os.listdir(ecg_folder):
        if fname[-3:] == 'xml':
            try:
                fnames.append(int(fname[:7]))
            except ValueError:
                if verbose: print('{} is not an integer'.format(fname[:7]))
            if fname[8:13] == '20205':
                rest_fnames.append(int(fname[:7]))
            elif fname[8:12] == '6025':
                stress_fnames.append(int(fname[:7]))
            else:
                print('Could not categorise {}'.format(fname))
    return {'rest ECG': rest_fnames, 'stress ECG': stress_fnames}


def selfdx_to_categorical(data):
    """ takes a pandas df with some headers in form '20002-x.y' converts the diagnosis labels to categorical columns and
    appends these columns to the df; returns the enhanced df """
    col_select = data.columns.map(lambda x: x[:5] == '20002')
    dx_cols = data[data.columns[col_select]]
    for dx in codes()['self']:
        data[dx] = dx_cols.isin(codes()['self'][dx]).astype('int16').max(axis=1)
    data = data.drop(dx_cols, axis=1)
    return data


def icd_to_categorical(data):
    """ takes a pandas df with some headers in form '4120x-y.z' converts the diagnosis labels to categorical columns and
    appends these columns to the df; returns the enhanced df """
    col_select = data.columns.map(lambda x: x[:5] in str(codes()['field']['ICD10']))
    dx_cols = data[data.columns[col_select]]
    for dx in codes()['icd']:
        data[dx] = dx_cols.apply(lambda x: x.str.startswith(codes()['icd'][dx]).any(0), axis=1).astype('int16')
    data = data.drop(dx_cols, axis=1)
    return data


def meds_to_categorical(data):
    col_select = data.columns.map(lambda x: x[:4] == '6177')
    med_cols = data[data.columns[col_select]]
    for med in codes()['meds']:
        data[med] = med_cols.isin(codes()['meds'][med]).astype('int16').max(axis=1)
    data = data.drop(med_cols, axis=1)
    return data


def select_field_value(data, criteria):
    """ selects rows from DataFrame data based on criteria
    criteria: dict with column headers as keys
    """
    for field, value in criteria.items():
        data = data[data[field] == value]
    return data


def merge_dfs(dfs):
    """ takes a list of dataframes as dfs and produces a single dataframe that includes indexes and columns from
    all the dataframes. If any duplicate index/column pairs are found, the value from the first list in which that
    pairing is found will be prioritised """
    if len(dfs) < 2: return dfs
    result = dfs[0]
    for df in dfs[1:]:
        result = result.combine_first(df)
    return result


def merge_dxs(data):
    """ if data contains columns with both self-reported diagnoses (x) and ICD coded diagnoses (y), creates a new column
    which is x OR y, then deletes the original x and y columns
    data: pandas DataFrame with rows as individual records """
    # merge any codes occurring in both
    for dx in codes()['icd']:
        base_dx = dx[4:]
        if 'self_' + base_dx in codes()['self']:
            data[base_dx] = (data[dx] | data['self_' + base_dx]).astype('int16')
            data = data.drop([dx, 'self_' + base_dx], axis=1)
        else:
            data = data.rename(columns={dx: base_dx})

    # rename any remaining self dx columns
    for field in data.columns:
        if field[:4] == 'self': data = data.rename(columns={field: field[5:]})

    return data


def parse_header(h):
    """ splits header string in format 'x-y.z' and returns x, y, z where x, y, z are integers"""
    if h in ['eid', 'stress ECG', 'rest ECG']: return [h]
    h0, h1 = h.split('-')
    h1, h2 = h1.split('.')
    try:
        h0, h1, h2 = int(h0), int(h1), int(h2)
    except:
        print('Could not convert header label to type int:', h0, h1, h2)
    return h0, h1, h2


if __name__ == '__main__':
    main()
