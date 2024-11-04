import my_tools as mt
import model_shapes

# Paths
folder = {'hcmr': 'G:\\HCMR\\',
          'hcmr ox': 'G:\\HCMR\\ECG data\\021raw\\',
          'ukbb': 'G:\\ukbb\\',
          'combined': 'G:\\Combined\\',
          'results': 'g:\\results\\',
          'mini': 'g:\\combined\\mini\\'}

label_files = {'hcmr': folder['hcmr'] + "Label data\\HCMR labels.csv",
               'hcmr ox': folder['hcmr ox'] + 'HCMR Ox labels.csv',
               'ukbb': folder['ukbb'] + 'labels\\processed\\UKBB labels.csv',
               'ukbb iso': folder['ukbb'] + 'labels\\processed\\UKBB labels.csv',
               'ukbb median': folder['ukbb'] + 'labels\\processed\\UKBB labels.csv',
               'ukbb mini': folder['ukbb'] + 'labels\\processed\\UKBB labels.csv',
               'combined': folder['combined'] + 'all labels.csv',
               'mini': folder['mini'] + 'all labels.csv'}

ecg_files = {'hcmr': [folder['hcmr'] + 'HCMR ECGs.npy',
                      folder['hcmr'] + 'HCMR metadata.npy'],
             'hcmr ox': [folder['hcmr ox'] + 'all_iso_ecgs.npy',
                         folder['hcmr ox'] + 'all_metadata.npy'],
             'ukbb': [folder['ukbb'] + 'ECG\\processed ECGs\\all_rest_ECGs.npy',
                      folder['ukbb'] + 'ECG\\processed ECGs\\metadata.npy'],
             'ukbb iso': [folder['ukbb'] + 'ECG\\processed ECGs\\all_rest_ECGs_iso.npy',
                          folder['ukbb'] + 'ECG\\processed ECGs\\metadata.npy'],
             'ukbb iso cols': [folder['ukbb'] + 'ECG\\processed ECGs\\all_rest_ECGs_iso_cols.npy',
                               folder['ukbb'] + 'ECG\\processed ECGs\\metadata.npy'],
             'ukbb median': [folder['ukbb'] + 'ECG\\processed ECGs\\all_median_ECGs.npy',
                             folder['ukbb'] + 'ECG\\processed ECGs\\metadata.npy'],
             'ukbb mini': [folder['ukbb'] + 'ECG\\processed ECGs\\mini set\\all_median_ECGs.npy',
                           folder['ukbb'] + 'ECG\\processed ECGs\\mini set\\metadata.npy'],
             'combined': [folder['combined'] + 'all_ECGs.npy',
                           folder['combined'] + 'all_metadata.npy'],
             'mini': [folder['mini'] + 'all_ECGs.npy',
                           folder['mini'] + 'all_metadata.npy']
             }

image_files = {'hcmr': folder['hcmr'] + 'images\\isolated\\',
               'ukbb median': folder['ukbb'] + 'ECG\\images\\medians\\'}

subsets = {
    'all BP': ['hypertension', ['combined', ['hypertension'], {}]],
    'UKBB BP': ['hypertension', ['combined', ['hypertension'], {'origin':0}]],
    'all sarc': ['sarc mutation', ['combined', ['sarc mutation'], {}]],
    'all': ['all', ['combined', ['origin'], {}]],
    'mini BP': ['hypertension', ['mini', ['hypertension'], {}]]
}


# Import
import_lead_order = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # import all leads
V_SCALE = 5  # expected voltage scale in uV
T_SCALE = 0.002  # expected time scale in seconds (= 1/sampling_frequency)


# CNN settings
lead_length = {'hcmr': 500,
               'hcmr ox': 500,
               'ukbb': 5000,
               'ukbb iso': 500,
               'ukbb median': 500,
               'ukbb mini': 500,
               'combined': 500,
               'mini': 500}  # in 500Hz values (e.g. 500 means 1 second)

# DEC
n_clusters = 8

# ECG measures
measures_cols = 33


class HyperParams():
    """ intialises key variables and hyperparameters for training, returns an object with multiple public variables """
    def __init__(self, data_selection):
        self.datasets = data_selection[1:]      # format as [data_title,
                                                # [ecg_src1, [label_cols1], selection1],
                                                # [ecg_src2, ...], ...]
        self.data_title = data_selection[0]     # string to use as title for this run e.g. "sarc mutation thick v thin"

        # experiment design
        self.repeats            = 5             # number of times to repeat experiment with identical parameters
        self.save_threshold     = 0.9           # save model if validation accuracy higher than this
        self.k                  = 10             # number of folds to use in k-folds
        self.show_ims           = False
        self.save_ims           = True

        # preprocessing
        self.aug_n              = 0             # number of times to apply augmentation functions (0 is none)
        self.t_scale_fac        = 5             # factor by which to downsample frequency of ECG
        self.cnn_lead_order     = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # selected leads for use in CNN
        self.lead_length        = self.match_lead_lengths(self.datasets)
        self.cnn_n_leads        = len(self.cnn_lead_order)

        # training parameters
        self.train_prop         = 0.9           # proportion of samples to be used for training instead of validation
        self.batch_size         = 50
        self.epochs             = 150
        if self.epochs == 1:
            self.save_ims=False
        self.early_stopping     = 0             # number of epochs with no improvement before stopping; 0 means no early stopping
        self.verbose            = 0

        # callbacks
        self.save_checkpoints   = False
        self.checkpoint_period  = 25            # save checkpoints every x epochs
        self.use_tensorboard    = False

        # model
        self.cnn_structure  = 'CNN general'     # name of a model architecture from ECG_models ('CNN general')
        self.model_name     = 'SJM_12_2023'            # best: 'SJM_12_2023'
        self.dims           = model_shapes.models[self.model_name]

        # files
        self.folder         = mt.make_dir(self.base_folder())

    def base_folder(self):
        lc = ['-'.join([d[0], '-'.join(d[1]), '-'.join([i[0] + '=' + str(i[1]) for i in d[2].items()])])
              for d in self.datasets]
        # f = folder['results'] + self.cnn_structure + '\\' + ' + '.join(lc) + '\\'
        f = folder['results'] + self.cnn_structure + '\\' + self.data_title + '\\'
        return f

    def set_folder(self, folder):
        self.folder = mt.make_dir(folder)
        return folder

    def describe(self):
        return ' '.join([self.data_title,
                         str(self.t_scale_fac),
                         str(self.epochs),
                         str(self.batch_size),
                         str(self.aug_n),
                         self.describe_model(self.dims)
                         ])

    def describe_model(self, model):
        """ converts list of model layers into a descriptive string for filing """
        output = ''
        for layer in model:
            if layer == 'ResNet':
                return str(model)
            if layer == 'Inception':
                return str(model)
            if layer[0] == 'Conv1D':
                output += '{}x{}-'.format(layer[1][0], layer[1][1])
            elif layer[0] == 'Dense':
                output += '{}-'.format(layer[1][0])
            elif layer[0] == 'MaxPool':
                output += 'pool{}-'.format(layer[1][0])
        return output

    def dims_to_string(self):
        """ converts a dims list to a tidy string for filenames """
        d1, d2, d3, d4 = self.dims
        d1 = ','.join(['x'.join([str(a[0]), str(a[1][0])]) for a in d1])
        d2 = ','.join(['x'.join([str(a[0]), str(a[1])]) for a in d2])
        d3 = ','.join([str(a) for a in d3])
        d4 = ','.join([str(a) for a in d4])
        output = '-'.join([d1, d2, d3, d4])
        return output

    def match_lead_lengths(self, datasets):
        lengths = [lead_length[d[0]] for d in datasets]
        if max(lengths) == min(lengths):
            return max(lengths)
        else:
            raise ValueError('Incompatible lead lengths in datasets:', datasets)


def abbreviate(series_list):
    mapping = {'UKBB normal BP': 'UKB NBP',
               'UKBB high BP': 'UKB HBP',
               'HCMR sarcomere negative': 'HCM SN',
               'HCMR sarcomere positive': 'HCM SP',
               'HCMR sarcomere negative hypertension': 'HCM SN HTN',
               'HCMR sarcomere negative no hypertension': 'HCM SN no HTN',
               'HCMR sarcomere positive no hypertension': 'HCM SP no HTN',
               'HCMR sarcomere positive hypertension': 'HCM SP HTN',}
    l = [list(map(lambda x: mapping[x], g)) for g in series_list]
    return l
