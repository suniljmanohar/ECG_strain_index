import model_shapes
import params


class VAE_params():
    def __init__(self):
        # VAE design
        self.latent_dims = 32
        self.beta = 1
        self.condition_weight = 0
        self.model_shape = model_shapes.models['VAE']
        self.hps = params.HyperParams(params.subsets['all'])
        self.input_shape = (500 // self.hps.t_scale_fac, self.hps.cnn_n_leads)  # (ECG lead length, n_leads)

        # training params
        self.learning_rate = 0.001    # using Adadelta
        self.batch_size = 32
        self.epochs = 400
        self.train_prop = 0.999

        # execution
        # self.do_train = 0     # not needed?
        self.generate_preds = 1

        # analysis
        self.adjust_outliers = 1
        self.do_split_SNHCM = False

        # plotting
        self.percentiles = (5, 50, 95)
        self.ecg_v_scale = 1.8   # zoom in on voltage by this factor for publication use
        self.show_ims = False

        # files
        self.model_dir = 'g:\\CVAE\\models\\input={}, latent={}, batch_size={}, epochs={}, condition={}, beta={}\\'.format(
            self.input_shape, self.latent_dims, self.batch_size, self.epochs, self .condition_weight, self.beta)
        preds_folder = 'G:\\results\\CNN general\\hypertension\\k-fold\\'
        preds_file = "2024-02-11 12.32.43 hypertension 5 150 50 0 32x7-32x7-64x5-64x3-128x3-128x3-256x3-256x3-128-32- train_on all\\preds_logits.csv"
        self.htn_scores_suffix = 'latent space values 2023-11-23 00.34.25.csv'
        self.htn_scores = preds_folder + preds_file
        # HTN_SCORES = 'G:\\results\\CNN channels\\hypertension\\k-fold\\2022-11-15 10xk-fold\\merged means only.csv'
