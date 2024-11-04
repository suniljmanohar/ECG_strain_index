import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

import params as p
import trainer
from beta_vae_1D import VAE
import vae_params
from vae_ECG_gen import generate_multi_ECG
from vae_train import get_data
import my_tools as mt
import eval_results as er
import preds_analysis
from plots_2D import multiseries_contour

import ECG_plots

tf.compat.v1.enable_eager_execution()


def main():
    var_list = ['ECG hypertension score', 'n_series'] + preds_analysis.BASELINE_CHARS + preds_analysis.ECG_METRICS

    vp = vae_params.VAE_params()
    vae = VAE.load(vp.model_dir)

    # LOAD DATA
    data = load_data(vp, vae)
    mean_htn = get_mean_htn_score(data)

    # ANALYSE
    analyse_training(vae)

    if vp.do_split_SNHCM: data = split_SNHCM(data)

    latent_space_covariance(vp, data)

    visualise(vp, data, var_list, save_to=vp.model_dir, display=False)

    plot_traversals(vp, vae, get_latent_val_cols(vp, data),
                    np.repeat([mean_htn], len(vp.percentiles), axis=0), display=False)

    plot_latent_spaces(vp, data, plot_cutoff=0.5)


def visualise(vp, data, var_list, n_components=2, display=True, save_to=''):
    series_list = list_series(data['series'])
    df = data.dropna()
    print('Calculating PCA')
    X_pca = pca(vp, df, n_components)
    plot_visualisation(df, X_pca, series_list, var_list,
                       save_to=save_to, display=display, vis_name='PCA')
    print('Calculating t-SNE')
    X_tsne = t_sne(vp, df, n_components, save_to=save_to+'t-SNE\\')
    plot_visualisation(df, X_tsne, series_list, var_list,
                       save_to=save_to, display=display, vis_name='t-SNE')
    # umap()
    return X_pca, X_tsne


def pca(vp, data, n_components):
    pca = PCA(n_components=n_components)
    X = get_latent_val_cols(vp, data)
    X_pca = pca.fit_transform(X)
    print(f'Explained variance = {pca.explained_variance_ratio_}')
    return X_pca


def t_sne(vp, data, n_components, perplexity=30, save_to=''):
    if vp.generate_preds:
        X = get_latent_val_cols(vp, data)
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)
        print(f't-SNE KL divergence = {tsne.kl_divergence_}')
        mt.make_dir(save_to)
        np.save(save_to + f'{n_components}D vals.npy', X_tsne)
    else:
        X_tsne = np.load(save_to + f'{n_components}D vals.npy', allow_pickle=True)
    return X_tsne


def plot_visualisation(data, X, series_list, var_list, discrete_cutoff=10, display=False, save_to='', vis_name=''):
    save_to = save_to + vis_name + '\\'
    mt.make_dir(save_to)

    # prepare data
    data['n_series'] = [series_list.index(i) for i in data['series']]

    # plot for each covariate
    for var in var_list:
        colours = data[var]
        fig, ax = plt.subplots(figsize=(16, 16))
        plt.suptitle(f"{vis_name} visualization of {var}")
        if len(set(colours)) < discrete_cutoff:
            scatter = ax.scatter(x=X[:, 0], y=X[:, 1], c=colours, alpha=0.8, s=8.0, edgecolors='none')
            plt.legend(handles=scatter.legend_elements()[0], labels=series_list, loc="lower right")
        else:
            scatter = ax.scatter(x=X[:, 0], y=X[:, 1], c=colours, cmap='rainbow', alpha=.8, s=8.0,  edgecolors='none')
            fig.colorbar(scatter)
        if display: plt.show()
        if save_to != '':
            plt.savefig(save_to + var + '.svg')
            plt.savefig(save_to + var + '.jpg')
        plt.close()


def get_latent_val_cols(vp, df):
    cols = [str(i) for i in range(vp.latent_dims)]
    df = df[cols]
    return df.to_numpy()


def plot_traversals(vp, model, mus, htn_scores, display=False):
    """ plots multi-ECGs along each axis in the latent space """
    mt.make_dir(vp.model_dir+'dimension traversals\\')
    ecgs = []
    for dim in range(vp.latent_dims):
        print(f'Plotting traversal along dim {dim}')
        vector = np.zeros(vp.latent_dims)
        vector[dim] = 1
        ecg = plot_axis_traversal(vp, vector, model, mus, htn_scores,
                                           display=display,
                                           title=f'Dimension {dim}',
                                           save_to=vp.model_dir + f'dimension traversals\\dimension {dim}.svg')
        ecgs.append(ecg)
    plot_all_traversals(vp, ecgs, display=display,
                        title='All dimension traversals', save_to=vp.model_dir + 'dimension traversals\\all dims.svg')


def plot_all_traversals(vp, ecgs, title='', save_to='', display=True):
    print('Plotting multi-dimensional traversals')
    ecgs = np.array(ecgs)
    ECG_plots.traversals(ecgs, vp.hps.cnn_lead_order,
                         series_titles=[f'{p}th percentile' for p in vp.percentiles], v_scale=vp.ecg_v_scale,
                         title=title, display=display, save_to=save_to)


def plot_axis_traversal(vp, vector, model, data, htn_scores, title='', save_to='', display=True):
    """ takes a vector in latent space and projects data onto the vector. Reconstructs and plots ECGs from various
     percentile points along that line according to the data distribution
    vector: any vector of dimension LATENT_DIMS
    model: variational autoencoder model, instance of VAE class
    data: dataframe or np array of dimensions (n_observations, LATENT_DIMS)
    perecentiles: percentiles of data to use, e.g. [1, 25, 50, 75, 99]
    """
    vector = vector / np.linalg.norm(vector) # make unit vector
    projected = np.dot(data, vector)   # project all data points onto grad line

    percs = [np.percentile(projected, p) for p in vp.percentiles]
    coord_series = np.array([vector * p for p in percs])
    ecg = generate_multi_ECG(model, coord_series, htn_scores, title=title,
                             display=display, series_titles=[f'{p}th percentile' for p in vp.percentiles],
                             save_to=save_to, v_scale=vp.ecg_v_scale)
    return ecg


def plot_latent_spaces(vp, data, plot_cutoff=0.3, subset_name=''):
    mt.make_dir(vp.model_dir + f'latent space plots {subset_name}\\')
    mins, maxes = low_high(get_latent_val_cols(vp, data), cutoff=plot_cutoff)
    for i in range(vp.latent_dims):
        print(f'Plotting multi-series for dimension {i}')
        plot_data = data.rename(columns={str(i): 'y_true'})
        series_list = preds_analysis.sorted_series_list(plot_data)
        multiseries_contour(plot_data, 'y_true', 'ECG hypertension score', series_list,
                            x_label='latent dim {}'.format(i), y_label='hypertension score',
                            show_ims=False, x_lim=[mins[i], maxes[i]],
                            save_to=vp.model_dir +f'latent space plots {subset_name}\\dim {i}')


def load_data_to_plot(vp, vae):
    """ loads HTN preds and join with latent space predicted values """
    # load htn scores
    preds = pd.read_csv(vp.htn_scores, index_col=0)
    preds.index = preds.index.astype(str)

    # get latent vars for each ECG
    mus, logvars, ids = get_latent_space(vae)
    mu_df = pd.DataFrame(mus, index=ids, columns=[str(x) for x in range(vp.latent_dims)])
    mu_df.index = mu_df.index.astype(str)

    # merge and tidy
    merged = preds.join(mu_df)
    # merged = merged.drop(columns='y_true')
    merged = merged.dropna()

    return merged


def get_latent_space(vae):
    X_tr, _, Con_tr, _, ids_tr, _ = get_data(1.)[:6]
    _, mus, logvars = vae.model.predict([X_tr, Con_tr])[0]
    return mus, logvars, ids_tr


def get_mean_htn_score(df):
    mean = df['hypertension'].mean()
    return np.array([1-mean, mean])


def low_high(mus, cutoff=0.1):
    lows = [np.percentile(mus[:, i], cutoff) for i in range(mus.shape[1])]
    highs = [np.percentile(mus[:, i], 100-cutoff) for i in range(mus.shape[1])]
    return lows, highs


def adjust_outliers(vp, data, cutoff=5):
    """ takes a df with and calculates z-scores, for ||z-score|| > cutoff, the value is replaced by the edge
     value at +/- cutoff """
    for dim in range(vp.latent_dims):
        series = data[str(dim)]
        lower = series.mean() - cutoff * series.std()
        upper = series.mean() + cutoff * series.std()
        data[str(dim)] = np.where(series > upper, upper, np.where(series < lower, lower, series))
    return data


def split_SNHCM(data, cutoff=0.5):
    """ splits SN-HCM subgroup into Group 1 and Group 2 based on hypertension score
    data: dataframe with column 'series' containing series names of which some will be "HCMR sarcomere negative"
    and another column 'y_true' containing hypertension scores
    cutoff: the value for the hypertension score used to split into Group 1 and Group 2
    :returns data, but with series name changed to Group 1 or Group 2 for all SN-HCM
    """
    df = data.copy()
    df['series'] = np.where((df['series']=='HCMR sarcomere negative') & (df['ECG hypertension score']>cutoff),
                              'SN_Group1', df['series'])
    df['series'] = np.where((df['series'] == 'HCMR sarcomere negative') & (df['ECG hypertension score'] <= cutoff),
                              'SN_Group2', df['series'])
    print(f"{sum(df['series']=='SN_Group1')} in Group 1")
    print(f"{sum(df['series']=='SN_Group2')} in Group 2")
    return df


def latent_space_covariance(vp, data):
    cov_mat = np.cov(get_latent_val_cols(vp, data).T)
    np.savetxt(vp.model_dir + 'latent space covariance matrix.csv', cov_mat, delimiter=",")


def load_data(vp, vae):
    if vp.generate_preds:
        data = load_data_to_plot(vp, vae)
        data.to_csv(vp.model_dir + vp.htn_scores_suffix)
    else:
        data = pd.read_csv(vp.model_dir + vp.htn_scores_suffix, index_col=0)
    data.index = data.index.map(str)
    # data = add_htn_data(data)
    if vp.adjust_outliers: data = adjust_outliers(vp, data, 10)

    labels = load_bl_chars()
    data = data.rename(columns={'y_pred': 'ECG hypertension score'})
    data = data.join(labels)
    return data


def load_bl_chars():
    chars = preds_analysis.ECG_METRICS + preds_analysis.BASELINE_CHARS + preds_analysis.IMAGING_CHARS
    labels = pd.read_csv(p.label_files['combined'], index_col=0).loc[:, chars]
    labels.index = labels.index.map(str)
    return labels


def list_series(series_list):
    output = []
    for x in series_list:
        if x not in output: output.append(x)
    return output


def analyse_training(vae):
    hist = vae.trainingHistory
    trainer.plot_train_history(hist, ['loss', 'encoder_loss', 'decoder_loss'], show_ims=True, val=True)


# OLD
def add_htn_data(data):
    raw_data = pd.read_csv(p.label_files['combined'], index_col=0).loc[:, ['hypertension']]
    raw_data.index = raw_data.index.map(str)
    data = data.join(raw_data)
    return data


if __name__ == "__main__":
    main()
