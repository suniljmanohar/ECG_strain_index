import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import eval_results
import my_tools as mt
import plots_2D
import preds_analysis
from vae_ECG_gen import generate_multi_ECG, gen_and_plot_ECG
import vae_params
from beta_vae_1D import VAE
from vae_analysis import load_data, get_mean_htn_score, get_latent_val_cols, visualise, split_SNHCM, pca
from vae_3D_plotting import plot_htn_axis3D, plot_htn_axis3D_split


def main():
    # settings
    subgroups = [['UKBB normal BP', 'UKBB high BP'], ['HCMR sarcomere negative'], ['HCMR sarcomere positive']]
    dependent_vars = ['log_echoprv', 'ECG hypertension score', 'log_NTproBNP', 'BMI1', 'age1',]

    # import params, load model and data
    vp = vae_params.VAE_params()
    vae = VAE.load(vp.model_dir)
    data = load_data(vp, vae)
    mean_htn = get_mean_htn_score(data)

    # ANALYSE
    if vp.do_split_SNHCM: data = split_SNHCM(data)

    compare_group_means(vp, vae, data, mean_htn, subgroups)

    compare_htn(vp, vae, data, mean_htn,
                subgroups=subgroups, dependent_vars=dependent_vars)

    # baseline_chars(vp, data, subset=['UKBB normal BP', 'UKBB high BP'], show_ims=vp.show_ims)


def compare_group_means(vp, vae, data, mean_htn, groups):
    means = []
    for g in groups:
        values = get_latent_val_cols(vp, data[data['series'].isin(g)])
        means.append(np.mean(values, axis=0))
    generate_multi_ECG(vae, np.array(means), np.array([mean_htn]*len(groups)),
                       groups, v_scale=vp.ecg_v_scale, title='', save_to=vp.model_dir + 'group centroid ECGs.svg', display=True)


def compare_htn(vp, vae, data, mean_htn, subgroups, dependent_vars):
    # generate hypertension axis in latent space
    if vp.condition_weight > 0:  # i.e. if condition input used
        analyse_cvae(vp, vae, data, mean_htn)
    else:
        analyse_vae(vp, data, mean_htn, vae, subgroups, dependent_vars,
                    display=vp.show_ims)


def analyse_cvae(vp, vae, data, mean_htn, display=True):
    percs = [np.percentile(data['ECG hypertension score'], p) for p in vp.percentiles]
    htn_score_series = np.array([[1 - x, x] for x in percs])
    title = 'CVAE hypertension axis multi-ECG'
    latent_val_series = np.zeros([len(vp.percentiles), vp.latent_dims])
    # htn_score_series = percs.reshape((percs.shape[0], 1))
    generate_multi_ECG(vae, latent_val_series, htn_score_series,
                       title=title, display=display, save_to=vp.model_dir + title + '.svg',
                       series_titles=[f'{p}th percentile' for p in vp.percentiles], v_scale=vp.ecg_v_scale
                       )


def analyse_vae(vp, data, mean_htn, vae, subgroups, dependent_vars, display=True):
    analyse_centroids_line(vp, data, mean_htn, vae, display=display)
    # vp.ecg_v_scale = 1.3

    for var in dependent_vars:
        cleaned_data = data.dropna(subset=var)

        # calculate multilinear LR coefficients to use as gradient
        res = lin_regr(vp, cleaned_data, dependent_var=var)
        intercept, coef = res.params[0], res.params[1:]

        for grp in subgroups:
            print(f'\nWorking on {var}, {grp}')
            sub_data = cleaned_data[cleaned_data['series'].isin(grp)]

            if len(sub_data) > 1:
                # plot HTN axes
                plot_htn_axis(coef, sub_data, display, mean_htn, grp, var, vae,
                              vp, vp.model_dir + f'VAE axis for {var} in {grp}.svg')

                # visualise collapsed HTN axis to hyperplane
                # hyperplane_vis(coef, sub_data, vp, subgroups, var)
            else:
                print(f'Insufficient data in {grp} for {var}')


def plot_htn_axis(coef, data, display, mean_htn, grp, var, vae, vp, save_to=''):
    projected = project_to_line(vp, get_latent_val_cols(vp, data),
                                coef, percentiles=vp.percentiles)
    ecg = generate_multi_ECG(vae, projected,
                             np.repeat([mean_htn], len(vp.percentiles), axis=0),
                             title=f'VAE axis for {var} in {grp}',
                             display=display, series_titles=[f'{p}th percentile' for p in vp.percentiles],
                             save_to=save_to, v_scale=vp.ecg_v_scale)


def hyperplane_vis(coef, data, vp, grp, var):
    # get hyperplane data
    collapsed_data = collapse_to_plane(vp, get_latent_val_cols(vp, data), coef)
    collapsed_data.index = data.index
    collapsed_data['series'] = data['series']
    collapsed_data[var] = data[var]

    # PCA and t_SNE plot
    X_pca, X_tsne = visualise(vp, collapsed_data, [var], 2,
                              display=False, save_to=vp.model_dir + f'HTN hyperplane\\{var} - {grp}')

    # contour plot
    plots = {'PCA': X_pca, 't-SNE': X_tsne}
    for coords in plots:
        df = pd.DataFrame({'series': data['series'], 'HTN score': data['ECG hypertension score'],
                           'y_pred': plots[coords][:, 0], 'y_true': plots[coords][:, 1]})
        eval_results.multiseries_contour(df, x_label='x1', y_label='x2', show_ims=False,
                                         save_to=vp.model_dir + f'HTN hyperplane\\{var} - {grp} - {coords}')


def lin_regr(vp, data, dependent_var):
    exog = sm.add_constant(get_latent_val_cols(vp, data))
    lin_model = sm.OLS(data[dependent_var], exog)
    res = lin_model.fit()
    print(res.summary())
    with open(vp.model_dir + f'{dependent_var} statsmodels lin_regr to latent dims', "wb") as f:
        pickle.dump(res, f)
    return res


def analyse_centroids_line(vp, data, mean_htn, vae, display=True):
    df = data[data['series']=='HCMR sarcomere negative']
    df = split_SNHCM(df)
    plot_centroids(df, display, mean_htn, vae, vp,
                   save_to=vp.model_dir + 'VAE axis Group1-Group2 centroids.svg')

    # subtract out hypertension by collapsing to hyperplane
    res = lin_regr(vp, df, dependent_var='ECG hypertension score')
    intercept, coef = res.params[0], res.params[1:]
    collapsed_data = collapse_to_plane(vp, get_latent_val_cols(vp, df), coef)
    collapsed_data.index = df.index
    collapsed_data['series'] = df['series']
    collapsed_data['ECG hypertension score'] = df['ECG hypertension score']
    # plot_htn_axis3D(vp, collapsed_data, save_to=save_file, plane=False, stalks=False)
    plot_htn_axis3D(vp, collapsed_data, save_to=vp.model_dir + 'SN-HCM HTN axis + 2-D PCA plot.svg')
    plot_htn_axis3D_split(vp, collapsed_data, save_to=vp.model_dir + 'SN-HCM HTN axis + 2-D PCA plot split.svg')

    # plot axis between centroids in collapsed data
    plot_centroids(collapsed_data, display, mean_htn, vae, vp,
                   save_to=vp.model_dir + 'VAE axis Group1-Group2 centroids collapsed to ECG HTN score hyperplane.svg')


def plot_centroids(data, display, mean_htn, vae, vp, save_to):
    grp1, grp2 = 'SN_Group1', 'SN_Group2'
    vec, offset = centroids_vec(vp, data, grp1, grp2)
    # plot axis using vector between two centroids
    old_scale = vp.ecg_v_scale
    vp.ecg_v_scale = 1.2
    plot_htn_axis(vec, data, display, mean_htn, 'HCMR sarcomere negative',
                  'Group 1 - Group 2 axis', vae, vp, save_to=save_to)
    vp.ecg_v_scale = old_scale


def project_to_line(vp, data, vector, offset=None, percentiles=True):
    """ projects data onto normalised vector then translates all of the projected points onto a line passing through
    the point offset (in direction vector)
    If offset is None, the resulting line passes through the mean point of the original data

    @data: array of shape (n_obs, latent_dims)
    @vector: vector in latent_dims space, does not need to be unit vector
    @percentiles: if False then simply returns all points in data. If True returns only percentiles from vp.percentiles
        along the line after transformation """
    if offset is None:
        offset = data.mean(axis=0)
    dot_prod = np.dot(data - offset, vector) / np.dot(vector, vector)
    if percentiles:
        dot_prod = [np.percentile(dot_prod, p) for p in vp.percentiles]
    output = offset + np.outer(dot_prod, vector)
    return output


def collapse_to_plane(vp, data, vector, offset=None, return_df=True):
    """ projects all points in data onto the hyperplane orthogonal to vector
    data: np array with shape (n_observations, LATENT_SPACE)
    vector: vector in LATENT_SPACE dimensions
    offset: vector from the origin to a point on the hplane. If None, use the mean of data
    returns:np array of vectors in LATENT_SPACE dimensions """
    if offset is None:
        offset = data.mean(axis=0)
    vector = vector / np.linalg.norm(vector)  # make unit vector
    dot_prod = np.dot(data - offset, vector)    # residual distance from hyperplane
    collapsed = data - np.outer(dot_prod, vector)
    if return_df:
        collapsed = pd.DataFrame(collapsed, columns=[str(i) for i in range(vp.latent_dims)])
        collapsed['HTN residuals'] = dot_prod
    return collapsed


def baseline_chars(vp, data, subset=(), show_ims=True):
    chars = preds_analysis.ECG_METRICS + preds_analysis.BASELINE_CHARS
    df = data[data['series'].isin(subset)]
    for v in chars:
        print(v)
        df = df.rename(columns={v:'y_pred'})
        df = df.dropna(subset='y_pred')
        dim_correlations(vp, df, subset=str(subset), show_ims=show_ims, y_label=v)


def dim_correlations(vp, data, y_label='', show_ims=False, subset=''):
    """ calculates correlations of each dimension with y_pred
    data: DataFrame with column 'y_pred' representing e.g. hypertension and columns numbered 0 to LATENT_DIMS which
    contain points in the VAE latent space representing transformed ECGs
    returns a DataFrame containing a row for each dimension, and giving values for the Pearson correlation between
    y_pred and that dimension, p value and CI. If do_sort then rows will be sorted by R^^2 value
    Saves density plots of the correlations """
    if len(data) > 2:
        r_list = []
        mt.make_dir(vp.model_dir + f'Dimensions vs {y_label} in {subset}\\')
        for dim in range(vp.latent_dims):
            print(f'Plotting correlations for dimension {dim} with {y_label}')
            series = data[str(dim)]
            pr = stats.pearsonr(series, data['y_pred'])
            ci = pr.confidence_interval(0.95)
            sr = stats.spearmanr(series, data['y_pred'])
            r_list.append([dim, pr.statistic, pr.pvalue, ci.low, ci.high, sr.pvalue])
            # density_plot(data, dim, model_dir, show_ims, subset)
            plot_data = pd.DataFrame({'y_true': series, 'y_pred': data['y_pred']})
            plots_2D.multiseries_scatter([[subset, plot_data]], f'Dim {dim}', y_label,
                                         save_file=vp.model_dir + f'Dimensions vs {y_label} in {subset}\\Dim {dim}.svg',
                                         title=f'Dimension {dim} correlation with {y_label} in ' + subset,
                                         show_ims=show_ims)

        r_list = pd.DataFrame(r_list, columns=['dimension', 'Pearson statistic', 'Pearson p value', 'CI low', 'CI high',
                                               'Spearman p value'])
        r_list = adjust_p(vp, r_list)
        r_list.to_csv(vp.model_dir + f'Dimension vs {y_label} correlations in  {subset}.csv')
        return r_list
    else:
        return


def density_plot(vp, data, dim, show_ims, subset):
    sns.kdeplot(data, x=str(dim), y='y_pred', fill=True, thresh=0.2, cmap='viridis')
    plt.suptitle(f'Dimension {dim} correlation with hypertension score' + subset)
    plt.xlim([-4, 4])
    if show_ims: plt.show()
    plt.savefig(vp.model_dir + f'Dimension correlation plots {subset}\\Dim {dim}.svg')
    plt.close()


def adjust_p(vp, r_list):
    r_list = r_list.sort_values(by=['Pearson p value'])
    r_list['adjusted Pearson p value (Bonferroni)'] = vp.latent_dims * r_list['Pearson p value']
    # Benjamini & Hochberg correction
    r_list['adjusted Pearson p value (B&H)'] = vp.latent_dims * r_list['Pearson p value'].div(r_list.index.to_series() +1, axis=0)
    return r_list


def centroids_vec(vp, data, grp1, grp2):
    """ returns the vector between the centroids of grp1 and grp 2
    :data: dataframe with column 'series' and other columns making up vector data
    :grp1: series name for group 1
    :grp2: series name for group 2 """
    d1, d2 = data[data['series']==grp1], data[data['series']==grp2]
    d1, d2 = get_latent_val_cols(vp, d1), get_latent_val_cols(vp, d2)
    vec = d1.mean(axis=0) - d2.mean(axis=0)
    vec = vec / np.linalg.norm(vec)
    return vec, d2.mean(axis=0)


# OLD
def plot_htn_axis_ECGs(vp, coef, intercept, vae):
    htn_ax = np.array([x * coef for x in range(-100, 100)])
    htn_scores = intercept + np.sum(htn_ax, axis=1)
    # decode axis into ECGs
    mt.make_dir(vp.model_dir + 'hypertension axis\\')
    for i in range(len(htn_ax)):
        gen_and_plot_ECG(vae, htn_ax[i].reshape((1, vp.latent_dims)), np.array([[0, 0]]),
                         fname='ECG_{:03d}'.format(i),
                         caption='Hypertension score = {:.2f}'.format(round(htn_scores[i], 3)),
                         output_folder=vp.model_dir + 'hypertension axis\\')


def compare_centroids(vp, vae, data, series1, series2):
    """ generates an axis between the latent-space-centroids of series 1 and series 2 and plots ECGs along the axis
    series1 and series2 are string names of series contained in the column called 'series' data DataFrame """
    data = data.drop(columns=['y_pred'])
    cluster1 = data[data['series']==series1].drop(columns=['series'])
    cluster2 = data[data['series']==series2].drop(columns=['series'])
    c1 = cluster1.mean(axis=0)
    c2 = cluster2.mean(axis=0)
    ax = np.linspace(c1, c2, 100)

    output_folder = vp.model_dir + '_'.join((series1, series2, '\\images\\'))
    mt.make_dir(output_folder)
    generate_multi_ECG(vae, [ax[0], ax[49], ax[99]], np.array([[0, 0]]),
                       series_titles=[series1, 'mid', series2], v_scale=vp.ecg_v_scale)
    for i in range(len(ax)):
        gen_and_plot_ECG(vae, ax[i].reshape((1, vp.latent_dims)), np.array([[0, 0]]),
                         fname='ECG_{:03d}'.format(i), caption='ECG_{:03d}'.format(i),
                         output_folder=output_folder)


def project_to_line_OLD(vp, data, vector, offset=None, percentiles=True):
    """ projects data onto normalised vector then translates all of the projected points onto a line passing through
    the point offset (in direction vector)
    If offset is None, the resulting line passes through the mean point of the original data

    @data: array of shape (n_obs, latent_dims)
    @vector: vector in latent_dims space, does not need to be unit vector
    @percentiles: if False then simply returns all points in data. If True returns only percentiles from vp.percentiles
        along the line after transformation """
    means = data.mean(axis=0)
    vector = vector / np.linalg.norm(vector)
    dot_prod = np.dot(data, vector)
    dot_prod = dot_prod - dot_prod.mean()
    if percentiles:
        dot_prod = [np.percentile(dot_prod, p) for p in vp.percentiles]
    output = np.outer(dot_prod, vector)
    output = output + means
    return output


def collapse_to_plane_OLD(vp, data, vector, offset=None, return_df=True):
    """ projects all points in data onto the hyperplane orthogonal to vector
    data: np array with shape (n_observations, LATENT_SPACE)
    vector: vector in LATENT_SPACE dimensions
    offset: vector from the origin to a point on the hplane. If None, use the mean of data
    returns:np array of vectors in LATENT_SPACE dimensions """
    if offset is None:
        offset = data.mean(axis=0)
    vector = vector / np.linalg.norm(vector)  # make unit vector
    dot_prod = np.dot(data, vector)
    collapsed = data - np.outer(dot_prod, vector)
    if return_df:
        collapsed = pd.DataFrame(collapsed, columns=[str(i) for i in range(vp.latent_dims)])
        collapsed['HTN residuals'] = dot_prod
    return collapsed


if __name__ == "__main__":
    main()
