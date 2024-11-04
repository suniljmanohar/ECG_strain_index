import colorsys
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt, patches as mpatches, colors as mcolors, lines as mlines
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.stats import pearsonr, linregress, sem
from scipy.interpolate import make_interp_spline
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import pairwise_logrank_test
from statannotations.Annotator import Annotator

import merge_datasets
import params
import eval_results as er

COLOURS = {'UKBB normal BP': 'green',
           'UKBB high BP': 'red',
           'HCMR sarcomere positive': 'orange',
           'HCMR sarcomere negative': 'blue',
           'HCMR sarcomere negative hypertension': 'magenta',
           'HCMR sarcomere negative no hypertension': 'blue',
           'HCMR sarcomere positive no hypertension': 'orange',
           'HCMR sarcomere positive hypertension': 'magenta',
           0: 'green',
           1: 'red'}

mpl.rcParams.update({'font.size': 18})


def multiseries_scatter(data, x_label, y_label, save_file='', show_ims=False, title=''):
    # dot_colours = ['yellow', 'red', 'limegreen', 'blue', 'lightgrey']
    # line_colours = ['gold', 'darkred', 'darkgreen', 'darkblue', 'darkgrey']
    patches = []
    results = []
    plt.figure(figsize=(10, 10))
    for i in range(len(data)):
        series, subset = data[i]
        x, y = subset[x_label].astype(float), subset[y_label].astype(float)
        r, p = pearsonr(x, y)
        results.append([x_label, series, 'pearson r', r, p, f'r = {r:.3f}'])
        grad, incpt, lr_r, lr_p, stderr = linregress(x, y)
        results.append([x_label, series, 'linear regression', lr_r, lr_p, f'grad = {grad:.5f}\nintercept = {incpt:.3f}'])

        plt.scatter(x, y, marker='o', s=80, facecolors='none', edgecolors=COLOURS[series[0]], alpha=0.5)
        line_x, line_y = np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x))
        plt.plot(line_x, line_y, color=scale_lightness(COLOURS[series[0]], 0.8))
        plt.text(max(line_x)*1.05, line_y[-1], f'r = {r:.3f}\nslope = {grad:.4f}\np = {lr_p:.4f}')
        patches.append(mpatches.Patch(color=COLOURS[series[0]], label=series))
        print('{} in {}: r = {:.3f}, p = {:.3f}'.format(x_label, data[i][0], r, p))

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title + x_label + ' vs ' + y_label)
    fix_legend(handles=patches)

    if save_file != '': plt.savefig(save_file)
    if show_ims: plt.show()
    plt.close()
    return results


def sliding_window_percentile(data, x_var, y_var, show_ims=False, save_to='', y_lim=None, width=0.1, resolution=25,
                              stack_plots=False):
    """ 2D plot showing the mean and SD bars in a sliding window of width width
    data: dataframe with column headers ['series', 'y_pred', 'y_true']
    y_pred column with numerical data type is used to split by percentile
    y_true column with numerical data type will be plotted on the y-axis
    width: number in [0,1] defining the size of the percentiles to use to calculate mean value
    resolution: number of windows to use in total
    output: 2D plot
    """
    if stack_plots:
        fig = plt.figure(figsize=(len(data)*8, 8))
        grid = plt.GridSpec(1, len(data), hspace=0, wspace=0.1)
        axs = [None for i in range(len(data))]
    else:
        fig = plt.figure(figsize=(8, 8))
    lines = []
    for i in range(len(data)):
        series, subset = data[i]
        if stack_plots:
            if i > 0:
                axs[i] = fig.add_subplot(grid[0, i], sharey=axs[0], sharex=axs[0])
                plt.setp(axs[i].get_yticklabels(), visible=False)
            else:
                axs[i] = fig.add_subplot(grid[0, 0])
            axs[i].set_title(str(abbreviate(series)))

        # calculate windows
        x, y_lower, y_mean, y_upper = [], [], [], []
        for start in range(int(resolution * (1-width))):
            window = er.get_window(subset, x_var, start/resolution, start/resolution + width)
            y = np.mean(window[y_var].astype('float32'))
            y_mean.append(y)
            stderr = sem(window[y_var].astype('float32'))
            y_lower.append(y - stderr)
            y_upper.append(y + stderr)
            x.append(np.mean(window[x_var]))
        X_sp, Y_sp = smoothed(x, y_mean)
        _, Y_up = smoothed(x, y_upper)
        _, Y_lo = smoothed(x, y_lower)

        # plot
        if stack_plots:
            axs[i].plot(X_sp, Y_sp, linewidth=3, color=COLOURS[series[0]])
            axs[i].fill_between(X_sp, Y_lo, Y_up, alpha=.3, color=COLOURS[series[0]])
            axs[i].set_xlabel(x_var)
        else:
            plt.plot(X_sp, Y_sp, linewidth=3, color=COLOURS[series[0]])
            plt.fill_between(X_sp, Y_lo, Y_up, alpha=.3, color=COLOURS[series[0]])
            r, p = pearsonr(subset[x_var].astype(float), subset[y_var].astype(float))
            plt.annotate(f'r = {r:.3f}\np = {p:.4f}', xy=(X_sp[-1], Y_up[-1]), color=COLOURS[series[0]])
            plt.xlabel(x_var)
        lines.append(mlines.Line2D([], [], color=COLOURS[series[0]], label=series))

    if stack_plots:
        axs[0].set_ylabel(y_var)
    else:
        plt.gca().set_ylabel(y_var)

    if y_lim is not None:
        plt.gca().set_ylim([y_lim[0], y_lim[1]])
    fix_legend(handles=lines, ncol=len(data))
    if save_to != '': plt.savefig(save_to)
    if show_ims: plt.show()
    plt.close()


def smoothed(x, y, resolution=200):
    zipped = pd.DataFrame({'x':x, 'y':y})
    dup = zipped.duplicated(subset=['x'])
    zipped = zipped[~dup]
    x, y = zipped['x'], zipped['y']
    X_Y_Spline = make_interp_spline(x, y)
    X_sp = np.linspace(min(x), max(x), resolution)
    Y_sp = X_Y_Spline(X_sp)
    return (X_sp, Y_sp)


def scale_lightness(colname, scale_l):
    rgb = mcolors.ColorConverter.to_rgb(colname)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, l * scale_l, s)


def multiseries_contour(data, x, y, series_list, x_label=None, y_label=None, bw_adjust=1., contour_threshold=0.6,
                        x_lim=None, y_lim=None, show_ims=False, save_to=''):
    """ plot multiple series of data in a scatter plot
    data: dataframe, with column headers ['series name', x, y]
    """
    colour_maps = {'green': 'Greens',
                   'red': 'Reds',
                   'orange': 'YlOrBr',
                   'blue': 'Blues',
                   'magenta': 'RdPu'}

    print('Calculating multi-series contour plot')
    g = sns.JointGrid(data=data, x=x, y=y, height=12)
    g.figure.set_size_inches(8, 10)
    patches = []

    for i in range(len(series_list)):
        subset = data[data['series'] == series_list[i]]
        colour = COLOURS[series_list[i]]
        sns.kdeplot(data=subset, x=x, y=y, cmap=colour_maps[colour], bw_adjust=bw_adjust, fill=True, alpha=0.5,
                    ax=g.ax_joint, levels=7, thresh=contour_threshold)
        sns.kdeplot(data=subset, x=x, color=colour, bw_adjust=bw_adjust,
                    fill=True, common_norm=False, legend=False, alpha=.3, linewidth=0, ax=g.ax_marg_x)
        sns.kdeplot(data=subset, y=y, color=colour, bw_adjust=bw_adjust,
                    fill=True, common_norm=False, legend=False, alpha=.3, linewidth=0, ax=g.ax_marg_y)
        patches.append(mpatches.Patch(color=colour, label=series_list[i]))

    fix_legend(handles=patches)

    # axis labels
    if x_label is None:
        g.ax_joint.set_xlabel(x)
    else:
        g.ax_joint.set_xlabel(x_label)
    if y_label is None:
        g.ax_joint.set_ylabel(y)
    else:
        g.ax_joint.set_ylabel(y_label)

    # axis ranges
    if y_lim is not None:
        g.ax_marg_y.set_ylim([y_lim[0], y_lim[1]])
    if x_lim is None:
        g.ax_marg_x.set_xlim([np.percentile(data[x], 1), np.percentile(data[x], 95)])
    else:
        g.ax_marg_x.set_xlim([x_lim[0], x_lim[1]])

    # for item in g.ax_joint.get_xticklabels() + g.ax_joint.get_yticklabels():
    #     item.set_fontsize(30)

    if save_to != '': plt.savefig(save_to + ' contour bw=' + str(bw_adjust) + '.svg')
    if show_ims: plt.show()
    plt.close()


def radial_plot(data, series, show_ims=False, save_to=''):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_direction(-1)
    ax.set_rticks([])
    for i in range(len(data)):
        mean_angle = math.atan2(sum(data[i].apply(math.sin)), sum(data[i].apply(math.cos)))
        c = ax.scatter(data[i], np.ones(len(data[i])), c=COLOURS[series[i]], s=5, alpha=0.7)
        l = ax.plot((mean_angle, mean_angle), (0, 1,), c=COLOURS[series[i]])

    if show_ims: plt.show()
    if save_to != '': plt.savefig(save_to)
    plt.close()


def violin_plot(df, order, x_var, y_var, bw=0.5, stat_test=None, show_ims=False, save_to='', **kwargs):
    n = len(df[x_var].value_counts())
    if x_var != 'series': n *= len(df['series'].value_counts())
    fig = plt.figure(figsize=(2.5*n, 8))
    ax = sns.violinplot(data=df, x=x_var, y=y_var, hue='series', common_norm=True, cut=0,
                        bw_adjust=bw, order=order, inner_kws={'box_width':6, 'whis_width':2}, **kwargs)
    if stat_test is not None:
        cols = list(set(df[x_var]))
        pairs = [ [[cols[i], cols[j]] for j in range(i+1, len(cols))] for i in range(len(cols)) ]
        pairs = sum(pairs, [])
        if len(pairs) > 0:
            annotator = Annotator(ax, pairs, data=df, x=x_var, y=y_var)
            annotator.configure(test=stat_test, text_format='simple', loc='inside', show_test_name=False, verbose=0)
            annotator.apply_and_annotate()

    # add n_obs
    nobs = df[x_var].value_counts().values / len(df[x_var])
    nobs = [f'{x*100:.0f}%' for x in nobs.tolist()]
    pos = range(len(nobs))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick], 1.05, nobs[tick], horizontalalignment='center')

    ax.legend().remove()
    fix_legend()
    if show_ims: plt.show()
    if save_to != '': plt.savefig(save_to)
    plt.close()


def box_plot(data, var, stat_test=None, rotate_labels=0, xfontsize=12, show_ims=False, save_to='', **kwargs):
    n = len(data[kwargs['x']].value_counts())
    fig = plt.figure(figsize=(2*n, 8))
    ax = sns.boxplot(data=data, y=var, width=0.5, boxprops=dict(alpha=.8), **kwargs)
    ax.legend().remove()

    # statistical comparison
    if stat_test is not None:
        cols = list(set(data[kwargs['x']]))
        pairs = [ [[cols[i], cols[j]] for j in range(i+1, len(cols))] for i in range(len(cols)) ]
        pairs = sum(pairs, [])
        if len(pairs) > 0:
            annotator = Annotator(ax, pairs, data=data, x=kwargs['x'], y=var)
            annotator.configure(test=stat_test, text_format='simple', loc='inside', show_test_name=False, verbose=0)
            annotator.apply_and_annotate()

    # add n_obs
    nobs = data[kwargs['x']].value_counts().values / len(data[kwargs['x']])
    nobs = [f'{x*100:.0f}%' for x in nobs.tolist()]
    pos = range(len(nobs))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick], 1.05, nobs[tick], horizontalalignment='center')

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_labels, fontsize=xfontsize, ha="right")

    fix_legend()
    if show_ims: plt.show()
    if save_to != '': plt.savefig(save_to)
    plt.close()


def bar_plot(data, series, p=None, show_ims=True, save_to=''):
    fig, ax = plt.subplots(figsize=(8, 8))
    bottom = np.zeros(len(series))
    for ser, values in data.items():
        ax.bar(series, values, label=ser, bottom=bottom, width=0.5)
        bottom += values
    if p is not None:
        annotator = Annotator(ax, [series], x=series, y=bottom)
        annotator.configure(loc='outside',verbose=0)
        annotator.set_custom_annotations([f'p = {p:.4f}'])
        annotator.annotate()
    fix_legend()

    if show_ims: plt.show()
    if save_to != '': plt.savefig(save_to)
    plt.close()


def histo_plot(data, series_label, var, show_ims=False, save_to=''):
    fig, ax = plt.subplots(figsize=(8, 8))
    df = data[[series_label, var]]
    sns.histplot(df, x=var, hue=series_label)

    fix_legend()
    if show_ims: plt.show()
    if save_to != '': plt.savefig(save_to)
    plt.close()


def bp_meds_plot(data, output_folder, show_ims=True):
    groups = ['HCMR sarcomere negative', 'HCMR sarcomere positive']
    df = data[['series', 'DBP', 'BP_meds', 'y_pred']]
    df = df.rename(columns={'series': 'group', 'BP_meds':'series'})
    for group in groups:
        grp_data = df[df['group']==group]
        grp_data = grp_data.dropna()
        multiseries_contour(grp_data, 'DBP', 'y_pred', [0,1], bw_adjust=0.7, y_lim=[0,1],
                            show_ims=show_ims, save_to=output_folder+f'BP_meds contour plot ({str(group)}).svg')
        subset_data = [[[i], grp_data[grp_data['series']==i]] for i in [0,1]]
        multiseries_scatter(subset_data, 'DBP', 'y_pred', show_ims=show_ims,
                            save_file=output_folder+f'BP_meds scatter plot ({str(group)}).svg')


def kaplan_meier(df, v, series, show_ims=False, save_to=''):
    df = get_km_data(df, v, censor_date='01-01-2023')

    # K-M fit and plot
    duration = df['duration']
    event = df[v]
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event)

    def plot_km(df, col, duration, event, lr):
        fig, ax = plt.subplots(figsize=(14, 12))
        for r in sorted(df[col].unique()):
            ix = df[col] == r
            kmf.fit(duration[ix], event[ix], label=r)
            kmf.plot_survival_function(ax=ax)
        # plt.text(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1], f'p = {lr.p_value[0]:.4f}')
        ax.set_title(f'Kaplan-Meier curves for {v}\np = {lr.p_value[0]:.4f}')
        ax.set_xlabel('time elapsed (days)')
        ax.set_ylabel('proportion without event')
        ax.get_figure().savefig(save_to + f'KM curve.svg')
        plt.close()

    # Helper function for printing out Log-rank test results
    def get_logrank(col):
        log_rank = pairwise_logrank_test(duration, df[col], event)
        return log_rank

    lr = get_logrank('series')
    plot_km(df, 'series', duration, event, lr)
    return [v, series, 'KM log-rank', lr.test_statistic[0], lr.p_value[0], '']


def get_km_data(df, v, censor_date='today'):
    oc = pd.read_csv('G:\\HCMR\\Label data\\Nov 2023\\HCMRAdjDataset_2023-11-13.csv')
    oc = pd.DataFrame(merge_datasets.hcmr_fu_mapping(oc))
    oc = oc[oc[v] == 1][[v, 'daystoevent', 'sitePatID']]  # select all cases with event
    oc.index = oc['sitePatID']

    # if more than one record take min no of days
    oc = merge_datasets.condense_rows(oc, min)
    oc = oc.drop(columns=['sitePatID', v])  # tidy up

    # if no event then calculate time since ICF
    labels = pd.read_csv(params.label_files['combined'], index_col=0).loc[:, ['icfDate']]
    labels['icfDate'] = pd.to_datetime(labels['icfDate'], dayfirst=True)
    labels['daystonow'] = (pd.to_datetime(censor_date, dayfirst=True) - labels['icfDate']).dt.days
    labels = labels.join(oc)

    # merge all data
    df = df.merge(labels, how='left', left_index=True, right_index=True)
    df = df.fillna({'daystoevent':0})
    df['duration'] = np.where(df['daystoevent']==0, df['daystonow'], df['daystoevent'])
    return df


def visualise_lr(vars, res, show_ims=True, save_to=''):
    # setup data table
    means = res.params[[v for v in vars]]
    interactions = res.params[[f'group:{v}' for v in vars]]
    interactions.index = vars
    int_pval = res.pvalues[[f'group:{v}' for v in vars]]
    int_pval.index = vars
    vars = [verbose_label(x) for x in vars]
    sn = pd.DataFrame(data={'vars':vars, 'coefficient':(means-interactions).values, 'series':len(vars)*['SN-HCM']})
    sp= pd.DataFrame(data={'vars':vars, 'coefficient':(means+interactions).values, 'series': len(vars) * ['SP-HCM']})
    data = pd.concat((sn,sp))

    # plot
    fig = plt.figure(figsize=(8, 1. * len(vars)), frameon=False)
    ax = sns.barplot(data, x='coefficient', y='vars', hue='series', orient='h', )
    sns.move_legend(ax, bbox_to_anchor=(0.5, -0.2), loc='upper center')

    # # annotate
    # for i in range(len(int_pval)):
    #     p = int_pval[i]
    #     txt = ''
    #     if p < 0.05: txt += '*'
    #     if p < 0.01: txt += '*'
    #     if p < 0.001: txt += '*'
    #     x=max(sn['coefficient'][i], sp['coefficient'][i])+0.002
    #     ax.annotate(txt, (x, i), va='center', color="k")

    # tidy
    ax.spines[['left']].set_position('zero')
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    plt.setp(ax.get_yticklabels(), transform=ax.get_yaxis_transform())
    ax.tick_params(left=False)
    plt.grid(True, 'major', 'x', color='lightgray')
    ax.set_axisbelow(True)
    plt.ylabel('')
    ax.get_legend().set_title("")
    plt.tight_layout()
    if show_ims: plt.show()
    if save_to != '': plt.savefig(save_to)
    plt.close()


def fix_legend(squash=0.7, x_offset=0.5, y_offset=0.2, loc='upper center', **kwargs):
    plt.tight_layout()
    plt.subplots_adjust(bottom=1-squash)
    leg = plt.gcf().legend(bbox_to_anchor=(x_offset, y_offset), loc=loc, **kwargs)

    if 'leg_line_width' in kwargs.keys():
        for line in leg.get_lines():
            line.set_linewidth(kwargs['leg_line_width'])


def abbreviate(series_list):
    mapping = {'UKBB normal BP': 'UKB NBP',
               'UKBB high BP': 'UKB HBP',
               'HCMR sarcomere negative': 'HCM SN',
               'HCMR sarcomere positive': 'HCM SP',
               'HCMR sarcomere negative hypertension': 'HCM SN HTN',
               'HCMR sarcomere negative no hypertension': 'HCM SN no HTN',
               'HCMR sarcomere positive no hypertension': 'HCM SP no HTN',
               'HCMR sarcomere positive hypertension': 'HCM SP HTN',}
    l = [mapping[s] for s in series_list]
    return l


def verbose_label(x):
    lookup = {'BMI1': 'BMI',
              'age1': 'age',
              'glsmean': 'GLS',
              'log_max_LVOT_grad': 'log max LVOT gradient',
              'log_NTproBNP': 'NTproBNP',
              'wallthkmax': 'max wall thickness',
              'log_lge_total6': 'log LGE',
              }
    try:
        output = lookup[x]
    except KeyError:
        output = x.replace('_', ' ')
    return output
