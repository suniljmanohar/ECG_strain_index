import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import params
from plots_2D import fix_legend


mpl.rcParams.update({'font.size': 10})


def format_ecg_plot(graph, v_scale=1, t_scale=1, grids=True, minor_ticks=True):
    """ adds formatting to a plotted ECG to make it look conventional """
    v_min, v_max = -400 / v_scale, 400 / v_scale  # standard is -400, 400
    graph.set_ylim([v_min, v_max])
    if grids:
        graph.minorticks_on()
        major_ticks_x = np.arange(graph.get_xlim()[0], graph.get_xlim()[1], 0.2 / params.T_SCALE/t_scale)
        minor_ticks_x = np.arange(graph.get_xlim()[0], graph.get_xlim()[1], 0.04 / params.T_SCALE/t_scale)
        big_squares = int((v_max - v_min) / 100)
        major_ticks_y = np.linspace(v_min, v_max, big_squares + 1)
        minor_ticks_y = np.linspace(v_min, v_max, 5 * big_squares + 1)
        graph.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.6)
        graph.set_xticks(major_ticks_x)
        graph.set_yticks(major_ticks_y)
        if minor_ticks:
            graph.grid(which='minor', linestyle=':', linewidth='0.5', color='red', alpha=0.3)
            graph.set_xticks(minor_ticks_x, minor=True)
            graph.set_yticks(minor_ticks_y, minor=True)
        graph.tick_params(which='both', top=False, left=False, right=False, bottom=False)
    graph.set_yticklabels([])
    graph.set_xticklabels([])
    graph.set_axisbelow(True)


def plot_ecg(leads, metadata, save_name='', display=True, truncate=False):
    """ plots a 12-lead ECG in conventional layout and highlights R waves """
    t_scale_fac = 1 # params.HyperParams(params.subsets['all']).t_scale_fac

    # rearrange leads to ECG positions
    max_x_width = 1300 // t_scale_fac  # length to truncate leads to if required assuming sample rate = 500Hz
    r_peaks = metadata['r waves']
    fig = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(4, 4, wspace=0)
    axs = [[None for i in range(4)] for j in range(4)]

    """ iterate over 12 leads in 4 columns """
    for x in range(3):
        for y in range(4):
            if y > 0:
                axs[x][y] = fig.add_subplot(grid[x, y], sharey=axs[x][0])
            else:
                axs[x][y] = fig.add_subplot(grid[x, y])
            axs[x][y].plot(leads[x + 3 * y], 'k-')
            axs[x][y].set_title(metadata['lead order'][x + 3 * y])
            format_ecg_plot(axs[x][y], t_scale=t_scale_fac)
            if len(leads[x + 3 * y]) > max_x_width and truncate:
                axs[x][y].set_xlim(0, max_x_width)
            for a in r_peaks[x + 3 * y]:
                axs[x][y].plot(a, 0, 'rv')  # height was metadata['minmax'][x + 3 * y][1]
            axs[x][y].plot(metadata['selected Rs'][x + 3 * y], 0, 'bv')

    """ add rhythm strip at bottom """
    r_strip = fig.add_subplot(grid[3, :])
    r_strip.plot(leads[metadata['rhythm strip']], 'k-')
    max_deflection = metadata["minmax"][metadata['rhythm strip']][1]
    for a in r_peaks[metadata['rhythm strip']]:
        r_strip.plot(a, 0, 'rv')
        r_strip.plot(metadata['selected Rs'][metadata['rhythm strip']], max_deflection, 'bv')
    r_strip.set_title(params.import_lead_order[metadata['rhythm strip']])
    format_ecg_plot(r_strip, t_scale=t_scale_fac)

    fig.suptitle(metadata['filename'])
    if save_name != '': plt.savefig(save_name)
    if display: plt.show()
    plt.close()


def plot_iso_ecg(iso_leads, metadata, save_name='', display=True, t_scale=1):
    """ plots 12 isolated complexes in conventional ECG layout, no rhythm strip"""
    fig = plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(3, 4, wspace=0)
    axs = [[None for i in range(4)] for j in range(3)]
    for x in range(3):
        for y in range(4):
            if y > 0:
                axs[x][y] = fig.add_subplot(grid[x, y], sharey=axs[x][0])
            else:
                axs[x][y] = fig.add_subplot(grid[x, y])
            axs[x][y].plot(iso_leads[x + 3 * y], 'k-')
            axs[x][y].set_title(metadata['lead order'][x + 3 * y])
            format_ecg_plot(axs[x][y], t_scale=t_scale)
    plt.suptitle(metadata['filename'], horizontalalignment='left', x=0, fontsize=30, fontname='monospace')
    if save_name != '': plt.savefig(save_name)
    if display: plt.show()
    plt.close()


def plot_iso_ecg_vert(iso_leads, metadata, save_name='', display=True):
    """ plots n isolated complexes in vertical layout, no rhythm strip"""
    n_leads = len(iso_leads)
    fig = plt.figure(figsize=(4, 24))
    grid = plt.GridSpec(n_leads, 1, hspace=0.5)
    axs = [None for j in range(n_leads)]
    for x in range(n_leads):
        axs[x] = fig.add_subplot(grid[x, 0])
        axs[x].plot(iso_leads[x], 'k-')
        axs[x].set_title(metadata['lead order'][x])
        format_ecg_plot(axs[x])
    fig.suptitle(metadata['filename'])
    if save_name != '': plt.savefig(save_name + '.svg')
    if display: plt.show()
    plt.close()


def plot_both(fname, iso_leads, leads, metadata, plotting):
    if plotting['images']:
        ecg_path = plotting['save images'] * (params.folder['hcmr'] + "ECG images\\" + fname + ".svg")
        iso_path = plotting['save images'] * (params.folder['hcmr'] + "ECG images\\isolated\\" + fname + ".svg")
        plot_ecg(leads, metadata, save_name=ecg_path, display=plotting['display images'],
                 truncate=plotting['truncate plots'])
        plot_iso_ecg(iso_leads, metadata, save_name=iso_path, display=plotting['display images'])


def plot_multiple(leads_series, metadata, series_titles, save_name='', display=False, v_scale=1, t_scale=1):
    """ plots multiple sets of 12 isolated complexes on same set of axes in conventional ECG layout, no rhythm strip"""
    colours = ['b', 'm', 'r', 'c', 'g']
    fig = plt.figure(figsize=(20, 18))
    grid = plt.GridSpec(3, 4, wspace=0, hspace=0)
    axs = [[None for i in range(4)] for j in range(3)]
    handles = []
    for x in range(3):
        for y in range(4):
            if y > 0:
                axs[x][y] = fig.add_subplot(grid[x, y], sharey=axs[x][0])
            else:
                axs[x][y] = fig.add_subplot(grid[x, y])
            for i in range(len(leads_series)):
                hnd, = axs[x][y].plot(leads_series[i][x + 3 * y], color=colours[i], linestyle='-', linewidth=1.2,
                                      label=series_titles[i])
                if x == 0 and y == 0: handles.append(hnd)
            axs[x][y].set_title(metadata['lead order'][x + 3 * y], loc='left', pad=-40, y=1.001)
            format_ecg_plot(axs[x][y], v_scale=v_scale, t_scale=t_scale)
    plt.suptitle(metadata['filename'], horizontalalignment='left', x=0, fontsize=36)
    fix_legend(handles=handles, squash=0.9, x_offset=0.5, y_offset=0.1, loc='upper center', ncols=3)   # x_offset=0.73, y_offset=0.5
    if save_name != '': plt.savefig(save_name)
    if display: plt.show()
    plt.close()


def traversals(ecgs, lead_selection, series_titles, title='', display=False, save_to='', v_scale=1):
    colours = ['r', 'm', 'b', 'c', 'g']
    lead_args = [params.import_lead_order.index(l) for l in lead_selection]
    ecgs = ecgs[:, :, lead_args, :]
    n_rows, n_traces, n_cols = ecgs.shape[:3]
    fig = plt.figure(figsize=(12, 32))
    grid = plt.GridSpec(n_rows, n_cols, wspace=0.)
    axs = [[None for i in range(n_cols)] for j in range(n_rows)]
    handles = []
    for x in range(n_rows):
        for y in range(n_cols):
            if y > 0:
                axs[x][y] = fig.add_subplot(grid[x, y], sharey=axs[x][0])
            else:
                axs[x][y] = fig.add_subplot(grid[x, y])
            for i in range(n_traces):
                hnd, = axs[x][y].plot(ecgs[x, i, y], color=colours[i], linestyle='-', linewidth=0.3,
                                      label=series_titles[i])
                if x == 0 and y == 0: handles.append(hnd)
            format_ecg_plot(axs[x][y], minor_ticks=False, v_scale=v_scale)
            if x == 0:
                axs[x][y].set_title(lead_selection[y])
        axs[x][0].set_ylabel(f'Dimension {x}', rotation='horizontal', ha='right')

    plt.suptitle(title, horizontalalignment='left', x=0, fontsize=30, fontname='monospace')
    fig.legend(handles=handles, loc='lower center')
    if save_to != '': plt.savefig(save_to)
    if display: plt.show()
    plt.close()


def plot_long_ecg(ecg, display=False, save_ims=''):
    """ plots 12 long ECG leads vertically ordered. ecg[0] is np array of shape (n_leads, lead_length) with ECG waveforms.
	ecg[1] is ECG metadata dict containing 'filename', 'lead order' and 'r waves'. Returns None """
    n = ecg[0].shape[0]
    lead_data, metadata = ecg  # unpack lead data and metadata
    fig = plt.figure(figsize=(12, 20))
    grid = plt.GridSpec(n, 1, hspace=0, wspace=0)
    axs = [None for i in range(n)]

    """ iterate over 12 leads """
    for i in range(n):
        axs[i] = fig.add_subplot(grid[i, 0])
        axs[i].plot(lead_data[i], 'k-')
        axs[i].set_title(metadata['lead order'][i])
        format_ecg_plot(axs[i], v_scale=1)
        for a in metadata['r waves'][i]:
            axs[i].plot(a + 500*2.5*(i//3), 300, 'rv')
        # if len(metadata['r waves'][i]) > 1: axs[i].plot(metadata['selected Rs'][i], 300, 'bv')

    fig.suptitle(metadata['filename'])
    if save_ims != '': plt.savefig(save_ims + metadata['filename'] + ".svg")
    if display: plt.show()
    plt.close()


def r_selection_plot(ecg, md, display=False, save_ims=''):
    n = ecg.shape[0]
    axs = [None] * n

    fig = plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(n//2, 2, hspace=0, wspace=0)

    """ iterate over 12 leads """
    for i in range(n):
        axs[i] = fig.add_subplot(grid[i%6, i//6])
        axs[i].plot(ecg[i], 'k-')
        axs[i].set_title(md['lead order'][i])
        #format_ecg_plot(axs[i], v_scale=1)
        for r_number in range(len(md['r waves'][i])):
            axs[i].annotate(str(i)+'-'+str(r_number), (md['r waves'][i][r_number], ecg[i][md['r waves'][i][r_number]]),
                            color='b') #, arrowprops=dict(facecolor='blue', shrink=1))
        if len(md['r waves'][i]) > 1: axs[i].plot(md['selected Rs'][i], 350, 'gv')

    fig.suptitle(md['filename'])
    if save_ims != '': plt.savefig(save_ims + md['filename'] + ".svg")
    if display: plt.show()
    plt.close()