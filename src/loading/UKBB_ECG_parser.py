import numpy as np
import time
import os
import xmltodict as xtd
import pickle

import ECG_plots
import params as p, signal_analysis as sig, my_timer, ECG_plots as eplt


def main():
    folder = p.folder['ukbb'] + 'ECG\\'
    iso_length = p.lead_length['hcmr']
    stop_at = ''
    subset = ['5724810']
    subset = None
    manual_adjust = False
    import_adjustments = True
    display_ims = False
    save_ims = True
    ac_filter = None  # 50/None  frequency for IIR notch filter - done in preprocessing module now

    import_all(folder, iso_length, ac_filter,
               start_at='1000037_20205_2_0.xml',    # 1000037_20205_2_0
               stop_at=stop_at,
               subset=subset,
               manual_adjust=manual_adjust,
               import_adjustments=import_adjustments,
               verbose=True,
               display_ims=display_ims,
               save_ims=save_ims)

    print('REMEMBER TO MERGE DATASETS NOW!')


def import_all(folder, iso_length, ac_filter, start_at='', stop_at='', subset=None, manual_adjust=False,
               import_adjustments=True, verbose=False, display_ims=False, save_ims=False):
    plotting = display_ims or save_ims  # whether to plot 12-lead ECGs for each file

    t0 = time.time()
    ecgs, median_ecgs, iso_ecgs, iso_ecgs_cols, metadata = [], [], [], [], []
    file_counter = 0

    if import_adjustments:
        with open(p.folder['ukbb'] + 'Manual adjustments.npy', 'rb') as f:
            adjustments = pickle.load(f)
    else:
        adjustments = {}

    go = False
    for fname in os.listdir(folder):
        if fname == start_at: go = True
        if fname == stop_at: go = False
        if subset is not None:
            go = fname[:7] in subset
        if go:
            if fname[-3:] == 'xml' and fname[8:13] == '20205':
                if verbose: print('{}: processing... '.format(fname), end='')
                f = open(folder + fname, "rt")
                raw_input = f.read()
                f.close()

                # extract ECG and metadata
                ecg, median_ecg, md = parse_xml(raw_input, verbose=verbose)
                md['filename'] = fname

                # get median ECG
                median_ecg = sig.trim(median_ecg, iso_length)

                # filter electrical noise
                if ac_filter:
                    ecg = sig.ac_filter(ecg, ac_filter)
                    median_ecg = sig.ac_filter(median_ecg, ac_filter)

                # manual adjustments
                if manual_adjust:
                    ecg, md, adjustments = adjust_rs(ecg, md, adjustments)

                # extract single complexes
                short_ecg, short_md = ecg, md
                if fname not in adjustments.keys():
                    short_ecg, short_md = sig.long_to_paper(short_ecg, short_md)
                else:
                    md['lead heads'] = [0] * 12
                iso_ecg_cols, iso_md = sig.ukbb_iso_cols(short_ecg, short_md, iso_length,
                                                         adjustments=adjustments, verbose=verbose)

                # add to list
                ecgs.append(ecg)
                median_ecgs.append(median_ecg)
                iso_ecgs_cols.append(iso_ecg_cols)
                metadata.append(md)

                # plot and save
                if plotting:
                    eplt.plot_long_ecg((ecg, md), display=display_ims, save_ims=folder + 'images\\')
                    eplt.plot_iso_ecg(iso_ecg_cols, iso_md, folder + 'images\\iso_cols\\' + fname[:-4], display=display_ims)
                    # eplt.plot_iso_ecg(median_ecg, md, folder + 'images\\medians\\' + fname[:-4], display=display_ims)

                if verbose: print('Done.')
                file_counter += 1

    # save output files
    if subset is None:
        np.save(folder + 'Processed ECGs\\all_rest_ECGs.npy', ecgs, allow_pickle=True)
        np.save(folder + 'Processed ECGs\\all_median_ECGs.npy', median_ecgs, allow_pickle=True)
        # np.save(folder + 'Processed ECGs\\all_rest_ECGs_iso.npy', iso_ecgs, allow_pickle=True)
        np.save(folder + 'Processed ECGs\\all_rest_ECGs_iso_cols.npy', iso_ecgs_cols, allow_pickle=True)
        np.save(folder + 'Processed ECGs\\metadata.npy', metadata, allow_pickle=True)
    print('%d files processed' % file_counter)
    my_timer.show_t(t0)


def parse_xml(input_data, detect_Rs=True, verbose=False):
    """ Takes input as raw xml data read from file (Cardiosoft specification), returns list of 12 lead waveforms
    and metadata. """
    md = {}  # metadata dictionary
    data = xtd.parse(input_data)['CardiologyXML']

    full_lead_nodes = [['StripData', 'WaveformData'], ['Strip', 'StripData', 'WaveformData']]
    median_lead_nodes = [['RestingECGMeasurements', 'MedianSamples', 'WaveformData'], []]
    full_leads = get_lead_data(data, full_lead_nodes)
    median_leads = get_lead_data(data, median_lead_nodes)

    md = get_metadata(data, md)

    # get any annotations?

    # detect R waves
    if detect_Rs:
        md['r waves'] = sig.find_r_waves((full_leads, md))
    else:
        md['r waves'] = [[] for i in range(12)]

    # get_full_disclosure(data)
    return np.array(full_leads, dtype=np.int16), np.array(median_leads, dtype=np.int16), md


def adjust_rs(ecg, md, adjustments, verbose=False):
    orig_ecg = ecg.copy()
    if md['filename'] in adjustments.keys():
        md['selected Rs'] = adjustments[md['filename']]
        if verbose: print('Imported adjustments for ', md['filename'])
    else:
        md['selected Rs'] = [0] * 12

    # plot full trace in all 12 leads with numbered R waves
    working = True
    adjusted = False
    while working:
        ECG_plots.r_selection_plot(ecg, md, save_ims='', display=True)
        ipt = input('\nFunction (s=select, f=flatten, c=copy, r=reset) followed by params (blank to finish): ')
        if ipt == '':
            working = False
        else:
            fn, par = ipt[0], ipt[1:]
            if fn == 's':
                rs = par.split(',')
                assert len(rs)==12
                r_vals = [md['r waves'][i][int(rs[i].strip())] for i in range(12)]
                md['selected Rs'] = r_vals
                adjusted = True
            if fn == 'f':
                lead_n, r_n = list(map(int, par.split(',')))
                f_left, f_right = max(md['r waves'][lead_n][r_n]-100,0), min(md['r waves'][lead_n][r_n]+100,5000)
                flattened = np.linspace(ecg[lead_n][f_left], ecg[lead_n][f_right], f_right-f_left)
                ecg[lead_n] = np.concatenate((ecg[lead_n][:f_left], flattened, ecg[lead_n][f_right:]))
                md['r waves'] = sig.find_r_waves((ecg, md), md['t scale'])
            if fn == 'c':
                from_lead, to_lead, r_n = list(map(int, par.split(',')))
                md['r waves'][to_lead] += [md['r waves'][from_lead][r_n]]
                md['r waves'][to_lead] = sorted(md['r waves'][to_lead])
            if fn == 'r':
                md['selected Rs'] = [0] * 12
                ecg = orig_ecg


    if adjusted:
        adjustments[md['filename']] = md['selected Rs']
        with open(p.folder['ukbb'] + 'Manual adjustments.npy', 'wb') as f:
            pickle.dump(adjustments, f)
        print('Manual adjustment finished. Adjustments saved to file.')
    else:
        print('Manual adjustment finished. No adjustments saved.')

    return ecg, md, adjustments


def get_full_disclosure(data):
    fd = data['FullDisclosure']['FullDisclosureData']['#text']
    if fd[-1] == ',': fd = fd[:-1]  # exclude trailing comma
    fd_vals = [int(x) for x in fd.split(",")]
    output_leads_2 = [sum([fd_vals[i * 6000 + j * 500:i * 6000 + (j + 1) * 500] for j in range(12)], []) for i in
                      range(12)]
    return output_leads_2


def get_lead_data(data, nodes):
    leads = [[] for i in range(12)]
    raw_lead_data = get_xml_node(data, nodes)

    # check number of leads
    if len(raw_lead_data) != 12:
        raise ValueError('Only %d leads found' % len(raw_lead_data))

    # split data into 12 leads and convert string into individual values
    else:
        for i in range(len(raw_lead_data)):
            lead_n = p.import_lead_order.index(raw_lead_data[i]['@lead'].upper())
            lead_data_string = raw_lead_data[i]['#text']
            lead_vals = [int(x) for x in lead_data_string.split(",")]  # handle error if int() fails #TODO
            # centre baseline for each lead
            m = np.mean(lead_vals)
            lead_vals = [x-m for x in lead_vals]
            leads[lead_n] = lead_vals

    # check all leads same length
    if max([len(leads[i]) for i in range(12)]) != min([len(leads[i]) for i in range(12)]):
        raise ValueError('Leads not equal length')
    return leads


def get_xml_node(data, node_list):
    output = data
    try:
        for x in node_list[0]:
            output = output[x]
    except KeyError:
        output = data[:]
        try:
            for x in node_list[1]:
                output = output[x]
        except:
            raise ValueError('No lead data found!')
    return output


def get_metadata(data, md):
    md['sample rate'] = float(data['StripData']['SampleRate']['#text'])
    md['t scale'] = 1. / md['sample rate']
    md['v scale'] = float(data['StripData']['Resolution']['#text'])
    md['lead order'] = p.import_lead_order
    md['filter 50Hz'] = data['FilterSetting']['Filter50Hz']
    md['filter 60Hz'] = data['FilterSetting']['Filter60Hz']
    md['low pass'] = float(data['FilterSetting']['LowPass']['#text'])
    md['high pass'] = float(data['FilterSetting']['HighPass']['#text'])

    try:
        meas = data['RestingECGMeasurements']
        md['Heart rate'] = meas['VentricularRate']['#text'] + ' ' + meas['VentricularRate']['@units']
        md['P duration'] = meas['PDuration']['#text'] + ' ' + meas['PDuration']['@units']
        md['PR interval'] = meas['PQInterval']['#text'] + ' ' + meas['PQInterval']['@units']
        md['QRS duration'] = meas['QRSDuration']['#text'] + ' ' + meas['QRSDuration']['@units']
        md['QT interval'] = meas['QTInterval']['#text'] + ' ' + meas['QTInterval']['@units']
        md['QTc interval'] = meas['QTCInterval']['#text'] + ' ' + meas['QTCInterval']['@units']
        md['P axis'] = meas['PAxis']['#text'] + ' ' + meas['PAxis']['@units']
        md['R axis'] = meas['RAxis']['#text'] + ' ' + meas['RAxis']['@units']
        md['T axis'] = meas['TAxis']['#text'] + ' ' + meas['TAxis']['@units']
    except KeyError:
        print('ECG measurement not found...', end='')

    # check voltage and time scale
    if md['v scale'] != p.V_SCALE: print('Unexpected voltage scale: {}').format(md['v scale'])
    if md['t scale'] != p.T_SCALE: print('Unexpected voltage scale: {}').format(md['t scale'])

    return md


if __name__ == '__main__':
    main()
