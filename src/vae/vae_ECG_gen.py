import numpy as np

import ECG_plots as ep
import params as p


def gen_and_plot_ECG(vae, latent_vals, htn_score, fname='', caption='', output_folder=''):
    gen_ECG = ECG_generator(vae, latent_vals, htn_score)
    print('Plotting {}'.format(fname))
    ep.plot_iso_ecg(gen_ECG, {'lead order':p.import_lead_order, 'filename':caption},
                    save_name=output_folder+fname, display=False, t_scale=5)


def generate_multi_ECG(vae, latent_val_series, htn_score_series, series_titles, v_scale=1,
                       title='', save_to='', display=True):
    leads_series = []
    for i in range(len(latent_val_series)):
        gen_ECG = ECG_generator(vae, latent_val_series[[i], :], htn_score_series[[i], :])
        leads_series.append(gen_ECG)
    ep.plot_multiple(leads_series, {'lead order':p.import_lead_order, 'filename':title}, series_titles,
                     save_name=save_to, display=display, v_scale=v_scale, t_scale=5)
    return leads_series


def ECG_generator(vae, latent_vals, htn_score):
    gen_ECG = vae.decoder([[latent_vals], [htn_score]])
    gen_ECG = gen_ECG.numpy()
    gen_ECG = gen_ECG.reshape(gen_ECG.shape[1:])
    gen_ECG = np.transpose(gen_ECG)
    gen_ECG = complete_ECG(gen_ECG)
    return gen_ECG


def complete_ECG(leads):
    """ takes 8 lead ECG with leads [I, II, V1, V2, V3, V4, V5, v6]
    return 12 lead ECG with additional leads calculated """
    avf = 2 / (3**0.5) * leads[1]              # aVF = 2/sqrt(3) * II
    avl = (3**0.5)/2 * leads[0] - 0.5 * avf    # aVL = sqrt(3)/2 * I + 1/2 * -aVF
    avr = (3**0.5)/2 * -leads[0] - 0.5 * avf   # aVR = sqrt(3)/2 * -I + 1/2 * -aVF
    iii = 0.5 * - leads[0] + (3**0.5)/2 * avf
    avf = avf.reshape((1,) + avf.shape)
    avl = avl.reshape((1,) + avl.shape)
    avr = avr.reshape((1,) + avr.shape)
    iii = iii.reshape((1,) + iii.shape)
    return np.concatenate((leads[:2], iii, avr, avl, avf, leads[2:]), axis=0)
