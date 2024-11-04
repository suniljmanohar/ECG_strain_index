import os
import numpy as np
import math

import pickle
import params, signal_analysis as sig
from ECG_plots import plot_iso_ecg, plot_both

annotation_types = []  # ['TIME_PD_RR', 'WAVC_RWAVE', 'WAVC_TWAVE', 'WAVC_SWAVE', 'WAVC_QWAVE', ] # probably not needed


def main():
	start_at = ''
	stop_at = ''
	subset = ['042-0003']
	subset = None
	verbose = True
	manual_adjust = False
	import_adjustments = True
	plotting = {'images'	: True,
				'truncate plots': True,  # truncates long leads to a maximum of 2.5s
				'save images'	: True,
				'display images': False}
	if manual_adjust and not import_adjustments:
		print('**** WARNING: current setting will cause overwrite of existing adjustments ****')
	import_files_to_ECG(plotting, start_at=start_at, stop_at=stop_at, subset=subset,
						verbose=verbose, manual_adjust=manual_adjust, import_adjustments=import_adjustments)

	print("DON'T FORGET TO MERGE DATASETS NOW!")


def time_diff(t1, t2):
	"""subtracts ECG timecodes: t1 - t2, assumes time difference is less than 1 hour. Returns value in seconds"""
	t1_secs, t2_secs = t1 % 100, t2 % 100
	t1_mins, t2_mins = (t1 // 100) % 100, (t2 // 100) % 100
	return (t1_mins - t2_mins) * 60 + (t1_secs - t2_secs)


def find_val(string, start):
	""" finds next "value=" expression in xml file """
	first = string.find('value=', start) + 7
	last = string.find('"', first)
	return float(string[first:last]), last


def xml_to_ecg(fname, data, verbose=False):
	""" from xml file returns a list of 12 lists of lead values, and dict of metadata """
	leads, lead_count = [[] for i in range(12)], 0
	lead_heads, lead_increments, lead_voltages = [0 for i in range(12)], [0 for i in range(12)], [0 for i in range(12)]

	""" find start and end times """
	pointer = data.find("low", 0)
	start, pointer = find_val(data, pointer)
	end, pointer = find_val(data, pointer)

	""" iterate through file """
	while pointer > -1 and lead_count < 12:
		# check if head/increment data exists then store it
		next_head = data.find("head", pointer)
		next_lead = data.find('MDC_ECG_LEAD', pointer)
		if next_lead > next_head > -1:
			head, pointer = find_val(data, next_head)
			increment = find_val(data, pointer)[0]

		""" get voltage and lead data """
		pointer = next_lead
		ldend = data.find('"', pointer)
		lead_name = data[pointer + 13:ldend].upper()
		next_digits = data.find("digits", ldend)
		next_voltage = data.find("scale", ldend)
		if -1 < next_voltage < next_digits:
			voltage, pointer = find_val(data, next_voltage)
		else:
			voltage, pointer = 0, ldend
		i = data.find("digits", pointer) + 7
		j = data.find("digits", i)
		vals = [int(x) for x in data[i:j - 3].split(" ")]

		pointer = j

		""" check which lead this is and store the data """
		lead_n					= params.import_lead_order.index(lead_name)
		leads[lead_n]			= vals
		lead_heads[lead_n]		= max(0, time_diff(head, start))
		lead_increments[lead_n]	= increment
		lead_voltages[lead_n]	= voltage
		lead_count += 1

	""" check consistency of scales """
	if max(lead_increments) != min(lead_increments) or max(lead_voltages) != min(lead_voltages):
		raise sig.ConvertError('Inconsistent scales: {}, {}'.format(str(lead_increments), str(lead_voltages)))

	""" store all metadata """
	lead_lengths = list(len(leads[i]) for i in range(12))
	rhythm_strip = lead_lengths.index(max(lead_lengths))
	metadata = {"filename"     : fname,
				"start"        : start,
				"end"          : end,
				"length"       : time_diff(end, start) // params.T_SCALE,
				"lead heads"   : [int(x // params.T_SCALE) for x in lead_heads],
				"lead lengths" : lead_lengths,
				"minmax"       : list([min(leads[i]), max(leads[i])] for i in range(12)),
				"t scale"      : lead_increments[0],
				"v scale"      : lead_voltages[0],
				"rhythm strip" : rhythm_strip,
				"lead order"   : params.import_lead_order}

	""" find all annotations """
	annotations = []
	pointer = data.find('MDC_ECG_', pointer) + 8  # find next annotation
	while pointer > 7:
		if data[pointer:pointer + 10] in annotation_types:
			annot_val = time_diff(find_val(data, pointer + 10)[0], start) // params.T_SCALE
			annotations.append([data[pointer:pointer + 10], annot_val])  # annots in format [name, value]
		pointer = data.find('MDC_ECG_', pointer + 1) + 8
	metadata["annotations"] = annotations

	metadata["r waves"] = sig.find_r_waves((leads, metadata), t_scale=lead_increments[0])
	leads, metadata = scale_ECG(leads, metadata, verbose)
	return leads, metadata


def scale_ECG(leads, metadata, verbose=False):
	""" scales ECG leads and metadata to the standard scales in params.py """
	# scale leads
	scale = int(params.T_SCALE / metadata['t scale'])
	if scale == 0: raise sig.ConvertError('ECG t scale not compatible: {}'.format(metadata['t scale']))
	if metadata['t scale'] != params.T_SCALE:
		if verbose: print('WARNING: time scale is {} not {}. '.format(metadata['t scale'], params.T_SCALE), end='')
	if metadata['v scale'] != params.V_SCALE:
		if verbose: print('WARNING: voltage scale is {} not {}. '.format(metadata['v scale'], params.V_SCALE), end='')
	# leads = [leads[i][::scale] for i in range(12)]  # by skip not average
	leads = resample_mean(leads, scale)
	leads = [[x * metadata['v scale'] / 5.0 for x in leads[i]] for i in range(12)]

	# scale metadata
	metadata['length'] //= scale
	metadata['lead heads'] = [x // scale for x in metadata['lead heads']]
	metadata['lead lengths'] = [x // scale for x in metadata['lead lengths']]
	metadata['r waves'] = [[x // scale for x in y] for y in metadata['r waves']]
	metadata['annotations'] = [(x[0], x[1] // scale) for x in metadata['annotations']]
	return leads, metadata


def resample_mean(ecg, fac):
	""" takes ECG data in nested lists of shape (n_leads, lead_length) and scaling factor fac (must be integer)
	Resamples ECG along axis 1 by factor """
	output = []
	for lead in ecg:
		pad_size = math.ceil(len(lead) / fac) * fac - len(lead)
		padded_lead = lead + [np.nan] * pad_size
		resampled_lead = [np.nanmean(padded_lead[ x *fac: x * fac + fac]) for x in range(len(padded_lead) // fac)]
		output.append(resampled_lead)
	return output


def import_files_to_ECG(plotting, start_at='001-0001.xml', stop_at='', subset=None, verbose=False, manual_adjust=False,
						import_adjustments=False):
	""" iterates through all xml files in a folder and converts them to ECG data, then isolates individual complexes,
	plots both sets of data and saves the images, data and metadata """
	all_iso_ecgs, all_meta, all_single_leads, all_trunc = [], [], [], []
	warnings, errors = '', ''
	file_counter = 0
	ld_length = params.lead_length['hcmr']
	if import_adjustments:
		with open(params.folder['hcmr'] + 'Manual adjustments.npy', 'rb') as f:
			adjustments = pickle.load(f)
	else:
		adjustments = {}

	go = start_at==''
	for fname in os.listdir(params.folder['hcmr'] + "ECG data\\XML data to use\\"):
		file_counter += 1
		if fname == start_at: go = True
		if fname == stop_at: go = False
		if subset is not None:
			go = fname[:8] in subset
		if go:
			if verbose: print('\n{}: processing... '.format(fname), end='')
			f = open(params.folder['hcmr'] + "ECG data\\XML data to use\\" + fname, "rt")
			raw_input = f.read()
			f.close()

			try:
				leads, metadata = xml_to_ecg(fname, raw_input, verbose=verbose)

				# get isolated synced lead data
				if sum([len(l) >= 5000 for l in leads]) == 12:
					print(fname + ' has long leads')
					leads, metadata = sig.long_to_paper(leads, metadata)
					iso_leads, metadata = sig.ukbb_iso_cols(leads, metadata, ld_length,
															adjustments=adjustments, manual_adjust=manual_adjust,
															verbose=verbose)
				else:
					iso_leads, metadata = sig.sync_isolate(leads, metadata, adjustments=adjustments,
														   width=ld_length, manual_adjust=manual_adjust, verbose=verbose)
				all_iso_ecgs.append(iso_leads)
				all_meta.append(metadata)

				# plot processed data
				plot_both(fname, iso_leads, leads, metadata, plotting)

				# get truncated lead data
				if plotting['truncate plots']:
					trunc, trunc_md = sig.truncate_all_leads(leads, metadata, ld_length * 2)
					all_trunc.append(trunc)
					trunc_path = plotting['save images'] * \
								 (params.folder['hcmr'] + "ECG images\\truncated\\" + fname + ".png")
					plot_iso_ecg(trunc, metadata, save_name=trunc_path, display=plotting['display images'])

				if verbose: print('Done.', end='')
			except sig.ConvertError as e:
				if verbose: print('EXCLUDED: ' + e.message, end='')
				errors = errors + ' '.join([fname, e.message, '\n'])

	print(str(np.asarray(all_iso_ecgs).shape[0]) + " files successfully processed out of " + str(file_counter))
	print(np.asarray(all_iso_ecgs).shape)

	if subset is None:
		# np.save('g:\\dti ecgs\\' + "HCM_ECGs.npy", np.asarray(all_iso_ecgs))
		# np.save('g:\\dti ecgs\\' + "HCM_metadata.npy", np.asarray(all_meta))
		np.save(params.folder['hcmr'] + "HCMR ECGs.npy", np.asarray(all_iso_ecgs))
		np.save(params.folder['hcmr'] + "HCMR metadata.npy", np.asarray(all_meta))
		np.save(params.folder['hcmr'] + "HCMR ECGs truncated.npy", all_trunc)


def save_adjustments(md):
	output = {}
	for m in md:
		if m['adjusted R selection']:
			output[m['filename']] = m['selected Rs']
	np.save(params.folder['hcmr'] + 'Manual adjustments.npy', output)


if __name__ == '__main__':
	main()
