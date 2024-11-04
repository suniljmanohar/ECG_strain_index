from scipy.signal import butter, sosfilt, iirnotch, lfilter
import matplotlib.pyplot as plt
import numpy as np
import pickle

import ECG_plots
import params


class ConvertError(Exception):
	def __init__(self, message):
		self.message = message

	def __str__(self):
		return self.message


def highpass(lead, cutoff):  # removes baseline wander
	sos = butter(5, cutoff, btype="hp", fs=500, output="sos")
	filtered = sosfilt(sos, lead)
	return filtered


def lowpass(lead, cutoff):  # removes noise
	sos = butter(5, cutoff, btype="lp", fs=500, output="sos")
	filtered = sosfilt(sos, lead)
	return filtered


def ac_filter(ecg, freq):
	quality = 5
	output = []
	for lead in ecg:
		b, a = iirnotch(freq, quality, 1/params.T_SCALE)
		lead = lfilter(b, a, lead)
		output.append(lead)
	return np.array(output)


def triangle_conv(lead, width):
	y = []
	for x in range(len(lead) - width * 2):
		y.append((lead[x + width] - lead[x]) * (lead[x + width] - lead[x + width * 2]))
	return y


def find_r_waves(ecg, t_scale=params.T_SCALE):
	""" finds R waves in each lead using triangle convolution then find-and-flatten threshold """
	max_HR = 150  # in bpm - R waves closer together than this will be omitted. Baseline value 150
	threshold = 0.05  # deflections below this ratio will not be counted as R peaks - initial value 0.15
	width = 10 * int(params.T_SCALE/t_scale)		# width of triangle convolution, scaled for sampling rate
	hp_cutoff, lp_cutoff = 5, 150
	leads, metadata = ecg
	min_gap = int(500 * (params.T_SCALE / metadata['t scale']) / (max_HR / 60))
	# n_leads = leads.shape[0]
	n_leads = len(leads)
	r_peaks = [[] for i in range(n_leads)]
	for i in range(n_leads):
		strip = leads[i]
		# strip = highpass(strip, cutoff=hp_cutoff)  # high pass seems to make spikes worse
		strip = triangle_conv(strip, width=width)
		strip = list(lowpass(strip, cutoff=lp_cutoff))
		j = strip.index(max(strip))
		mean_peak, max_peak = strip[j], strip[j]
		while strip[j] > threshold * mean_peak:
			r_peaks[i].append(j)
			mean_peak = (mean_peak * (len(r_peaks[i]) - 1) + strip[j]) / len(r_peaks[i])
			begin, end = max(0, j - min_gap), min(j + min_gap, len(strip))
			strip[begin:end] = [0 for k in range(end - begin)]
			j = strip.index(max(strip))
		r_peaks[i].sort()
	return r_peaks


def sync_isolate(leads, metadata, width=None, manual_adjust=False, adjustments=None, verbose=False):
	"""synchronise R waves in each triad of ECG leads into bins """
	bin_size = 50  # width within which to consider Rs synchronised
	splt = 0.376  # proportion of beat before R wave (proportion of beat after R wave is 1 - splt)
	selected_Rs, output_leads, failed_sync = [], [], []
	r_waves = metadata['r waves']
	for i in range(4):	# for each column of 3 leads
		bins = []
		for j in range(3):	# for each lead in that column
			n = i * 3 + j
			head = metadata['lead heads'][n]

			# allocate all R waves to a bin
			for k in range(len(r_waves[n])):	# for each R wave in that lead
				r = r_waves[n][k] + head
				beat_start, beat_end = r - int(splt * width), r + int((1 - splt) * width)
				offset = max(0, head - beat_start) + max(0, beat_end - head - len(
					leads[n]))  # measure of displacement if edge beat used
				in_bin = [bins[b][0] - bin_size < r < bins[b][0] + bin_size for b in range(len(bins))]
				if True in in_bin:
					in_bin = in_bin.index(True)
					bins[in_bin].append(
						[n, k, r, offset])  # each item in a bin contains [lead, r index, r value, offset]
				else:
					bins.append([r])
					bins[-1].append([n, k, r, offset])  # create new bin if R doesn't fit in any other bins

		""" choose the bin that minimises the total offset of all 3 leads"""
		min_offset, best_bin = 10000, None
		for b in range(len(bins)):
			bins[b][1:].sort(key=lambda l: l[0])  # ensure leads are in order
			if len(bins[b]) == 4:
				total_offset = sum([bins[b][k][3] for k in range(1, 4)])
				if total_offset < min_offset:  # minimise value of offset
					best_bin = b
					min_offset = total_offset
		""" else choose the bin with 2 leads that minimises the offset
		if best_bin is None:
			min_offset = 10000
			for b in range(len(bins)):
				if len(bins[b]) == 3:
					total_offset = sum([bins[b][k][3] for k in range(1, 3)])
					if total_offset < min_offset:  # minimise value of offset
						best_bin = b
						min_offset = total_offset"""

		""" select appropriate Rs and return individual beat """
		for j in range(3):			# for each lead in this column
			if best_bin is None:	# pick least offset in each lead if no pot has 2 or 3 leads
				if j == 0: failed_sync.append(i)
				min_offset = 10000
				for b in bins:
					for x in b[1:]:
						n = x[0]
						if n == (i * 3 + j) and x[3] < min_offset:
							min_offset = x[3]
							r = r_waves[n][x[1]]
			else:					# pick selected R wave for this lead if best bin identified
				n = i * 3 + j
				r = r_waves[n][bins[best_bin][j + 1][1]]  # TODO problem if only 2 Rs in pot
			selected_Rs.append(r)

	metadata['selected Rs'] = selected_Rs

	# apply imported adjustments
	if adjustments != {}:
		if metadata['filename'] in adjustments.keys():
			metadata['selected Rs'] = adjustments[metadata['filename']]
			if verbose: print('Imported adjustments for ', metadata['filename'])

	# allow manual adjustments then save
	if manual_adjust:
		manual(leads, metadata, adjustments)

	# get isolated beat
	for n in range(12):
		r = metadata['selected Rs'][n]
		start, end = r - int(splt * width), r - int(splt * width) + width
		if start < 0:
			start, end = 0, width
		elif end > len(leads[n]):
			start, end = len(leads[n]) - width, len(leads[n])
		output_leads.append(leads[n][start:end])
	# print([len(output_leads[i]) for i in range(12)])  # for debugging only
	# are complexes centred at the same timestamp or around individual R waves? time stamp would be better aligned TODO
	# NB blue arrows are NOT aligned

	# check number and lengths of leads
	if len(output_leads) != 12: raise ConvertError('only {} leads found'.format(len(output_leads)))
	for i in range(len(output_leads)):
		if len(output_leads[i]) != params.lead_length['hcmr']:
			raise ConvertError('incorrect lead length {} in lead {}'.format(len(output_leads[i]), i))
	if failed_sync and verbose: print("WARNING: sync unsuccessful in columns {}. ".format(failed_sync), end='')

	output_leads = np.array(output_leads) - np.array(output_leads)[:, [0]]  # set start voltage to zero in each lead
	return output_leads, metadata


def manual(leads, metadata, adjustments):
	working = True
	adjusted = False
	while working:
		ECG_plots.plot_ecg(leads, metadata, save_name='', display=True, truncate=False)
		ipt = input('\nLead number to edit [0-11]? (blank to finish): ')
		if ipt == '':
			working = False
		else:
			n = int(ipt)
			ipt = input(f'Use different R wave from lead {n} [d] or copy from another lead [c]?: ')
			if ipt == 'd':
				if metadata['selected Rs'][n] in metadata['r waves'][n]:
					current_r = metadata['r waves'][n].index(metadata['selected Rs'][n])
				else:
					current_r = '[adjusted value]'
				val = input(f'Number of R wave to select in lead {n}? (current selection '
							f'is {current_r} of {str(list(range(len(metadata["r waves"][n]))))}): ')
				val = int(val)
				metadata['selected Rs'][n] = metadata['r waves'][n][val]
				adjusted = True
			if ipt == 'c':
				ld = int(input('Lead to copy R position from? [0-11]: '))
				metadata['selected Rs'][n] = metadata['selected Rs'][ld]
				adjusted = True
			if ipt == 'reset':
				metadata['r waves'] = find_r_waves((leads, metadata), t_scale=metadata['t scale'])
	if adjusted:
		adjustments[metadata['filename']] = metadata['selected Rs']
		with open(params.folder['hcmr'] + 'Manual adjustments.npy', 'wb') as f:
			pickle.dump(adjustments, f)
		print('Manual adjustment finished. Adjustments saved to file.')
	else:
		print('Manual adjustment finished. No adjustments saved.')


def ukbb_iso_cols(ecg, md, iso_length, adjustments={}, manual_adjust=False, verbose=False):
	md['r waves'] = find_r_waves((ecg, md), md['t scale'])
	output_leads, md = sync_isolate(ecg, md, iso_length,
									manual_adjust=manual_adjust, adjustments=adjustments, verbose=verbose)
	return output_leads, md


def long_to_paper(ecg, md):
	ecg = np.array([l[:5000] for l in ecg])
	col_ecg = np.zeros((12, 1250))
	for n in range(4):
		col_ecg[n * 3:n * 3 + 3] = ecg[n * 3:n * 3 + 3, n * 1250:(n + 1) * 1250]
	md['lead heads'] = [0] * 3 + [1250] * 3 + [2500] * 3 + [3750] * 3
	return col_ecg, md


def ukbb_iso(ecg, md, length, show_ims=False):
	""" takes UKBB ECGs and corresponding metadata as np arrays, and length as integer
	Extracts synchronised single PQRST complex from each lead with length of length
	ecg: np array with shape (n_leads, lead_length) with lead_length > length
	md: metadata dict containing list
	returns array of extracted lead data with shape (n_leads, length) with corresponding metadata """

	splt = 0.376  # proportion of beat before R wave (proportion of beat after R wave is 1 - splt)
	bin_width = 50
	# r_waves = find_r_waves((ecg, md))

	"""
	test_extracts = []
	selected_lead = -1
	while len(r_waves[selected_lead]) < 4:
		selected_lead-=1
	selected_r = r_waves[selected_lead][1]
	"""

	class Bin():
		def __init__(self, pos, width):
			self.pos = pos
			self.width = width
			self.r_waves = []

	def add_to_bins(bins, pos, n):
		""" adds an R wave with position pos to a list of bins """
		binned = False
		for b in bins:
			if b.pos - b.width < pos < b.pos + b.width:
				b.r_waves.append(n)
				binned = True
		if not binned:  # AND NOT NEAR EDGE
			new_bin = Bin(pos, bin_width)
			new_bin.r_waves.append(n)
			bins.append(new_bin)
		return bins

	bins = []
	for n in range(len(ecg)):
		r_list = md['r waves'][n]
		for r in r_list:
			add_to_bins(bins, r, n)
	bins.sort(key=lambda b: len(b.r_waves))  # sort by bin with most R waves

	# extract if not too close to start or end of ecg
	i = -1
	while (bins[i].pos > (ecg.shape[1] - int((1-splt)*length))) or (bins[i].pos < int(length*splt)):
		i -= 1
	selected_r = bins[i].pos
	md['selected r'] = selected_r
	output_leads = ecg[:, int(selected_r-splt*length): selected_r+int((1-splt)*length)]
	output_leads = output_leads - output_leads[:, [0]]	# set start voltage to zero in each lead

	# preview
	if show_ims:
		plt.plot(output_leads[0])
		plt.show()

	return output_leads, md


def trim(ecg, l, extend=True):
	""" trims an ecg (np array with shape (n_leads, lead_length) so that all leads have length l
	Extracts from the centre of the ECG.
	If extend is True and the ecg is shorter than l, adds static signal at the start and end to fill out to length l """
	if ecg.shape[1] < l:
		if extend:
			start_l = int((l - ecg.shape[1])/2)
			end_l = l - start_l - ecg.shape[1]
			return np.pad(ecg, ((0,0), (start_l, end_l)), 'edge')
		else:
			return ecg
	else:
		start = int((ecg.shape[1] - l)/2)
		finish = int(start + l)
		return ecg[:, start:finish]


def all_Rs_one_lead(leads, metadata, lead_n, width):
	""" isolates all complete beats from one lead, e.g. the rhythm strip """
	splt = 0.376
	beats = []
	r_waves = metadata['r waves'][lead_n]
	lead = leads[lead_n]
	for r in r_waves:
		start, end = r - int(splt * width), r - int(splt * width) + width
		if start > 0 and end < len(lead):
			beats.append(lead[start:end])
	return beats


def truncate_all_leads(leads, metadata, length):
	""" returns a fixed uncentred length from the start of each lead """
	leads_trunc = leads[:]
	for i in range(12):
		l = metadata['lead lengths'][i]
		if l < length:
			raise ConvertError("Truncation failed - lead {} too short (length = {})".format(i, l))
		else:
			leads_trunc[i] = leads_trunc[i][:length]
			metadata['lead lengths'][i] = length
	return leads_trunc, metadata


def align_Rs(lead_data, md):
	""" align multiple ECGs in lead_data such that R waves coincide (returns aligned copies of each ECG) """
	r = []
	for i in range(len(lead_data)):
		r.append(find_r_waves(lead_data[i]))  # list of all R peaks in each lead in each ECG [shape (m, 12, n_R_waves) ]

# How will this work with different heart rates???? Need to rescale by rate?

# switch axis to align by lead number

# find first complete complex in each lead
