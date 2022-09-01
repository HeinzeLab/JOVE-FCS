# Tested with tttrlib 0.21.9
###################################
# Katherina Hemmen ~ Core Unit Fluorescence Imaging ~ RVZ
# katherina.hemmen@uni-wuerzburg.de
###################################

import tttrlib
import pylab as p
import numpy as np
import glob

########################################################
#  This script joins all ptu-files with a certain name
#  export as decay, one file per channel, or in jordi format
#  Image is also saved in svg-format
########################################################
#  Input parameter
search_term = "DNA10bpPIE*.ptu"  # * marks variable area of file name
channel_1 = 0  # usually green perpendicular channel (VH, s)
channel_2 = 2  # usually green parallel channel (VV, p)
channel_3 = 1  # usually red perpendicular channel (VH, s)
channel_4 = 3  # usually red parallel channel (VV, p)
binning_factor = 4  # 1 = no binning, reduces histogram resolution by rebinning of channels
save_filename_channel_1 = "Green decay_s.txt"
save_filename_channel_2 = "Green decay_p.txt"
save_filename_channel_3 = "Red decay_s.txt"
save_filename_channel_4 = "Red decay_p.txt"
save_figure = "Decay.svg"
jordi_format = True  # True if jordi format (p-s stacked vertically) should be saved additionally
save_filename_jordi_green = "jordi_green.txt"
save_filename_jordi_red = "jordi_red.txt"

########################################################
#  Read data & header
########################################################

# Caution! Data will be appended not in a sorted way
filename = glob.glob(search_term)  # search term: which files should be joined
first_curve = filename[0]
data = tttrlib.TTTR(first_curve, 'PTU')
header = data.header
macro_time_calibration = data.header.macro_time_resolution  # unit nanoseconds
micro_times = data.micro_times  # unit seconds
micro_time_resolution = data.header.micro_time_resolution

data_sets = [tttrlib.TTTR(fn, 'PTU') for fn in filename[0:]]
ch1_list = list()
ch2_list = list()
ch3_list = list()
ch4_list = list()

for ds in data_sets:
    ########################################################
    #  Data rebinning (native resolution often too high, 16-32 ps sufficient)
    ########################################################

    binning = binning_factor  # Binning factor
    # This is the max nr of bins the data should contain:
    expected_nr_of_bins = int(macro_time_calibration // micro_time_resolution)
    # After binning the nr of bins is reduced:
    binned_nr_of_bins = int(expected_nr_of_bins // binning)

    ########################################################
    #  Histogram creation
    ########################################################

    # Select the channels & get the respective microtimes
    green_s_indices = np.array(data.get_selection_by_channel([channel_1]), dtype=np.int64)
    green_p_indices = np.array(data.get_selection_by_channel([channel_2]), dtype=np.int64)
    red_s_indices = np.array(data.get_selection_by_channel([channel_3]), dtype=np.int64)
    red_p_indices = np.array(data.get_selection_by_channel([channel_4]), dtype=np.int64)

    green_s = micro_times[green_s_indices]
    green_p = micro_times[green_p_indices]
    red_s = micro_times[red_s_indices]
    red_p = micro_times[red_p_indices]

    # Build the histograms
    green_s_counts = np.bincount(green_s // binning, minlength=binned_nr_of_bins)
    green_p_counts = np.bincount(green_p // binning, minlength=binned_nr_of_bins)
    red_s_counts = np.bincount(red_s // binning, minlength=binned_nr_of_bins)
    red_p_counts = np.bincount(red_p // binning, minlength=binned_nr_of_bins)

    #  observed problem: data contains more bins than possible, rounding errors?
    #  cut down to expected length:
    green_s_counts_cut = green_s_counts[0:binned_nr_of_bins:]
    ch1_list.append(np.array(green_s_counts_cut))
    green_p_counts_cut = green_p_counts[0:binned_nr_of_bins:]
    ch2_list.append(np.array(green_p_counts_cut))
    red_s_counts_cut = red_s_counts[0:binned_nr_of_bins:]
    ch3_list.append(np.array(red_s_counts_cut))
    red_p_counts_cut = red_p_counts[0:binned_nr_of_bins:]
    ch4_list.append(np.array(red_p_counts_cut))

# Build the time axis
dt = micro_time_resolution * 1e9  # unit nanoseconds
x_axis = np.arange(green_s_counts_cut.shape[0]) * dt * binning_factor  # identical for data from same time window

# sum the decays from all datasets
decay_ch1 = np.array(ch1_list)
sum_decay_ch1 = decay_ch1.sum(axis=0)
decay_ch2 = np.array(ch2_list)
sum_decay_ch2 = decay_ch2.sum(axis=0)
decay_ch3 = np.array(ch3_list)
sum_decay_ch3 = decay_ch3.sum(axis=0)
decay_ch4 = np.array(ch4_list)
sum_decay_ch4 = decay_ch4.sum(axis=0)

########################################################
#  Saving & plotting
########################################################

output_filename = save_filename_channel_1
np.savetxt(
    output_filename,
    np.vstack([x_axis, sum_decay_ch1]).T
)

output_filename = save_filename_channel_2
np.savetxt(
    output_filename,
    np.vstack([x_axis, sum_decay_ch2]).T
)

output_filename = save_filename_channel_3
np.savetxt(
    output_filename,
    np.vstack([x_axis, sum_decay_ch3]).T
)

output_filename = save_filename_channel_4
np.savetxt(
    output_filename,
    np.vstack([x_axis, sum_decay_ch4]).T
)

p.semilogy(x_axis, sum_decay_ch1, label='Ch1')
p.semilogy(x_axis, sum_decay_ch2, label='Ch2')
p.semilogy(x_axis, sum_decay_ch3, label='Ch3')
p.semilogy(x_axis, sum_decay_ch4, label='Ch4')

p.xlabel('time [ns]')
p.ylabel('Counts')
p.legend()
p.savefig(save_figure, dpi=150)
p.show()

# Optional: jordi format for direct reading in FitMachine & ChiSurf
if jordi_format:
    jordi_counts_green = np.concatenate([sum_decay_ch2, sum_decay_ch1])
    jordi_counts_red = np.concatenate([sum_decay_ch4, sum_decay_ch3])

    output_filename = save_filename_jordi_green
    np.savetxt(
        output_filename,
        np.vstack([jordi_counts_green]).T
    )

    output_filename = save_filename_jordi_red
    np.savetxt(
        output_filename,
        np.vstack([jordi_counts_red]).T
    )
