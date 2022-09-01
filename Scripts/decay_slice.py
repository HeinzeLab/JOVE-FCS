# Tested with tttrlib 0.21.9
###################################
# Katherina Hemmen ~ Core Unit Fluorescence Imaging ~ RVZ
# katherina.hemmen@uni-wuerzburg.de
###################################

import tttrlib
import pylab as p
import numpy as np

########################################################
#  Data input & reading
########################################################

data = tttrlib.TTTR('A488_1.ptu', 'PTU')
channel_1 = 0  # usually perpendicular channel (VH, s)
channel_2 = 2  # usually parallel channel (VV, p)
binning_factor = 4  # 1 = no binning, reduces histogram resolution by rebinning of channels
save_filename_channel_1 = "Decay_s"
save_filename_channel_2 = "Decay_p"
save_figure = "Decay"
jordi_format = True  # True if jordi format (p-s stacked vertically) should be saved additionally
save_filename_jordi = "jordi"
n_chunks = 3 # number of pieces the data is to be split into

########################################################
#  Read data & header
########################################################

header = data.header
macro_time_calibration = data.header.macro_time_resolution  # unit seconds
macro_times = data.macro_times
micro_times = data.micro_times  # unit seconds
micro_time_resolution = data.header.micro_time_resolution

duration = float(header.tag("TTResult_StopAfter")["value"])  # unit millisecond
duration_sec = duration / 1000
window_length = duration_sec / n_chunks  # in seconds

print("macro_time_calibration:", macro_time_calibration)
print("micro_time_resolution:", micro_time_resolution)
print("Duration [sec]:", duration_sec)
print("Time window lenght [sec]:", window_length)

########################################################
#  Data rebinning (native resolution often too high, 16-32 ps sufficient)
########################################################

binning = binning_factor  # Binning factor
# This is the max nr of bins the data should contain:
expected_nr_of_bins = int(macro_time_calibration//micro_time_resolution)
# After binning the nr of bins is reduced:
binned_nr_of_bins = int(expected_nr_of_bins//binning)

########################################################
#  Selecting time windows
########################################################

# Get the start-stop indices of the data slices
time_windows = data.get_ranges_by_time_window(
    window_length, macro_time_calibration=macro_time_calibration)
start_stop = time_windows.reshape((len(time_windows)//2, 2))
print(start_stop)

########################################################
#  Histogram creation
########################################################
i=0
for start, stop in start_stop:
    indices = np.arange(start, stop, dtype=np.int64)
    tttr_slice = data[indices]
    green_s = micro_times[tttr_slice.get_selection_by_channel([channel_1])]
    green_p = micro_times[tttr_slice.get_selection_by_channel([channel_2])]
    # Build the histograms
    green_s_counts = np.bincount(green_s // binning, minlength=binned_nr_of_bins)
    green_p_counts = np.bincount(green_p // binning, minlength=binned_nr_of_bins)
    #  observed problem: data contains more bins than possible, rounding errors?
    #  cut down to expected length:
    green_s_counts_cut = green_s_counts[0:binned_nr_of_bins:]
    green_p_counts_cut = green_p_counts[0:binned_nr_of_bins:]
    # Build the time axis
    dt = micro_time_resolution * 1e9  # unit nanoseconds
    x_axis = np.arange(green_s_counts_cut.shape[0]) * dt * binning  # identical for data from same time window

    ########################################################
    #  Saving & plotting
    ########################################################

    output_filename_s = save_filename_channel_1 + '_' + str(i) + '.txt'
    np.savetxt(
         output_filename_s,
         np.vstack([x_axis, green_s_counts_cut]).T
     )
    output_filename_p = save_filename_channel_1 + '_' + str(i) + '.txt'
    np.savetxt(
         output_filename_p,
         np.vstack([x_axis, green_p_counts_cut]).T
     )

    p.semilogy(x_axis, green_s_counts_cut, label='gs')
    p.semilogy(x_axis, green_p_counts_cut, label='gp')

    p.xlabel('time [ns]')
    p.ylabel('Counts')
    p.legend()
    p.savefig(save_figure + '_' + str(i) + '.svg', dpi=150)
    p.show()

    # Optional: jordi format for direct reading in FitMachine & ChiSurf(2015-2017)
    if jordi_format:
        jordi_counts_green = np.concatenate([green_p_counts_cut, green_s_counts_cut])

        output_filename = save_filename_jordi + '_' + str(i) + '.txt'
        np.savetxt(
            output_filename,
            np.vstack([jordi_counts_green]).T
        )
        
    i+=1
