# Tested with tttrlib 0.21.9
###################################
# Katherina Hemmen ~ Core Unit Fluorescence Imaging ~ RVZ
# katherina.hemmen@uni-wuerzburg.de
###################################

#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import pylab as p
import tttrlib

########################################################
#  Data input & correlation selection
########################################################

data = tttrlib.TTTR("HEK 293T b2Ar.ptu", 'PTU')  # file to be processed, filetype
correlation_channel1 = [0]  # correlation channel 1, can be one or multiple, e.g. [0,2]
correlation_channel2 = [2]  # correlation channel 2, can be one or multiple, e.g. [0,2]
n_casc = 25  # n_bins and n_casc defines the settings of the multi-tau
n_bins = 9  # correlation algorithm
n_chunks = 3 # number of pieces the data is to be split into
save_crosscorrelation_as = "ch0_ch2_cross"  # filename for crosscorrelation curve
save_autocorrelation1_as = "ch0_auto"  # filename for autocorrelation curve, channel 1
save_autocorrelation2_as = "ch2_auto"  # filename for autocorrelation curve, channel 2
save_figure_as = "correlation"  # filename for saving of figure

########################################################
#  Read information from header
########################################################

header = data.header
macro_time_calibration = data.header.macro_time_resolution  # unit secondss
macro_times = data.macro_times
micro_times = data.micro_times
micro_time_resolution = data.header.micro_time_resolution
duration = float(header.tag("TTResult_StopAfter")["value"])  # unit millisecond
duration_sec = duration / 1000
window_length = duration_sec / n_chunks  # in seconds
print("macro_time_calibration [sec]:", macro_time_calibration)
print("Duration [sec]:", duration_sec)
print("Time window lenght [sec]:", window_length)

########################################################
#  Select the indices of the events to be correlated
########################################################

green_s_indices = data.get_selection_by_channel(correlation_channel1)
green_p_indices = data.get_selection_by_channel(correlation_channel2)

nr_of_green_s_photons = len(green_s_indices)
nr_of_green_p_photons = len(green_p_indices)

# Get the start-stop indices of the data slices
time_windows = data.get_ranges_by_time_window(
    window_length, macro_time_calibration=macro_time_calibration)
start_stop = time_windows.reshape((len(time_windows)//2, 2))
print(start_stop)

########################################################
#  Correlate the pieces
########################################################
# Correlator settings, define the identical settings once
settings = {
    "method": "default",
    "n_bins": n_bins,  # n_bins and n_casc defines the settings of the multi-tau
    "n_casc": n_casc,  # correlation algorithm
    "make_fine": False  # Do not use the microtime information
}

# Crosscorrelation
crosscorrelation = tttrlib.Correlator(**settings)
crosscorrelations = list()
for start, stop in start_stop:
    indices = np.arange(start, stop, dtype=np.int64)
    tttr_slice = data[indices]
    tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel(correlation_channel1)]
    tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel(correlation_channel2)]
    crosscorrelation.set_tttr(
        tttr_1=tttr_ch1,
        tttr_2=tttr_ch2
    )
    crosscorrelations.append(
        (crosscorrelation.x_axis, crosscorrelation.correlation)
    )
    
crosscorrelations = np.array(crosscorrelations)
   
# Autocorrelations
autocorr_ch1 = tttrlib.Correlator(**settings)
autocorrs_ch1 = list()
for start, stop in start_stop:
    indices = np.arange(start, stop, dtype=np.int64)
    tttr_slice = data[indices]
    tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel(correlation_channel1)]
    tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel(correlation_channel1)]
    autocorr_ch1.set_tttr(
        tttr_1=tttr_ch1,
        tttr_2=tttr_ch2
    )
    autocorrs_ch1.append(
        (autocorr_ch1.x_axis, autocorr_ch1.correlation)
    )

autocorrs_ch1 = np.array(autocorrs_ch1)
 
autocorr_ch2 = tttrlib.Correlator(**settings)
autocorrs_ch2 = list()
for start, stop in start_stop:
    indices = np.arange(start, stop, dtype=np.int64)
    tttr_slice = data[indices]
    tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel(correlation_channel2)]
    tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel(correlation_channel2)]
    autocorr_ch2.set_tttr(
        tttr_1=tttr_ch1,
        tttr_2=tttr_ch2
    )
    autocorrs_ch2.append(
        (autocorr_ch2.x_axis, autocorr_ch2.correlation)
    )

autocorrs_ch2 = np.array(autocorrs_ch2)

########################################################
#  Get mean and standard deviation
########################################################

correlation_amplitudes = crosscorrelations[:, 1, :]
average_correlation_amplitude = correlation_amplitudes.mean(axis=0)
std_correlation_amplitude = correlation_amplitudes.std(axis=0)

curves_ch1 = autocorrs_ch1[:, 1, :]
avg_curve_ch1 = np.mean(curves_ch1, axis=0)
std_curve_ch1 = np.std(curves_ch1, axis=0)

curves_ch2 = autocorrs_ch2[:, 1, :]
avg_curve_ch2 = np.mean(curves_ch2, axis=0)
std_curve_ch2 = np.std(curves_ch2, axis=0)

########################################################
#  Save correlation curves
########################################################
# calculate the correct time axis by multiplication of x-axis with macro_time
time_axis = crosscorrelations[0, 0] * 1000

# fill 3rd column with 0's for compatibility with ChiSurf & Kristine
# 1st and 2nd entry of 3rd column are measurement duration & average countrate
suren_column_ccf = np.zeros_like(time_axis)
suren_column_acf1 = np.zeros_like(time_axis)
suren_column_acf2 = np.zeros_like(time_axis)

nr_of_green_s_photons = len(green_s_indices)
nr_of_green_p_photons = len(green_p_indices)

cr_green_ch1 = nr_of_green_s_photons / duration_sec / 1000  # kHz
cr_green_ch2 = nr_of_green_p_photons / duration_sec / 1000  # kHz
avg_cr = (cr_green_ch1 + cr_green_ch2) / 2

suren_column_ccf[0] = duration_sec
suren_column_ccf[1] = avg_cr

suren_column_acf1[0] = duration_sec
suren_column_acf1[1] = cr_green_ch1

suren_column_acf2[0] = duration_sec
suren_column_acf2[1] = cr_green_ch2

# 4th column contains standard deviation from the averaged curve calculated above
std_avg_correlation_amplitude = std_correlation_amplitude/np.sqrt(n_chunks)
std_avg_correlation_amplitude_ch1 = std_curve_ch1/np.sqrt(n_chunks)
std_avg_correlation_amplitude_ch2 = std_curve_ch2/np.sqrt(n_chunks)

filename_cc = save_crosscorrelation_as + '.cor'  # change file name!
np.savetxt(
    filename_cc,
    np.vstack(
        [
            time_axis,
            average_correlation_amplitude,
            suren_column_ccf,
            std_avg_correlation_amplitude
         ]
    ).T,
    delimiter='\t'
)

filename_acf1 = save_autocorrelation1_as + '.cor'  # change file name!
np.savetxt(
    filename_acf1,
    np.vstack(
        [
            time_axis,
            avg_curve_ch1,
            suren_column_acf1,
            std_avg_correlation_amplitude_ch1
         ]
    ).T,
    delimiter='\t'
)

filename_acf2 = save_autocorrelation2_as + '.cor'  # change file name!
np.savetxt(
    filename_acf2,
    np.vstack(
        [
            time_axis,
            avg_curve_ch2,
            suren_column_acf2,
            std_avg_correlation_amplitude_ch2
         ]
    ).T,
    delimiter='\t'
)

########################################################
#  Plot & save figure
########################################################
p.semilogx(time_axis, average_correlation_amplitude, label='CCF')
p.semilogx(time_axis, avg_curve_ch1, label='ACF1')
p.semilogx(time_axis, avg_curve_ch2, label='ACF2')

p.ylim(ymin=1)
p.xlabel('correlation time [ms]')
p.ylabel('correlation amplitude')
p.legend()
p.savefig(save_figure_as + '.svg', dpi=150)
p.show()