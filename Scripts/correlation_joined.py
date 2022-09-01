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
import glob

########################################################
#  This script joins all ptu-files with a certain name
#  Cross- & Autocorrelations are provided
#  Image is also saved in svg-format
########################################################
#  Input parameter
search_term = "C:/Users/kah73xs/PycharmProjects/scripts/A488_*.ptu"  # * marks variable area of file name
correlation_channel1 = 0  # correlation channel 1, can be one or multiple, e.g. [0,2]
correlation_channel2 = 2  # correlation channel 2, can be one or multiple, e.g. [0,2]
n_casc = 25  # n_bins and n_casc defines the settings of the multi-tau
n_bins = 9  # correlation algorithm
save_crosscorrelation_as = "ch0_ch2_cross"  # filename for crosscorrelation curve
save_autocorrelation1_as = "ch0_auto"  # filename for autocorrelation curve, channel 1
save_autocorrelation2_as = "ch2_auto"  # filename for autocorrelation curve, channel 2
save_figure_as = "correlation"  # filename for saving of figure

########################################################
#  Read information from header
########################################################

# Caution! Data will be appended not in a sorted way
filename = glob.glob(search_term)  # search term: which files should be appended
nr_of_curves = len(filename)

data_sets = [tttrlib.TTTR(fn, 'PTU') for fn in filename[0:]]
cross_list = list()
acf1_list = list()
acf2_list = list()

nr_of_green_s_photons = 0
nr_of_green_p_photons = 0
total_duration = 0

for ds in data_sets:
    ########################################################
    #  Indices of data to correlate
    ########################################################
    data = tttrlib.TTTR(ds)
    header = data.header
    micro_times = data.micro_times
    micro_time_resolution = data.header.micro_time_resolution
    macro_time_calibration_ns = data.header.macro_time_resolution  # unit nanoseconds
    macro_time_calibration_ms = macro_time_calibration_ns / 1e6  # macro time calibration in milliseconds
    macro_times = data.macro_times

    green_s_indices = data.get_selection_by_channel([0])
    green_p_indices = data.get_selection_by_channel([2])
    nr_of_green_s_photons += len(green_s_indices)
    nr_of_green_p_photons += len(green_p_indices)
    duration = float(header.tag("TTResult_StopAfter")["value"])   # unit millisecond
    total_duration += duration

    ########################################################
    #  Correlate
    ########################################################
    
    # Correlator settings, define the identical settings once
    settings = {
        "method": "default",
        "n_bins": n_bins,  # n_bins and n_casc defines the settings of the multi-tau
        "n_casc": n_casc,  # correlation algorithm
        "make_fine": False  # Do not use the microtime information
    }

    # Crosscorrelation
    crosscorrelation_curve = tttrlib.Correlator(
        channels=([correlation_channel1], [correlation_channel2]),
        tttr=data,
        **settings
    )
    
    crosscorrelation = crosscorrelation_curve.correlation
    cross_list.append(np.array(crosscorrelation))
    
    # Autocorrelation channel 1
    autocorr_curve_ch1 = tttrlib.Correlator(
        channels=([correlation_channel1], [correlation_channel1]),
        tttr=data,
        **settings
    )
    
    autocorrelation_ch1 = autocorr_curve_ch1.correlation
    acf1_list.append(np.array(autocorrelation_ch1))
    
    # Autocorrelation channel 2
    autocorr_curve_ch2 = tttrlib.Correlator(
        channels=([correlation_channel2], [correlation_channel2]),
        tttr=data,
        **settings
    )
    
    autocorrelation_ch2 = autocorr_curve_ch2.correlation
    acf2_list.append(np.array(autocorrelation_ch2))
    

########################################################
#  Get mean and standard deviation
########################################################

correlation_amplitude = np.array(cross_list)
average_correlation_amplitude = correlation_amplitude.mean(axis=0)
std_correlation_amplitude = correlation_amplitude.std(axis=0)

autocorr1_amplitude = np.array(acf1_list)
avg_curve_ch1 = np.mean(autocorr1_amplitude, axis=0)
std_curve_ch1 = np.std(autocorr1_amplitude, axis=0)

autocorr2_amplitude = np.array(acf2_list)
avg_curve_ch2 = np.mean(autocorr2_amplitude, axis=0)
std_curve_ch2 = np.std(autocorr2_amplitude, axis=0)

########################################################
#  Save correlation curves
########################################################
# calculate the correct time axis by multiplication of x-axis with macro_time
time_axis_sec = crosscorrelation_curve.x_axis
time_axis = time_axis_sec * 1000  # bring time axis to the common unit millisecond

# 2nd column contains the average correlation amplitude calculated above
avg_correlation_amplitude = average_correlation_amplitude
avg_correlation_amplitude_ch1 = avg_curve_ch1
avg_correlation_amplitude_ch2 = avg_curve_ch2

# fill 3rd column with 0's for compatibility with ChiSurf & Kristine
# 1st and 2nd entry of 3rd column are measurement duration & average countrate
suren_column_ccf = np.zeros_like(time_axis)
suren_column_acf1 = np.zeros_like(time_axis)
suren_column_acf2 = np.zeros_like(time_axis)

duration_sec = total_duration / 1000
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
std_avg_correlation_amplitude = std_correlation_amplitude/np.sqrt(nr_of_curves)
std_avg_correlation_amplitude_ch1 = std_curve_ch1/np.sqrt(nr_of_curves)
std_avg_correlation_amplitude_ch2 = std_curve_ch2/np.sqrt(nr_of_curves)

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
            avg_correlation_amplitude_ch1,
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
            avg_correlation_amplitude_ch2,
            suren_column_acf2,
            std_avg_correlation_amplitude_ch2
         ]
    ).T,
    delimiter='\t'
)

########################################################
#  Plot & save figure
########################################################
p.semilogx(time_axis, avg_correlation_amplitude, label='gs-gp')
p.semilogx(time_axis, avg_curve_ch1, label='gs-gs')
p.semilogx(time_axis, avg_curve_ch2, label='gp-gp')

p.ylim(ymin=1)
p.xlabel('correlation time [ms]')
p.ylabel('correlation amplitude')
p.legend()
p.savefig(save_figure_as + '.svg', dpi=150)
p.show()
