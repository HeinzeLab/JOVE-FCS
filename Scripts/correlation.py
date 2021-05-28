#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import pylab as p
import tttrlib
import functions

########################################################
#  This script correlates the specified dataset
#  Cross- & Autocorrelations are provided
#  Image is also saved in svg-format
########################################################
# Data input
data = tttrlib.TTTR(r'\\132.187.2.213\rvz03\users\AGHeinze\PROJECTS\P019_JovE_FCCS\Simulation\eGFP\Cell_Block-123_middle of 2.ptu', 'PTU')  # file to be read
correlation_channel1 = [3]  # correlation channel 1, can be one or multiple, e.g. [0,2]
correlation_channel2 = [2]  # correlation channel 2, can be one or multiple, e.g. [0,2]
n_correlation_casc = 25  # number of correlation cascades, increase for increasing lag-times
save_crosscorrelation_as = "ch0_ch2_cross"  # filename for crosscorrelation curve
save_autocorrelation1_as = "ch0_auto"  # filename for autocorrelation curve, channel 1
save_autocorrelation2_as = "ch2_auto"  # filename for autocorrelation curve, channel 2
save_figure_as = "correlation"  # filename for saving of figure

########################################################
#  Read information from header
########################################################

header = data.get_header()
header_data = header.data
macro_time_calibration_ns = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration_ms = macro_time_calibration_ns / 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()
duration = float(header_data["TTResult_StopAfter"])  # unit millisecond

########################################################
#  Indices of data to correlate
########################################################
# the dtype to int64 otherwise numba jit has hiccups
green_s_indices = np.array(data.get_selection_by_channel([correlation_channel1]), dtype=np.int64)
green_p_indices = np.array(data.get_selection_by_channel([correlation_channel2]), dtype=np.int64)

nr_of_green_s_photons = len(green_s_indices)
nr_of_green_p_photons = len(green_p_indices)

########################################################
#  Correlate
########################################################
crosscorrelation_curve = functions.correlate(
    macro_times=macro_times,
    indices_ch1=green_s_indices,
    indices_ch2=green_p_indices,
    n_casc=n_correlation_casc
)

autocorr_curve_ch1 = functions.correlate(
    macro_times=macro_times,
    indices_ch1=green_s_indices,
    indices_ch2=green_s_indices,
    n_casc=n_correlation_casc
)

autocorr_curve_ch2 = functions.correlate(
    macro_times=macro_times,
    indices_ch1=green_p_indices,
    indices_ch2=green_p_indices,
    n_casc=n_correlation_casc
)

########################################################
#  Save correlation curve
########################################################
# calculate the correct time axis by multiplication of x-axis with macro_time
time_axis = crosscorrelation_curve[0] * macro_time_calibration_ms

# 2nd column contains the average correlation amplitude
crosscorrelation_curve = crosscorrelation_curve[1]
autocorrelation_ch1 = autocorr_curve_ch1[1]
autocorrelation_ch2 = autocorr_curve_ch2[1]

# fill 3rd column with 0's for compatibility with ChiSurf & Kristine
# 1st and 2nd entry of 3rd column are measurement duration & average countrate
suren_column_ccf = np.zeros_like(time_axis)
suren_column_acf1 = np.zeros_like(time_axis)
suren_column_acf2 = np.zeros_like(time_axis)

duration_sec = duration / 1000
cr_green_ch1 = nr_of_green_s_photons / duration_sec / 1000  # kHz
cr_green_ch2 = nr_of_green_p_photons / duration_sec / 1000  # kHz
avg_cr = (cr_green_ch1 + cr_green_ch2) / 2

suren_column_ccf[0] = duration_sec
suren_column_ccf[1] = avg_cr

suren_column_acf1[0] = duration_sec
suren_column_acf1[1] = cr_green_ch1

suren_column_acf2[0] = duration_sec
suren_column_acf2[1] = cr_green_ch2

filename_cc = save_crosscorrelation_as + '.cor'  # change file name!
np.savetxt(
    filename_cc,
    np.vstack(
        [
            time_axis,
            crosscorrelation_curve,
            suren_column_ccf
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
            autocorrelation_ch1,
            suren_column_acf1
        ]
    ).T,
    delimiter='\t'
)

filename_acf2 = save_autocorrelation1_as + '.cor'  # change file name!
np.savetxt(
    filename_acf2,
    np.vstack(
        [
            time_axis,
            autocorrelation_ch2,
            suren_column_acf2
        ]
    ).T,
    delimiter='\t'
)

########################################################
#  Plotting
########################################################
p.semilogx(time_axis, crosscorrelation_curve, label='CCF')
p.semilogx(time_axis, autocorrelation_ch1, label='ACF1')
p.semilogx(time_axis, autocorrelation_ch2, label='ACF2')

p.xlabel('correlation time [ms]')
p.ylabel('correlation amplitude')
p.legend()
p.savefig(save_figure_as + ".svg", dpi=150)
p.show()
