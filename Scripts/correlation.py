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
#  This script correlates the specified dataset
#  Cross- & Autocorrelations are provided
#  Image is also saved in svg-format
########################################################
# Data input
data = tttrlib.TTTR('C:/Users/kah73xs/PycharmProjects/scripts/A488_1.ptu', 'PTU')  # file to be read
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

header = data.header
macro_time_calibration_ns = data.header.macro_time_resolution  # unit nanoseconds
macro_time_calibration_ms = macro_time_calibration_ns / 1e6  # macro time calibration in milliseconds
macro_times = data.macro_times
duration = float(header.tag("TTResult_StopAfter")["value"])  # unit millisecond

########################################################
#  Indices of data to correlate
########################################################

green_s_indices = data.get_selection_by_channel([correlation_channel1])
green_p_indices = data.get_selection_by_channel([correlation_channel2])

nr_of_green_s_photons = len(green_s_indices)
nr_of_green_p_photons = len(green_p_indices)

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

# Autocorrelation channel 1
autocorr_curve_ch1 = tttrlib.Correlator(
    channels=([correlation_channel1], [correlation_channel1]),
    tttr=data,
    **settings
)

# Autocorrelation channel 2
autocorr_curve_ch2 = tttrlib.Correlator(
    channels=([correlation_channel2], [correlation_channel2]),
    tttr=data,
    **settings
)

########################################################
#  Save correlation curve
########################################################
# 1st column contains the time axis (identical for all correlations)
time_axis_sec = crosscorrelation_curve.x_axis
time_axis = time_axis_sec * 1000  # bring time axis to the common unit millisecond

# 2nd column contains the correlation amplitude
crosscorrelation = crosscorrelation_curve.correlation
autocorrelation_ch1 = autocorr_curve_ch1.correlation
autocorrelation_ch2 = autocorr_curve_ch2.correlation

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
            crosscorrelation,
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
p.semilogx(time_axis, crosscorrelation, label='CCF')
p.semilogx(time_axis, autocorrelation_ch1, label='ACF1')
p.semilogx(time_axis, autocorrelation_ch2, label='ACF2')

p.ylim(ymin=1)
p.xlabel('correlation time [ms]')
p.ylabel('correlation amplitude')
p.legend()
p.savefig(save_figure_as + ".svg", dpi=150)
#p.show()
