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
#  This script correlates the specified data set
#  Cross- & Autocorrelations are provided
#  Microtimes are used to define "prompt" & "delay" of PIE
#  Image is also saved in svg-format
########################################################
#  Input parameter
data = tttrlib.TTTR('C:/Users/kah73xs/PycharmProjects/scripts/DNA10bpPIE_1.ptu', 'PTU')   # file to be read
green_channel1 = 0  # green correlation channel 1, can be one or multiple, e.g. [0,2]
green_channel2 = 2  # green correlation channel 2, can be one or multiple, e.g. [0,2]
red_channel1 = 1  # red correlation channel 1, can be one or multiple, e.g. [0,2]
red_channel2 = 3  # red correlation channel 2, can be one or multiple, e.g. [0,2]
n_casc = 25  # n_bins and n_casc defines the settings of the multi-tau
n_bins = 9  # correlation algorithm
save_PIEcrosscorrelation_as = "PIE_CCF"  # filename for CCF of green-prompt and red-delay
save_FRETcrosscorrelation_as = "FRET_CCF"  # filename for CCF of green-prompt and red-prompt
save_ACF_green_prompt_as = "ACF_green_prompt"  # filename for ACF green-prompt
save_ACF_red_prompt_as = "ACF_red_prompt"  # filename for ACF red-prompt
save_ACF_red_delay_as = "ACF_red_delay"  # filename for ACF red-delay
save_figure_as = "correlation"  # filename for saving of figure

########################################################
#  Read information from header
########################################################

header = data.header
macro_time_calibration = data.header.macro_time_resolution  # unit seconds
micro_time_resolution = data.header.micro_time_resolution
macro_times = data.macro_times
micro_times = data.micro_times
number_of_bins = macro_time_calibration/micro_time_resolution
PIE_windows_bins = int(number_of_bins/2)

print("macro_time_calibration_sec:", macro_time_calibration)
print("micro_time_resolution_ns:", micro_time_resolution)
print("number_of_bins:", number_of_bins)
print("PIE_windows_bins:", PIE_windows_bins)

########################################################
#  Indices of data to correlate
########################################################

all_green_indices = data.get_selection_by_channel([green_channel1, green_channel2])
all_red_indices = data.get_selection_by_channel([red_channel1, red_channel2])
green_indices1 = data.get_selection_by_channel([green_channel1])
green_indices2 = data.get_selection_by_channel([green_channel2])
red_indices1 = data.get_selection_by_channel([red_channel1])
red_indices2 = data.get_selection_by_channel([red_channel2])

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

# Select macrotimes for crosscorrelations
t_green = macro_times[all_green_indices]
t_red = macro_times[all_red_indices]

# Select microtimes for crosscorrelations
mt_green = micro_times[all_green_indices]
mt_red = micro_times[all_red_indices]

# Define and apply weights
w_gp = np.ones_like(t_green, dtype=float)
w_gp[np.where(mt_green > PIE_windows_bins)] *= 0.0
w_rp = np.ones_like(t_red, dtype=float)
w_rp[np.where(mt_red > PIE_windows_bins)] *= 0.0
w_rd = np.ones_like(t_red, dtype=float)
w_rd[np.where(mt_red < PIE_windows_bins)] *= 0.0

# PIE crosscorrelation (green prompt - red delay)
PIEcorrelation_curve = tttrlib.Correlator(**settings)
PIEcorrelation_curve.set_events(t_green, w_gp, t_red, w_rd)

PIE_time_axis = PIEcorrelation_curve.x_axis
PIE_amplitude = PIEcorrelation_curve.correlation

# FRET crosscorrelation
FRETcrosscorrelation_curve = tttrlib.Correlator(**settings)
FRETcrosscorrelation_curve.set_events(t_green, w_gp, t_red, w_rp)

FRET_time_axis = FRETcrosscorrelation_curve.x_axis
FRET_amplitude = FRETcrosscorrelation_curve.correlation

# Select macrotimes for autocorrelations
t_green1 = macro_times[green_indices1]
t_green2 = macro_times[green_indices2]
t_red1 = macro_times[red_indices1]
t_red2 = macro_times[red_indices2]

# Select microtimes for autocorrelation
mt_green1 = micro_times[green_indices1]
mt_green2 = micro_times[green_indices2]
mt_red1 = micro_times[red_indices1]
mt_red2 = micro_times[red_indices2]

# Define and apply weights
w_g1 = np.ones_like(t_green1, dtype=float)
w_g1[np.where(mt_green1 > PIE_windows_bins)] *= 0.0
w_g2 = np.ones_like(t_green2, dtype=float)
w_g2[np.where(mt_green2 > PIE_windows_bins)] *= 0.0

w_r1p = np.ones_like(t_red1, dtype=float)
w_r1p[np.where(mt_red1 > PIE_windows_bins)] *= 0.0
w_r2p = np.ones_like(t_red2, dtype=float)
w_r2p[np.where(mt_red2 > PIE_windows_bins)] *= 0.0

w_r1d = np.ones_like(t_red1, dtype=float)
w_r1d[np.where(mt_red1 < PIE_windows_bins)] *= 0.0
w_r2d = np.ones_like(t_red2, dtype=float)
w_r2d[np.where(mt_red2 < PIE_windows_bins)] *= 0.0

autocorr_prompt_g = tttrlib.Correlator(**settings)
autocorr_prompt_g.set_events(t_green1, w_g1, t_green2, w_g2)

autocorr_prompt_r = tttrlib.Correlator(**settings)
autocorr_prompt_r.set_events(t_red1, w_r1p, t_red2, w_r2p)

autocorr_delay_r = tttrlib.Correlator(**settings)
autocorr_delay_r.set_events(t_red1, w_r1d, t_red2, w_r2d)

ACF_prompt_g_time_axis = autocorr_prompt_g.x_axis
ACF_prompt_g_amplitude = autocorr_prompt_g.correlation

ACF_prompt_r_time_axis = autocorr_prompt_r.x_axis
ACF_prompt_r_amplitude = autocorr_prompt_r.correlation

ACF_delay_r_time_axis = autocorr_delay_r.x_axis
ACF_delay_r_amplitude = autocorr_delay_r.correlation

########################################################
#  Save correlation curve
########################################################
# bring the time axis to miliseconds
time_axis = PIE_time_axis * macro_time_calibration *1000

# fill 3rd column with 0's for compatibility with ChiSurf & Kristine
# 1st and 2nd entry of 3rd column are measurement duration & average countrate
suren_columnPIE = np.zeros_like(time_axis)
suren_columnFRET = np.zeros_like(time_axis)
suren_column_gp = np.zeros_like(time_axis)
suren_column_rp = np.zeros_like(time_axis)
suren_column_rd = np.zeros_like(time_axis)

duration = float(header.tag("TTResult_StopAfter")["value"])  # unit millisecond
duration_sec = duration / 1000

all_green_photons = micro_times[all_green_indices]
nr_of_green_photons = (np.array(np.where(all_green_photons <= PIE_windows_bins), dtype=np.int64)).size
all_red_photons = micro_times[all_red_indices]
nr_of_red_p_photons = (np.array(np.where(all_red_photons <= PIE_windows_bins), dtype=np.int64)).size
nr_of_red_d_photons = (np.array(np.where(all_red_photons > PIE_windows_bins), dtype=np.int64)).size

cr_green_p = nr_of_green_photons / 2 / duration_sec / 1000  # kHz, avg of two detectors
cr_red_p = nr_of_red_p_photons / 2 / duration_sec / 1000  # kHz, avg of two detectors
cr_red_d = nr_of_red_d_photons / 2 / duration_sec / 1000  # kHz, avg of two detectors
avg_cr_PIE = (cr_green_p + cr_red_d)  # avg of green and red
avg_cr_FRET = (cr_green_p + cr_red_p)  # avg of green and red

suren_columnPIE[0] = duration_sec
suren_columnPIE[1] = avg_cr_PIE

suren_columnFRET[0] = duration_sec
suren_columnFRET[1] = avg_cr_FRET

suren_column_gp[0] = duration_sec
suren_column_gp[1] = cr_green_p

suren_column_rp[0] = duration_sec
suren_column_rp[1] = cr_red_p

suren_column_rd[0] = duration_sec
suren_column_rd[1] = cr_red_d

filename_ccf = save_PIEcrosscorrelation_as + '.cor'   # change file name!
np.savetxt(
    filename_ccf,
    np.vstack(
        [
            time_axis,
            PIE_amplitude,
            suren_columnPIE
         ]
    ).T,
    delimiter='\t'
)

filename_fret = save_FRETcrosscorrelation_as + '.cor'  # change file name!
np.savetxt(
    filename_fret,
    np.vstack(
        [
            time_axis,
            FRET_amplitude,
            suren_columnFRET
         ]
    ).T,
    delimiter='\t'
)


filename_acf_prompt = save_ACF_green_prompt_as + '.cor'  # change file name!
np.savetxt(
    filename_acf_prompt,
    np.vstack(
        [
            time_axis,
            ACF_prompt_g_amplitude,
            suren_column_gp
         ]
    ).T,
    delimiter='\t'
)

filename_acf_prompt_red = save_ACF_red_prompt_as + '.cor'  # change file name!
np.savetxt(
    filename_acf_prompt_red,
    np.vstack(
        [
            time_axis,
            ACF_prompt_r_amplitude,
            suren_column_rp
         ]
    ).T,
    delimiter='\t'
)

filename_acf_delay = save_ACF_red_delay_as + '.cor'  # change file name!
np.savetxt(
    filename_acf_delay,
    np.vstack(
        [
            time_axis,
            ACF_delay_r_amplitude,
            suren_column_rd
         ]
    ).T,
    delimiter='\t'
)

########################################################
#  Plotting
########################################################

p.semilogx(time_axis, PIE_amplitude, label='PIE (gp-rd)')
p.semilogx(time_axis, ACF_prompt_r_amplitude, label='Red prompt')
p.semilogx(time_axis, ACF_prompt_g_amplitude, label='Green prompt')
p.semilogx(time_axis, ACF_delay_r_amplitude, label='Red delay')
p.semilogx(time_axis, FRET_amplitude, label='FRET (gp-rp)')

p.ylim(ymin=1)
p.xlabel('correlation time [ms]')
p.ylabel('correlation amplitude')
p.legend()
p.savefig(save_figure_as + ".svg", dpi=150)
p.show()
