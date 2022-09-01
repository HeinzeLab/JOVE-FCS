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
#  Microtimes are used to define "prompt" & "delay" of PIE
#  Image is also saved in svg-format
########################################################
#  Input parameter
# * marks variable area of file name
search_term = 'C:/Users/kah73xs/PycharmProjects/scripts/DNA10bpPIE_*.ptu'
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

# Caution! Data will be appended not in a sorted way
files = glob.glob(search_term)  # search term: which files should be appended
nr_of_curves = len(files)

data_sets = [tttrlib.TTTR(fn, 'PTU') for fn in files[0:]]
PIE_list = list()
FRET_list = list()
green_list = list()
red_prompt_list = list()
red_delay_list = list()

all_green_photons = 0
nr_of_green_photons = 0
all_red_photons = 0
nr_of_red_p_photons = 0
nr_of_red_d_photons = 0
total_duration = 0

for ds in data_sets:
    ########################################################
    #  Indices of data to correlate
    ########################################################
    data = tttrlib.TTTR(ds)
    header = data.header
    macro_time_calibration = data.header.macro_time_resolution  # unit seconds
    micro_time_resolution = data.header.micro_time_resolution
    macro_times = data.macro_times
    micro_times = data.micro_times
    number_of_bins = macro_time_calibration / micro_time_resolution
    PIE_windows_bins = int(number_of_bins / 2)

    all_green_indices = data.get_selection_by_channel([green_channel1, green_channel2])
    all_red_indices = data.get_selection_by_channel([red_channel1, red_channel2])
    green_indices1 = data.get_selection_by_channel([green_channel1])
    green_indices2 = data.get_selection_by_channel([green_channel2])
    red_indices1 = data.get_selection_by_channel([red_channel1])
    red_indices2 = data.get_selection_by_channel([red_channel2])

    all_green_photons = micro_times[all_green_indices]
    nr_of_green_photons += (np.array(np.where(all_green_photons <= PIE_windows_bins), dtype=np.int64)).size
    all_red_photons = micro_times[all_red_indices]
    nr_of_red_p_photons += (np.array(np.where(all_red_photons <= PIE_windows_bins), dtype=np.int64)).size
    nr_of_red_d_photons += (np.array(np.where(all_red_photons > PIE_windows_bins), dtype=np.int64)).size
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

    PIE_amplitude = PIEcorrelation_curve.correlation
    PIE_list.append(np.array(PIE_amplitude))

    # FRET crosscorrelation
    FRETcrosscorrelation_curve = tttrlib.Correlator(**settings)
    FRETcrosscorrelation_curve.set_events(t_green, w_gp, t_red, w_rp)

    FRET_amplitude = FRETcrosscorrelation_curve.correlation
    FRET_list.append(np.array(FRET_amplitude))

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
    
    ACF_prompt_g_amplitude = autocorr_prompt_g.correlation   
    green_list.append(np.array(ACF_prompt_g_amplitude))

    ACF_prompt_r_amplitude = autocorr_prompt_r.correlation
    red_prompt_list.append(np.array(ACF_prompt_r_amplitude))

    ACF_delay_r_amplitude = autocorr_delay_r.correlation
    red_delay_list.append(np.array(ACF_delay_r_amplitude))

########################################################
#  Get mean and standard deviation
########################################################

PIEcorrelation_amplitudes = np.array(PIE_list)
average_PIEcorrelation_amplitude = PIEcorrelation_amplitudes.mean(axis=0)
std_PIEcorrelation_amplitude = PIEcorrelation_amplitudes.std(axis=0)

FRETcrosscorrelation_amplitudes = np.array(FRET_list)
average_FRETcorrelation_amplitude = FRETcrosscorrelation_amplitudes.mean(axis=0)
std_FRETcorrelation_amplitude = FRETcrosscorrelation_amplitudes.std(axis=0)

GreenACF_amplitudes = np.array(green_list)
average_greenACF_amplitude = GreenACF_amplitudes.mean(axis=0)
std_greenACF_amplitude = GreenACF_amplitudes.std(axis=0)

RedACF_amplitudes = np.array(red_prompt_list)
average_redACF_amplitude = RedACF_amplitudes.mean(axis=0)
std_redACF_amplitude = RedACF_amplitudes.std(axis=0)

RedACF_amplitudes_delay = np.array(red_delay_list)
average_redACF_amplitude_delay = RedACF_amplitudes_delay.mean(axis=0)
std_redACF_amplitude_delay = RedACF_amplitudes_delay.std(axis=0)

########################################################
#  Save correlation curve
########################################################

#  bring the time axis to miliseconds
time_axis = PIEcorrelation_curve.x_axis * macro_time_calibration *1000

# fill 3rd column with 0's for compatibility with ChiSurf & Kristine
# 1st and 2nd entry of 3rd column are measurement duration & average countrate
suren_columnPIE = np.zeros_like(time_axis)
suren_columnFRET = np.zeros_like(time_axis)
suren_column_gp = np.zeros_like(time_axis)
suren_column_rp = np.zeros_like(time_axis)
suren_column_rd = np.zeros_like(time_axis)

duration_sec = total_duration / 1000
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

# 4th column will contain uncertainty
std_avg_PIEcorrelation_amplitude = std_PIEcorrelation_amplitude/np.sqrt(nr_of_curves)
std_FRETcrosscorrelation = std_FRETcorrelation_amplitude/np.sqrt(nr_of_curves)
std_autocorrelation_green_prompt = std_greenACF_amplitude/np.sqrt(nr_of_curves)
std_autocorrelation_red_prompt = std_redACF_amplitude/np.sqrt(nr_of_curves)
std_autocorrelation_red_delay = std_redACF_amplitude_delay/np.sqrt(nr_of_curves)


filename_ccf = save_PIEcrosscorrelation_as + '.cor'  # change file name!
np.savetxt(
    filename_ccf,
    np.vstack(
        [
            time_axis,
            average_PIEcorrelation_amplitude,
            suren_columnPIE,
            std_avg_PIEcorrelation_amplitude
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
            average_FRETcorrelation_amplitude,
            suren_columnFRET,
            std_FRETcrosscorrelation
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
            average_greenACF_amplitude,
            suren_column_gp,
            std_autocorrelation_green_prompt
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
            average_redACF_amplitude,
            suren_column_rp,
            std_autocorrelation_red_prompt
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
            average_redACF_amplitude_delay,
            suren_column_rd,
            std_autocorrelation_red_delay
         ]
    ).T,
    delimiter='\t'
)

########################################################
#  Plotting
########################################################

p.semilogx(time_axis, average_PIEcorrelation_amplitude, label='gp-rd')
p.semilogx(time_axis, average_redACF_amplitude, label='rp-rp')
p.semilogx(time_axis, average_greenACF_amplitude, label='gp-gp')
p.semilogx(time_axis, average_redACF_amplitude_delay, label='rd-rd')
p.semilogx(time_axis, average_FRETcorrelation_amplitude, label='gp-rp')

p.ylim(ymin=1)
p.xlabel('correlation time [ms]')
p.ylabel('correlation amplitude')
p.legend()
p.savefig(save_figure_as + ".svg", dpi=150)
p.show()
