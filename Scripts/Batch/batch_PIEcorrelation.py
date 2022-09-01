# Tested with tttrlib 0.21.9
###################################
# Katherina Hemmen ~ Core Unit Fluorescence Imaging ~ RVZ
# katherina.hemmen@uni-wuerzburg.de
###################################

#!/usr/bin/env python

from __future__ import annotations
import argparse
import os
import pathlib
import yaml

import numpy as np
import pylab as p
import tttrlib


###################################
# Definition of input parameter
##################################

def main(
        filename: str = '1_20min_1.ptu',
        filetype: str = 'PTU',
        PIE_suffix: str = '_PIE.cor',
        FRET_suffix: str = '_FRET.cor',
        green_prompt_suffix: str = '_gp.cor',
        red_prompt_suffix: str = '_rp.cor',
        red_delay_suffix: str = '_rd.cor',
        green_channel_ch1: tuple = (0,),
        green_channel_ch2: tuple = (2,),
        red_channel_ch1: tuple = (1,),
        red_channel_ch2: tuple = (3,),
        n_casc: int = 25,
        n_bins: int = 9,
        make_plots: bool = True,
        display_plot: bool = False
):
    """
    for batch export, please change in the settings file
    :param filename: name of file to be read
    :param filetype: filetype to be read, can be ht3, ptu, ...
    :param PIE_suffix: suffix appended to saved results from PIE channels
    :param FRET_suffix: suffix appended to saved results from FRET channels
    :param green_prompt_suffix: suffix appended to saved results from donor prompt channels
    :param red_prompt_suffix: suffix appended to saved results from acceptor prompt channels
    :param red_delay_suffix: suffix appended to saved results from acceptor delay channels
    :param PIE_number_ch1: prompt channels of PIE experiment, here: [0,2]
    :param PIE_number_ch2: delay channels of PIE experiment, here: [1,3]
    :param green_channel_ch1: donor channel 1 (perpendicular), here: [0]
    :param green_channel_ch2: donor channel 2 (parallel), here: [0]
    :param red_channel_ch1: acceptor channel 1 (perpendicular), here: [0]
    :param red_channel_ch2: acceptor channel 2 (parallel), here: [0]
    :param n_casc: n_bins and n_casc defines the settings of the multi-tau
    :param n_bins: correlation algorithm
    :param make_plots: set "true" if images should be saved
    :param display_plot: set "true" if images should be displayed
    :return:
    """
    ################################################
    # Dataselection for the different correlation types
    ###############################################

    basename = os.path.abspath(filename).split(".")[0]
    data = tttrlib.TTTR(filename, filetype)
    # rep rate = 80 MHz
    header = data.header
    macro_time_calibration = data.header.macro_time_resolution  # unit seconds
    micro_time_resolution = data.header.micro_time_resolution
    macro_times = data.macro_times
    micro_times = data.micro_times
    number_of_bins = macro_time_calibration/micro_time_resolution
    PIE_windows_bins = int(number_of_bins/2)

    print("macro_time_calibration:", macro_time_calibration)
    print("micro_time_resolution:", micro_time_resolution)
    print("number_of_bins:", number_of_bins)
    print("PIE_windows_bins:", PIE_windows_bins)

    ########################################################
    #  Indices of data to correlate
    ########################################################

    # the dtype to int64 otherwise numba jit has hiccups
    all_green_indices = data.get_selection_by_channel([green_channel_ch1, green_channel_ch2])
    all_red_indices = data.get_selection_by_channel([red_channel_ch1, red_channel_ch2])
    green_indices1 = data.get_selection_by_channel([green_channel_ch1])
    green_indices2 = data.get_selection_by_channel([green_channel_ch2])
    red_indices1 = data.get_selection_by_channel([red_channel_ch1])
    red_indices2 = data.get_selection_by_channel([red_channel_ch2])

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

    filename_ccf = basename + PIE_suffix  # change file name!
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

    filename_fret = basename + FRET_suffix  # change file name!
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

    filename_acf_prompt = basename + green_prompt_suffix
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

    filename_acf_prompt_red = basename + red_prompt_suffix
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

    filename_acf_delay = basename + red_delay_suffix
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
    if make_plots:
        p.semilogx(time_axis, PIE_amplitude, label='PIE (gp-rd)')
        p.semilogx(time_axis, ACF_prompt_r_amplitude, label='Red prompt')
        p.semilogx(time_axis, ACF_prompt_g_amplitude, label='Green prompt')
        p.semilogx(time_axis, ACF_delay_r_amplitude, label='Red delay')
        p.semilogx(time_axis, FRET_amplitude, label='FRET (gp-rp)')
        
        p.ylim(ymin=1)
        p.xlabel('correlation time [ms]')
        p.ylabel('correlation amplitude')
        p.legend()
        p.savefig(basename + ".svg", dpi=150)
        if display_plot:
            p.show()
        p.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute a FCS curves in folder.')

    parser.add_argument(
        '--settings',
        help='YAML file containing the settings that are used to compute the correlations in the sub-folders.'
    )

    parser.add_argument(
        '--path',
        help='Folder name that will be processed. The folder is recursively searched for files.'
    )

    args = parser.parse_args()
    search_path = args.path
    settings_file = args.settings
    settings = dict()
    with open(settings_file, 'r') as fp:
        settings.update(
            yaml.load(fp.read(), Loader=yaml.FullLoader)
        )
    search_string = settings.pop('search_string')
    print("Compute correlations")
    print("====================")
    print("Settings file: %s" % settings_file)
    print("Search path: %s" % search_path)
    print("Search string: %s" % search_string)

    for path in pathlib.Path(search_path).rglob(search_string):
        filename = str(path.absolute())
        print("Processing: %s" % filename)
        settings['filename'] = filename
        main(**settings)
