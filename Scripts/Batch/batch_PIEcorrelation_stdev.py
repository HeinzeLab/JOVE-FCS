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
        n_chunks: int = 3,
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
    :param n_chunks: number of data slices
    :param make_plots: set "true" if images should be saved
    :param display_plot: set "true" if images should be displayed
    :return:
    """
    ################################################
    # Dataselection for the different correlation types
    ###############################################

    basename = os.path.abspath(filename).split(".")[0]
    data = tttrlib.TTTR(filename, filetype)
    header = data.header
    macro_time_calibration = data.header.macro_time_resolution  # unit nanoseconds
    micro_time_resolution = data.header.micro_time_resolution
    macro_times = data.macro_times
    micro_times = data.micro_times
    number_of_bins = macro_time_calibration/micro_time_resolution
    PIE_windows_bins = int(number_of_bins/2)
    n_correlation_casc = 25
    duration = float(header.tag("TTResult_StopAfter")["value"])    # unit millisecond
    duration_sec = duration / 1000
    window_length = duration_sec / n_chunks  # in seconds

    print("macro_time_calibration:", macro_time_calibration)
    print("micro_time_resolution:", micro_time_resolution)
    print("number_of_bins:", number_of_bins)
    print("PIE_windows_bins:", PIE_windows_bins)
    print("Duration [sec]:", duration_sec)
    print("Time window lenght [sec]:", window_length)

    ########################################################
    #  Indices of data to correlate
    ########################################################
    
    all_green_indices = data.get_selection_by_channel([green_channel_ch1, green_channel_ch2])
    all_red_indices = data.get_selection_by_channel([red_channel_ch1, red_channel_ch2])
    green_indices1 = data.get_selection_by_channel([green_channel_ch1])
    green_indices2 = data.get_selection_by_channel([green_channel_ch2])
    red_indices1 = data.get_selection_by_channel([red_channel_ch1])
    red_indices2 = data.get_selection_by_channel([red_channel_ch2])

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

    # PIE crosscorrelation (green prompt - red delay)
    PIEcrosscorrelation = tttrlib.Correlator(**settings)
    PIEcrosscorrelations = list()
    for start, stop in start_stop:
        indices = np.arange(start, stop, dtype=np.int64)
        tttr_slice = data[indices]
        tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel([green_channel_ch1, green_channel_ch2])]
        tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel([red_channel_ch1, red_channel_ch2])]
        mt_ch1 = micro_times[tttr_slice.get_selection_by_channel([green_channel_ch1, green_channel_ch2])]
        mt_ch2 = micro_times[tttr_slice.get_selection_by_channel([red_channel_ch1, red_channel_ch2])]
        w_ch1 = np.ones_like(mt_ch1, dtype=float)
        w_ch1[np.where(mt_ch1 > PIE_windows_bins)] *= 0.0
        w_ch2 = np.ones_like(mt_ch2, dtype=float)
        w_ch2[np.where(mt_ch2 < PIE_windows_bins)] *= 0.0
        PIEcrosscorrelation.set_tttr(
            tttr_1=tttr_ch1,
            tttr_2=tttr_ch2
        )
        PIEcrosscorrelation.set_weights(
            w_ch1,
            w_ch2
        )
        PIEcrosscorrelations.append(
            (PIEcrosscorrelation.x_axis, PIEcrosscorrelation.correlation)
        )
        
    PIEcrosscorrelations = np.array(PIEcrosscorrelations)

    # FRET crosscorrelation
    FRETcrosscorrelation = tttrlib.Correlator(**settings)
    FRETcrosscorrelations = list()
    for start, stop in start_stop:
        indices = np.arange(start, stop, dtype=np.int64)
        tttr_slice = data[indices]
        tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel([green_channel_ch1, green_channel_ch2])]
        tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel([red_channel_ch1, red_channel_ch2])]
        mt_ch1 = micro_times[tttr_slice.get_selection_by_channel([green_channel_ch1, green_channel_ch2])]
        mt_ch2 = micro_times[tttr_slice.get_selection_by_channel([red_channel_ch1, red_channel_ch2])]
        w_ch1 = np.ones_like(mt_ch1, dtype=float)
        w_ch1[np.where(mt_ch1 > PIE_windows_bins)] *= 0.0
        w_ch2 = np.ones_like(mt_ch2, dtype=float)
        w_ch2[np.where(mt_ch2 > PIE_windows_bins)] *= 0.0
        FRETcrosscorrelation.set_tttr(
            tttr_1=tttr_ch1,
            tttr_2=tttr_ch2
        )
        FRETcrosscorrelation.set_weights(
            w_ch1,
            w_ch2
        )
        FRETcrosscorrelations.append(
            (FRETcrosscorrelation.x_axis, FRETcrosscorrelation.correlation)
        )
        
    FRETcrosscorrelations = np.array(FRETcrosscorrelations)

    # Green grompt autocorrelation
    autocorr_prompt_g = tttrlib.Correlator(**settings)
    ACFs_prompt_green = list()
    for start, stop in start_stop:
        indices = np.arange(start, stop, dtype=np.int64)
        tttr_slice = data[indices]
        tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel([green_channel_ch1])]
        tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel([green_channel_ch2])]
        mt_ch1 = micro_times[tttr_slice.get_selection_by_channel([green_channel_ch1])]
        mt_ch2 = micro_times[tttr_slice.get_selection_by_channel([green_channel_ch2])]
        w_ch1 = np.ones_like(mt_ch1, dtype=float)
        w_ch1[np.where(mt_ch1 > PIE_windows_bins)] *= 0.0
        w_ch2 = np.ones_like(mt_ch2, dtype=float)
        w_ch2[np.where(mt_ch2 > PIE_windows_bins)] *= 0.0
        autocorr_prompt_g.set_tttr(
            tttr_1=tttr_ch1,
            tttr_2=tttr_ch2
        )
        autocorr_prompt_g.set_weights(
            w_ch1,
            w_ch2
        )
        ACFs_prompt_green.append(
            (autocorr_prompt_g.x_axis, autocorr_prompt_g.correlation)
        )
        
    ACFs_prompt_green = np.array(ACFs_prompt_green)

    # Red prompt autocorrelation
    autocorr_prompt_r = tttrlib.Correlator(**settings)
    ACFs_prompt_red = list()
    for start, stop in start_stop:
        indices = np.arange(start, stop, dtype=np.int64)
        tttr_slice = data[indices]
        tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel([red_channel_ch1])]
        tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel([red_channel_ch2])]
        mt_ch1 = micro_times[tttr_slice.get_selection_by_channel([red_channel_ch1])]
        mt_ch2 = micro_times[tttr_slice.get_selection_by_channel([red_channel_ch2])]
        w_ch1 = np.ones_like(mt_ch1, dtype=float)
        w_ch1[np.where(mt_ch1 > PIE_windows_bins)] *= 0.0
        w_ch2 = np.ones_like(mt_ch2, dtype=float)
        w_ch2[np.where(mt_ch2 > PIE_windows_bins)] *= 0.0
        autocorr_prompt_r.set_tttr(
            tttr_1=tttr_ch1,
            tttr_2=tttr_ch2
        )
        autocorr_prompt_r.set_weights(
            w_ch1,
            w_ch2
        )
        ACFs_prompt_red.append(
            (autocorr_prompt_r.x_axis, autocorr_prompt_r.correlation)
        )
        
    ACFs_prompt_red = np.array(ACFs_prompt_red)

    # Red delay autocorrelation
    autocorr_delay_r = tttrlib.Correlator(**settings)
    ACFs_delay_red = list()
    for start, stop in start_stop:
        indices = np.arange(start, stop, dtype=np.int64)
        tttr_slice = data[indices]
        tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel([red_channel_ch1])]
        tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel([red_channel_ch2])]
        mt_ch1 = micro_times[tttr_slice.get_selection_by_channel([red_channel_ch1])]
        mt_ch2 = micro_times[tttr_slice.get_selection_by_channel([red_channel_ch2])]
        w_ch1 = np.ones_like(mt_ch1, dtype=float)
        w_ch1[np.where(mt_ch1 < PIE_windows_bins)] *= 0.0
        w_ch2 = np.ones_like(mt_ch2, dtype=float)
        w_ch2[np.where(mt_ch2 < PIE_windows_bins)] *= 0.0
        autocorr_delay_r.set_tttr(
            tttr_1=tttr_ch1,
            tttr_2=tttr_ch2
        )
        autocorr_delay_r.set_weights(
            w_ch1,
            w_ch2
        )
        ACFs_delay_red.append(
            (autocorr_delay_r.x_axis, autocorr_delay_r.correlation)
        )
        
    ACFs_delay_red = np.array(ACFs_delay_red)
  
    ########################################################
    #  Get mean and standard deviation
    ########################################################

    PIEcorrelation_amplitudes = PIEcrosscorrelations[:, 1, :]
    average_PIEcorrelation_amplitude = PIEcorrelation_amplitudes.mean(axis=0)
    std_PIEcorrelation_amplitude = PIEcorrelation_amplitudes.std(axis=0)

    FRETcrosscorrelation_amplitudes = FRETcrosscorrelations[:, 1, :]
    average_FRETcorrelation_amplitude = FRETcrosscorrelation_amplitudes.mean(axis=0)
    std_FRETcorrelation_amplitude = FRETcrosscorrelation_amplitudes.std(axis=0)

    GreenACF_amplitudes = ACFs_prompt_green[:, 1, :]
    average_greenACF_amplitude = GreenACF_amplitudes.mean(axis=0)
    std_greenACF_amplitude = GreenACF_amplitudes.std(axis=0)

    RedACF_amplitudes = ACFs_prompt_red[:, 1, :]
    average_redACF_amplitude = RedACF_amplitudes.mean(axis=0)
    std_redACF_amplitude = RedACF_amplitudes.std(axis=0)

    RedACF_amplitudes_delay = ACFs_delay_red[:, 1, :]
    average_redACF_amplitude_delay = RedACF_amplitudes_delay.mean(axis=0)
    std_redACF_amplitude_delay = RedACF_amplitudes_delay.std(axis=0)

    ########################################################
    #  Save correlation curve
    ########################################################
    # calculates the correct time axis by multiplication of x-axis with macro_time
    time_axis = PIEcrosscorrelations[0, 0] * 1000

    # 2nd column contains the correlation amplitude
    PIEcrosscorrelation = average_PIEcorrelation_amplitude
    FRETcrosscorrelation = average_FRETcorrelation_amplitude
    autocorrelation_green_prompt = average_greenACF_amplitude
    autocorrelation_red_prompt = average_redACF_amplitude
    autocorrelation_red_delay = average_redACF_amplitude_delay

    # fill 3rd column with 0's for compatibility with ChiSurf & Kristine
    # 1st and 2nd entry of 3rd column are measurement duration & average countrate
    suren_columnPIE = np.zeros_like(time_axis)
    suren_columnFRET = np.zeros_like(time_axis)
    suren_column_gp = np.zeros_like(time_axis)
    suren_column_rp = np.zeros_like(time_axis)
    suren_column_rd = np.zeros_like(time_axis)

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

    # 4th column will contain uncertainty
    std_avg_PIEcorrelation_amplitude = std_PIEcorrelation_amplitude / np.sqrt(n_chunks)
    std_FRETcrosscorrelation = std_FRETcorrelation_amplitude / np.sqrt(n_chunks)
    std_autocorrelation_green_prompt = std_greenACF_amplitude / np.sqrt(n_chunks)
    std_autocorrelation_red_prompt = std_redACF_amplitude / np.sqrt(n_chunks)
    std_autocorrelation_red_delay = std_redACF_amplitude_delay / np.sqrt(n_chunks)

    filename_ccf = basename + PIE_suffix  # change file name!
    np.savetxt(
        filename_ccf,
        np.vstack(
            [
                time_axis,
                PIEcrosscorrelation,
                suren_columnPIE,
                std_avg_PIEcorrelation_amplitude
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
                FRETcrosscorrelation,
                suren_columnFRET,
                std_FRETcrosscorrelation
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
                autocorrelation_green_prompt,
                suren_column_gp,
                std_autocorrelation_green_prompt
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
                autocorrelation_red_prompt,
                suren_column_rp,
                std_autocorrelation_red_prompt
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
                autocorrelation_red_delay,
                suren_column_rd,
                std_autocorrelation_red_delay
             ]
        ).T,
        delimiter='\t'
    )

    ########################################################
    #  Plotting
    ########################################################
    if make_plots:
        p.semilogx(time_axis, PIEcrosscorrelation, label='PIE (gp-rd)')
        p.semilogx(time_axis, autocorrelation_red_prompt, label='Red prompt')
        p.semilogx(time_axis, autocorrelation_green_prompt, label='Green prompt')
        p.semilogx(time_axis, autocorrelation_red_delay, label='Red delay')
        p.semilogx(time_axis, FRETcrosscorrelation, label='FRET (gp-rp)')
        
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
