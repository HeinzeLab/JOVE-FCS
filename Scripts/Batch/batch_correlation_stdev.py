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


def main(
        filename: str = '1_20min_1.ptu',
        filetype: str = 'PTU',
        cc_suffix: str = '_cross.cor',
        acf1_suffix: str = '_ch0_auto.cor',
        acf2_suffix: str = '_ch2_auto.cor',
        channel_number_ch1: tuple = (0,),
        channel_number_ch2: tuple = (2,),
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
    :param cc_suffix: suffix appended to saved results from crosscorrelation
    :param acf1_suffix: suffix appended to saved results from autocorrelation from ch1
    :param acf2_suffix: suffix appended to saved results from autocorrelation from ch2
    :param channel_number_ch1: channel 1 of experiment (perpendicular), here: [0]
    :param channel_number_ch2: channel 2 of experiment (parallel), here: [2]
    :param n_casc: n_bins and n_casc defines the settings of the multi-tau
    :param n_bins: correlation algorithm
    :param n_chunks: number of data slices
    :param make_plots: set "true" if images should be saved
    :param display_plot: set "true" if images should be displayed
    :return:
    """
    ########################################################
    #  Dataselection for the different correlations
    ########################################################
    basename = os.path.abspath(filename).split(".")[0]
    data = tttrlib.TTTR(filename, filetype)
    header = data.header
    macro_time_calibration = data.header.macro_time_resolution  # unit nanoseconds
    macro_times = data.macro_times
    duration = float(header.tag("TTResult_StopAfter")["value"])  # unit nanosecond
    duration_sec = duration / 1000
    window_length = duration_sec / n_chunks  # in seconds

    ########################################################
    #  Select the indices of the events to be correlated
    ########################################################

    green_s_indices = data.get_selection_by_channel([channel_number_ch1])
    green_p_indices = data.get_selection_by_channel([channel_number_ch2])

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
        tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel([channel_number_ch1])]
        tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel([channel_number_ch2])]
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
        tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel([channel_number_ch1])]
        tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel([channel_number_ch1])]
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
        tttr_ch1 = tttr_slice[tttr_slice.get_selection_by_channel([channel_number_ch2])]
        tttr_ch2 = tttr_slice[tttr_slice.get_selection_by_channel([channel_number_ch2])]
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
    #  Save correlation curve
    ########################################################
    # calculate the correct time axis in milliseconds
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
    std_avg_correlation_amplitude = std_correlation_amplitude / np.sqrt(n_chunks)
    std_avg_correlation_amplitude_ch1 = std_curve_ch1 / np.sqrt(n_chunks)
    std_avg_correlation_amplitude_ch2 = std_curve_ch2 / np.sqrt(n_chunks)

    filename_cc = basename + cc_suffix
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

    filename_acf1 = basename + acf1_suffix
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

    filename_acf2 = basename + acf2_suffix
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
    #  Plotting
    ########################################################
    if make_plots:
        p.semilogx(time_axis, average_correlation_amplitude, label='CCF')
        p.semilogx(time_axis, avg_curve_ch1, label='ACF1')
        p.semilogx(time_axis, avg_curve_ch2, label='ACF2')
        
        p.ylim(ymin=1)
        p.xlabel('correlation time [ms]')
        p.ylabel('correlation amplitude')
        p.legend()
        p.savefig(basename + "_corr.svg", dpi=150)
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
