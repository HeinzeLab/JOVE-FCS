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
        filename: str = 'A488_1.ptu',
        filetype: str = 'PTU',
        binning: int = 4,
        ch1_suffix: str = '_green_s',
        ch2_suffix: str = '_green_p',
        channel_number_ch1: tuple = (0,),
        channel_number_ch2: tuple = (2,),
        n_chunks: int = 3, 
        make_plots: bool = True,
        display_plot: bool = False,
        jordi: bool = True,
        anisotropy: bool = True,
        g_factor: float = 0.98,
        time_window_size: float = 59.0
):
    """

    for batch export, please change in the settings file
    :param filename: name of file to be read
    :param filetype: filetype to be read, can be ht3, ptu, ...
    :param binning: factor with which the histogram should be rebinned for export, use multiples of 2, =1 for no binning
    :param ch1_suffix: suffix appended to saved results from channel 1
    :param ch2_suffix: suffix appended to saved results from channel 2
    :param channel_number_ch1: channel 1 of experiment (perpendicular), here: [0]
    :param channel_number_ch2: channel 2 of experiment (parallel), here: [2]
    :param n_chunks: number of data slices
    :param make_plots: set "true" if images should be saved
    :param display_plot: set "true" if images should be displayed
    :param jordi: set "true" for jordi-format export (stacked parallel-perpendicular, no time axis)
    :param anisotropy: set "true" for calculation and export of anisotropy decay
    :param g_factor: g-factor, corrects for difference in detection between the parallel and perpendicular channel
    :param time_window_size: averaging window in seconds
    :return:
    """
    ########################################################
    #  Dataselection for the different correlations
    ########################################################
    basename = os.path.abspath(filename).split(".")[0]
    data = tttrlib.TTTR(filename, filetype)
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

    binning = binning  # Binning factor
    # This is the max nr of bins the data should contain:
    expected_nr_of_bins = int(macro_time_calibration // micro_time_resolution)
    # After binning the nr of bins is reduced:
    binned_nr_of_bins = int(expected_nr_of_bins // binning)

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
        green_s = micro_times[tttr_slice.get_selection_by_channel([channel_number_ch1])]
        green_p = micro_times[tttr_slice.get_selection_by_channel([channel_number_ch2])]
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

        output_filename_s = basename + ch1_suffix + str(i) + '.txt'
        np.savetxt(
            output_filename_s,
            np.vstack([x_axis, green_s_counts_cut]).T
        )
        output_filename_p = basename + ch2_suffix + str(i) + '.txt'
        np.savetxt(
            output_filename_p,
            np.vstack([x_axis, green_p_counts_cut]).T
        )

        if make_plots:
            p.semilogy(x_axis, green_s_counts_cut, label='gs')
            p.semilogy(x_axis, green_p_counts_cut, label='gp')

            p.xlabel('time [ns]')
            p.ylabel('Counts')
            p.legend()
            p.savefig(basename + str(i) + ".svg", dpi=150)
            if display_plot:
                p.show()
            p.close()

        # Optional: jordi format for direct reading in FitMachine & ChiSurf(2015-2017)
        if jordi:
            jordi_counts_green = np.concatenate([green_p_counts_cut, green_s_counts_cut])

            output_filename = basename + '_jordi_' + str(i) + '.txt'
            np.savetxt(
                output_filename,
                np.vstack([jordi_counts_green]).T
            )

        ########################################################
        #  Optional: calculation of anisotropy decay
        ########################################################

        if anisotropy:
            aniso_decay = (green_p_counts_cut - g_factor * green_s_counts_cut) / (
                    green_p_counts_cut + 2 * g_factor * green_s_counts_cut)

            aniso_export = basename + "_aniso_" + str(i) + ".txt"
            np.savetxt(
                aniso_export,
                np.vstack([x_axis, aniso_decay]).T
            )

            if make_plots:
                p.plot(x_axis, aniso_decay, label='r(t)')
                p.xlabel('time [ns]')
                p.ylabel('r(t)')
                p.legend()
                p.savefig(basename + "_aniso_" + str(i) + ".svg", dpi=150)
                if display_plot:
                    p.show()
                p.close()
                
        i+=1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute a decays histograms in slices for data in folder.')

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
    print("Compute decays")
    print("====================")
    print("Settings file: %s" % settings_file)
    print("Search path: %s" % search_path)
    print("Search string: %s" % search_string)

    for path in pathlib.Path(search_path).rglob(search_string):
        filename = str(path.absolute())
        print("Processing: %s" % filename)
        settings['filename'] = filename
        main(**settings)




