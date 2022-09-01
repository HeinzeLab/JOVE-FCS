# Tested with tttrlib 0.21.9
###################################
# Katherina Hemmen ~ Core Unit Fluorescence Imaging ~ RVZ
# katherina.hemmen@uni-wuerzburg.de
###################################

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
        average_count_rates_suffix: str = '_avg_countrate.txt',
        anisotropy_suffix: str = '_anisotropy.txt',
        channel_number_ch1: tuple = (0,),
        channel_number_ch2: tuple = (2,),
        window_size: int = 3, 
        make_plots: bool = True,
        display_plot: bool = False,
        anisotropy: bool = True,
        g_factor: float = 0.8
):
    """
    for batch export, please change in the settings file
    :param filename: name of file to be read
    :param filetype: filetype to be read, can be ht3, ptu, ...
    :param average_count_rates_suffix: suffix appended to saved results
    :param anisotropy_suffix: suffix appended to saved anisotropy calculation
    :param channel_number_ch1: channel 1 of experiment (perpendicular), here: [0]
    :param channel_number_ch2: channel 2 of experiment (parallel), here: [2]
    :param n_chunks: number of data slices
    :param make_plots: set "true" if images should be saved
    :param display_plot: set "true" if images should be displayed
    :param anisotropy: set "true" if anisotropy should be calculated
    :param g_factor: enter value of the g-factor calibrated fram a reference experiment
    :return:
    """
    ########################################################
    #  Dataselection
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
    
    # Get the start-stop indices of the data slices
    time_windows = data.get_ranges_by_time_window(
        window_size, macro_time_calibration=macro_time_calibration)
    start_stop = time_windows.reshape((len(time_windows)//2, 2))
    
    CR_ch1 = list()
    CR_ch2 = list()
    
    for start, stop in start_stop:
        indices = np.arange(start, stop, dtype=np.int64)
        tttr_slice = data[indices]
        ch1 = micro_times[tttr_slice.get_selection_by_channel([channel_number_ch1])]
        nr_photons_ch1 = len(ch1)
        ch2 = micro_times[tttr_slice.get_selection_by_channel([channel_number_ch2])]
        nr_photons_ch2 = len(ch2)

        avg_countrate_ch1 = nr_photons_ch1 / window_size
        avg_countrate_ch2 = nr_photons_ch2 / window_size

        CR_ch1.append(avg_countrate_ch1)
        CR_ch2.append(avg_countrate_ch2)
    
    avg_CR_ch1 = np.array(CR_ch1)
    avg_CR_ch2 = np.array(CR_ch2)
    total_countrate = avg_CR_ch1 + avg_CR_ch2

    filename_avg_countrate = basename + average_count_rates_suffix
    np.savetxt(
        filename_avg_countrate,
        np.vstack(
            [
                total_countrate,
                avg_CR_ch1,
                avg_CR_ch2,
            ]
        ).T,
        delimiter='\t'
    )

    if make_plots:
        p.plot(avg_CR_ch1, label='CR Ch1(perpendicular)')
        p.plot(avg_CR_ch2, label='CR Ch2(parallel)')
        p.xlabel('slice #')
        p.ylabel('countrate [Hz]')
        p.legend()
        p.savefig(basename + ".svg", dpi=150)
        if display_plot:
            p.show()
        p.close()

    if anisotropy:
        parallel_channel = avg_CR_ch2
        perpendicular_channel = avg_CR_ch1
        rss = (parallel_channel - g_factor * perpendicular_channel)/(parallel_channel + 2 * g_factor * perpendicular_channel)

        filename_anisotropy = basename + anisotropy_suffix
        np.savetxt(
            filename_anisotropy,
            np.vstack(
                [
                    rss
                ]
            ).T,
            delimiter='\t'
        )

        if make_plots:
            p.plot(rss, label='rss')
            p.xlabel('slice #')
            p.ylabel('steady-state anisotropy')
            p.legend()
            p.savefig(basename + "_anisotropy.svg", dpi=150)
            if display_plot:
                p.show()
            p.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computes the average count rates and optional the anisotropy.')

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
    print("Compute average count rates")
    print("====================")
    print("Settings file: %s" % settings_file)
    print("Search path: %s" % search_path)
    print("Search string: %s" % search_string)

    for path in pathlib.Path(search_path).rglob(search_string):
        filename = str(path.absolute())
        print("Processing: %s" % filename)
        settings['filename'] = filename
        main(**settings)
