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

def select_by_deviation(
        deviations: typing.List[float],
        d_max: float = 2e-5,
) -> typing.List[int]:
    """ The single correlated time windows are now selected for further analysis based on their deviation
    to the first n curves
    :param deviations: list of deviations, calculated in the calculate_deviation function
    :param d_max: threshold, all curves which have a deviation value smaller than this are selected for further analysis
    :return: list of indices, the indices corresponds to the curves' number/time window
    """
    print("Selecting indices of curves by deviations.")
    devs = deviations[0]
    selected_curves_idx = list()
    for i, d in enumerate(devs):
        if d < d_max:
            selected_curves_idx.append(i)
    print("Total number of curves: ", len(devs))
    print("Selected curves: ", len(selected_curves_idx))
    return selected_curves_idx
     
def calculate_deviation(
        correlation_amplitudes: np.ndarray,
        comparison_start: int = 120,
        comparison_stop: int = 180,
        plot_dev: bool = False,
        n: int = 1,
) -> typing.List[float]:
    """Determines the similarity of the individual curves towards the first n curves
    The values of each correlation amplitude are averaged over a time range defined by start and stop
    This time range usually encompasses the diffusion time, i.e. is sample-specific
    The calculated average is compared to the mean of the first n curves

    :param correlation_amplitudes: array of correlation amplitudes
    :param comparison_start: index within the array of correlation amplitude which marks the start of comparison range
    :param comparison_stop: index within the array of correlation amplitude which marks the end of comparison range
    :param plot_dev: if set to True (default is True) plots the deviation
    :param n: to how many curves of the beginning the other curves should be compared to
    :return: list of deviations calculated as difference to the starting amplitudes
    """
    print("Calculating deviations.")
    print("Comparison time range:", comparison_start, "-", comparison_stop)
    deviation = list()
    print("compared to the first", n, "curves")
    # calculates from every curve the difference to the mean of the first N curves, squares this value
    # and divides this value by the number of curves
    if (comparison_start is None) or (comparison_stop is None):
        ds = np.mean(
            (correlation_amplitudes - correlation_amplitudes[:n].mean(axis=0))**2.0, axis=1
        ) / len(correlation_amplitudes)
        deviation.append(ds)
    else:
        ca = correlation_amplitudes[:, comparison_start: comparison_stop]
        ds = np.mean(
            (ca - ca[:n].mean(axis=0)) ** 2.0, axis=1
        ) / (comparison_stop - comparison_start)
        deviation.append(ds)
    if plot_dev:
        x = np.arange(len(ds))
        p.semilogy(x, ds)
        p.show()
    return deviation

def calculate_countrate(
        timewindows: typing.List[np.ndarray],
        time_window_size_seconds: float = 2.0,
) -> typing.List[float]:
    """based on the sliced timewindows the average countrate for each slice is calculated
    :param timewindows: list of numpy arrays, the indices which have been returned from getting_indices_of_time_windows
    :param time_window_size_seconds: The size of the time windows in seconds
    :return: list of average countrate (counts/sec) for the individual time windows
    """
    print("Calculating the average count rate...")
    avg_count_rate = list()
    index = 0
    while index < len(timewindows):
        nr_of_photons = len(timewindows[index])  # determines number of photons in a time slice
        avg_countrate = nr_of_photons / time_window_size_seconds  # division by length of time slice in seconds
        avg_count_rate.append(avg_countrate)
        index += 1
    return avg_count_rate

def main(
        filename: str = '1_20min_1.ptu',
        filetype: str = 'PTU',
        n_casc_fine: int = 37,
        n_casc_coarse: int = 25,
        n_bins: int = 9,
        comparison_start: int = 80,
        comparison_stop: int = 100,
        n_comparison: int = 1,
        deviation_max: float = 2e-5,
        cc_suffix: str = '_cross.cor',
        acf1_suffix: str = '_ch0_auto.cor',
        acf2_suffix: str = '_ch2_auto.cor',
        average_count_rates_suffix: str = '_avg_countrate.txt',
        deviations_suffix: str = '_deviations.txt',
        channel_number_ch1: tuple = (0,),
        channel_number_ch2: tuple = (2,),
        make_plots: bool = True,
        display_plot: bool = False,
        g_factor: float = 0.8,
        time_window_size: float = 60.0  # time window size in seconds
):
    """

    :param filename: name of file to be read
    :param filetype: filetype to be read, can be ht3, ptu, ...
    :param n_casc_fine: nr of correlation cascades for full cross-correlation
    :param n_casc_coarse: nr of correlation cascades for auto-correlation (no microtimes used)
    :param n_bins: n_bins and n_casc defines the settings of the multi-tau correlation algorithm
    :param comparison_start: start bin for comparison range of correlation amplitude (not time but bin nr!)
    :param comparison_stop: end bin for comparison range of correlation amplitude (not time but bin nr!)
    :param n_comparison: nr of curves to average and to compare all other curves to
    :param deviation_max: upper threshold of deviation, only curve with d < dmax are selected
    :param cc_suffix: suffix appended to saved results from crosscorrelation
    :param acf1_suffix: suffix appended to saved results from autocorrelation from ch1
    :param acf2_suffix: suffix appended to saved results from autocorrelation from ch2
    :param average_count_rates_suffix: suffix appended to saved results from count rate calculation
    :param deviations_suffix: suffix appended to saved deviations
    :param channel_number_ch1: channel 1 of experiment (perpendicular), here: [0]
    :param channel_number_ch2: channel 2 of experiment (parallel), here: [2]
    :param make_plots: set "true" if images should be saved
    :param display_plot: set "true" if images should be displayed
    :param g_factor: enter value of the g-factor calibrated from a reference experiment
    :param time_window_size: averaging window in seconds
    :return:
    """
    ########################################################
    #  Dataselection for the different correlations
    ########################################################
    basename = os.path.abspath(filename).split(".")[0]
    data = tttrlib.TTTR(filename, filetype)
    # rep rate = 80 MHz
    header = data.header
    macro_time_calibration = data.header.macro_time_resolution  # unit seconds
    macro_times = data.macro_times
    micro_times = data.micro_times
    micro_time_resolution = data.header.micro_time_resolution

    ########################################################
    #  Select the indices of the events to be correlated
    ########################################################

    green_s_indices = data.get_selection_by_channel([channel_number_ch1])
    green_p_indices = data.get_selection_by_channel([channel_number_ch2])

    nr_of_green_s_photons = len(green_s_indices)
    nr_of_green_p_photons = len(green_p_indices)

    # Get the start-stop indices of the data slices
    time_windows = data.get_ranges_by_time_window(
        time_window_size, macro_time_calibration=macro_time_calibration)
    start_stop = time_windows.reshape((len(time_windows)//2, 2))
    print(start_stop)
    
    ########################################################
    #  Correlate the pieces
    ########################################################

    # Correlator settings, define the identical settings once
    settings = {
        "method": "default",
        "n_bins": n_bins,  # n_bins and n_casc defines the settings of the multi-tau
        "n_casc": n_casc_fine,  # correlation algorithm
        "make_fine": True  # Use the microtime information
    }

    # Crosscorrelation
    crosscorrelation = tttrlib.Correlator(**settings)
    crosscorrelations = list()
    nr_photons_ch1 = list()
    nr_photons_ch2 = list()
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
        photons_ch1 = len(tttr_ch1)
        photons_ch2 = len(tttr_ch2)
        nr_photons_ch1.append(photons_ch1)
        nr_photons_ch2.append(photons_ch2)
               
    crosscorrelations = np.array(crosscorrelations)
    nr_photons_ch1 = np.array(nr_photons_ch1)
    nr_photons_ch2 = np.array(nr_photons_ch2)

    ########################################################
    #  Option: get autocorrelation curves
    ########################################################
    
    # Correlator settings, define the identical settings once
    settings_acf = {
        "method": "default",
        "n_bins": n_bins,  # n_bins and n_casc defines the settings of the multi-tau
        "n_casc": n_casc_coarse,  # correlation algorithm
        "make_fine": True  # Use the microtime information
    }
    
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
    #  Select the curves
    ########################################################

    # comparison is only made for the crosscorrelation curves
    # autocorrelation curves are calculated based on the curve_ids selected by crosscorr
    correlation_amplitudes = crosscorrelations[:, 1, :]
    average_correlation_amplitude = correlation_amplitudes.mean(axis=0)

    # adjust comparison_start & stop according to your diffusion time
    # the selected values here encompass 1 ms -> 100 ms
    deviation_from_mean = calculate_deviation(
        correlation_amplitudes=correlation_amplitudes,
        comparison_start=comparison_start,
        comparison_stop=comparison_stop,
        n=n_comparison
    )

    # select the curves with a small enough deviation to be considered in the further analysis
    selected_curves_idx = select_by_deviation(
        deviations=deviation_from_mean,
        d_max=deviation_max
    )

    ########################################################
    #  Average selected curves
    ########################################################
    selected_curves = list()
    for curve_idx in selected_curves_idx:
        selected_curves.append(
            correlation_amplitudes[curve_idx]
        )

    selected_curves = np.array(selected_curves)
    avg_curve = np.mean(selected_curves, axis=0)
    std_curve = np.std(selected_curves, axis=0)

    ########################################################
    #  Average selected autocorrelation curves
    ########################################################
    selected_curves_ch1 = list()
    for curve_idx in selected_curves_idx:
        selected_curves_ch1.append(
            autocorrs_ch1[curve_idx]
        )
    selected_curves_ch1 = np.array(selected_curves_ch1)
    avg_curve_ch1 = np.mean(selected_curves_ch1, axis=0)
    std_curve_ch1 = np.std(selected_curves_ch1, axis=0)

    selected_curves_ch2 = list()
    for curve_idx in selected_curves_idx:
        selected_curves_ch2.append(
            autocorrs_ch2[curve_idx]
        )
    selected_curves_ch2 = np.array(selected_curves_ch2)
    avg_curve_ch2 = np.mean(selected_curves_ch2, axis=0)
    std_curve_ch2 = np.std(selected_curves_ch2, axis=0)

    ########################################################
    #  Save correlation curve
    ########################################################
    time_axis = crosscorrelations[0, 0] * 1000  # time axis in milliseconds
    time_axis_acf = autocorrs_ch2[0, 0] * 1000  # time axis in milliseconds
    suren_column = np.zeros_like(time_axis)  # fill 3rd column with 0's for compatibility with ChiSurf
    suren_column_acf = np.zeros_like(time_axis_acf)
    std_avg_correlation_amplitude = std_curve / np.sqrt(len(selected_curves))
    std_avg_correlation_amplitude_ch1 = std_curve_ch1[1] / np.sqrt(len(selected_curves))
    std_avg_correlation_amplitude_ch2 = std_curve_ch2[1] / np.sqrt(len(selected_curves))
    # 4th column contains standard deviation from the average curve calculated above
    filename_cc = basename + cc_suffix
    np.savetxt(
        filename_cc,
        np.vstack(
            [
                time_axis,
                avg_curve,
                suren_column,
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
                time_axis_acf,
                avg_curve_ch1[1],
                suren_column_acf,
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
                time_axis_acf,
                avg_curve_ch2[1],
                suren_column_acf,
                std_avg_correlation_amplitude_ch2
            ]
        ).T,
        delimiter='\t'
    )

    ########################################################
    #  Calculate steady-state anisotropy & save count rate per slice
    ########################################################

    total_countrate = (nr_photons_ch1 + nr_photons_ch2) / time_window_size
    parallel_channel = nr_photons_ch2 / time_window_size
    perpendicular_channel = nr_photons_ch1 / time_window_size
    rss = (parallel_channel - g_factor * perpendicular_channel) / (parallel_channel + 2 * g_factor * perpendicular_channel)

    filename_average_count_rates = basename + average_count_rates_suffix
    np.savetxt(
        filename_average_count_rates,
        np.vstack(
            [
                total_countrate,
                perpendicular_channel,
                parallel_channel,
                rss
            ]
        ).T,
        delimiter='\t'
    )

    ########################################################
    #  Save deviations
    ########################################################
    deviations = np.array(deviation_from_mean)

    filename_deviations = basename + deviations_suffix
    np.savetxt(
        filename_deviations,
        np.vstack(
            [
                deviations,
            ]
        ).T,
        delimiter='\t'
    )

    print("Done.")

    ########################################################
    #  Plotting
    ########################################################
    if make_plots:
        fig, ax = p.subplots(nrows=2, ncols=2, constrained_layout=True)

        devx = np.arange(len(deviation_from_mean[0]))

        ax[0, 0].semilogy(devx, deviation_from_mean[0], label='deviations')
        ax[0, 1].semilogx(time_axis, avg_curve, label='gs-gp')
        ax[0, 1].semilogx(time_axis_acf, avg_curve_ch1[1], label='gs-gs')
        ax[0, 1].semilogx(time_axis_acf, avg_curve_ch2[1], label='gp-gp')
        ax[1, 0].plot(perpendicular_channel, label='CR gs(perpendicular)')
        ax[1, 0].plot(parallel_channel, label='CR gp(parallel)')
        ax[1, 1].plot(rss, label='rss')

        ax[0, 0].set_xlabel('slice #')
        ax[0, 0].set_ylabel('deviation')
        ax[0, 1].set_ylim(ymin=1)
        ax[0, 1].set_xlabel('correlation time [ms]')
        ax[0, 1].set_ylabel('correlation amplitude')
        ax[1, 0].set_xlabel('slice #')
        ax[1, 0].set_ylabel('countrate [Hz]')
        ax[1, 1].set_xlabel('slice #')
        ax[1, 1].set_ylabel('steady-state anisotropy')

        legend = ax[0, 0].legend()
        legend = ax[0, 1].legend()
        legend = ax[1, 0].legend()
        legend = ax[1, 1].legend()
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




