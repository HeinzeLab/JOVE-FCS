#!/usr/bin/env python

from __future__ import annotations

import glob
import numpy as np
import tttrlib

########################################################
#  This script calculates the average count rates for each
#  channel from all file in the specified folder and with
#  the specified search pattern (e.g. *.ptu)
########################################################

path = r"\\HC1008\Users\AG Heinze\DATA\FCSSetup\2021\20210223_MF_khm_Cal\*.ptu"
channel1 = [0]  # usually green perpendicular
channel2 = [2]  # usually green parallel
channel3 = [1]  # usually red perpendicular
channel4 = [3]  # usually red parallel
save_file_as = r"\\HC1008\Users\AG Heinze\DATA\FCSSetup\2021\20210223_MF_khm_Cal\avg_countrate.txt"

# initialize list of parameter to be saved at the end
list_filenames = list()  # filenames
list_durations = list()  # measurement duration in seconds
list_cr_green_ch1 = list()  # countrate green channel 1 in kHz
list_cr_green_ch2 = list()  # countrate green channel 2 in kHz
list_cr_red_ch1 = list()  # countrate red channel 1 in kHz
list_cr_red_ch2 = list()  # countrate red channel 2 in kHz

# loop of the whole folder and calculate the countrates
for file in glob.glob(path):
    data = tttrlib.TTTR(file, 'PTU')  # read information from data file
    header = data.get_header()
    header_data = header.data
    micro_times = data.get_micro_time()
    micro_time_resolution = header.micro_time_resolution
    macro_time_calibration_ns = header.macro_time_resolution  # unit nanoseconds
    macro_time_calibration_ms = macro_time_calibration_ns / 1e6  # macro time calibration in milliseconds
    macro_times = data.get_macro_time()  # get macro times

    # get indices of counted photon events per channels
    green_s_indices = np.array(data.get_selection_by_channel(channel1), dtype=np.int64)
    green_p_indices = np.array(data.get_selection_by_channel(channel2), dtype=np.int64)
    red_s_indices = np.array(data.get_selection_by_channel(channel3), dtype=np.int64)
    red_p_indices = np.array(data.get_selection_by_channel(channel4), dtype=np.int64)

    # get number of collected photons per channel based on the number of photon events
    nr_of_green_s_photons = len(green_s_indices)
    nr_of_green_p_photons = len(green_p_indices)
    nr_of_red_s_photons = len(red_s_indices)
    nr_of_red_p_photons = len(red_p_indices)

    # determine the measurement time in seconds
    duration = float(header_data["TTResult_StopAfter"])  # unit millisecond
    duration_sec = duration / 1000  # convert time to seconds

    # the average countrate in kHz is calculated by dividing the number of collected photons in the respective channel
    # by the measurement time
    cr_green_ch1 = nr_of_green_s_photons / duration_sec / 1000  # kHz
    cr_green_ch2 = nr_of_green_p_photons / duration_sec / 1000  # kHz
    cr_red_ch1 = nr_of_red_s_photons / duration_sec / 1000  # kHz
    cr_red_ch2 = nr_of_red_p_photons / duration_sec / 1000  # kHz

    # all parameter are appended to a growing list
    list_filenames.append(str(file))
    list_durations.append(duration_sec)
    list_cr_green_ch1.append(cr_green_ch1)
    list_cr_green_ch2.append(cr_green_ch2)
    list_cr_red_ch1.append(cr_red_ch1)
    list_cr_red_ch2.append(cr_red_ch2)

# column header of saved txt file
header = 'Filename\t Duration [s]\t CR Channel 1 [kHz]\t CR Channel 2 [kHz]\t CR Channel 3 [kHz]\t CR Channel 4 [kHz]'

# save results as txt file
np.savetxt(
    save_file_as,
    np.vstack(
        [
            list_filenames,
            list_durations,
            list_cr_green_ch1,
            list_cr_green_ch2,
            list_cr_red_ch1,
            list_cr_red_ch2
        ]
    ).T,
    delimiter='\t', fmt="%s", header=header
)
