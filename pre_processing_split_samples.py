"""
File which pre processes the data and splits into equal length chunks
"""

import glob, os
import numpy as np
import re
import scipy
import sys  
from scipy import signal
from scipy import pi
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np    
from scipy.signal import butter, lfilter, freqz 
from recategorise_classes import recategorise_classes

global_file_counter = 0
destination_filename = "./split_processed_data/"

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def write_to_file(ecg_plot, rhythm_name, global_file_counter):
    if not os.path.exists(destination_filename+rhythm_name):
        os.makedirs(destination_filename+rhythm_name)

    if len(ecg_plot) != 0:
        #f= open("./split_processed_data/"+rhythm_name+"/ecg_"+str(global_file_counter)+".ecg","wb")
        n_counter = 0
        LENGTH = 400
        cutoff = 100
        order = 6
        fs = 100000.0
        ecg_plot_filtered = butter_highpass_filter(ecg_plot, cutoff, fs, order)
        #ecg_plot_filtered = butter_lowpass_filter(ecg_plot_filtered, 10000, fs, order)

        if len(ecg_plot_filtered) >= LENGTH:
            complete_chunks = len(ecg_plot_filtered)//LENGTH

            for i in range(complete_chunks):
                start_index = i * LENGTH
                end_index = start_index + LENGTH

                new_ecg = ecg_plot_filtered[start_index:end_index]
                new_ecg = [int(round(x*100)) for x in new_ecg]

                if(len(set(new_ecg))==1):
                    print("All elements in list are same "+rhythm_name)
                else:
                    f= open(destination_filename+rhythm_name+"/ecg_"+str(global_file_counter)+".ecg","wb")

                    for i,value_int in enumerate(new_ecg):
                      
                        #value = value * 100
                        #value_int = int(round(value))
                        to_write = str(value_int) + " "
                        bin = np.int16(to_write)
                        f.write(bin)

                    global_file_counter+=1
                    f.close()

        # for i,value in enumerate(ecg_plot_filtered):
        #     value_int = int(round(value))
        #     if (n_counter % samples == 0):
        #         f.close()
        #         global_file_counter = global_file_counter + 1
                
            
        #     n_counter = n_counter + 1
        #     to_write = str(value_int) + " "
        #     bin = np.int16(to_write)

        #     f.write(bin)
        # f.close()
        # global_file_counter = global_file_counter + 1
    #global_file_counter+=1
    return global_file_counter

ecg_files_found = []
ecg_plots = []
labels = []
information_label = []

onsets = []
offsets = []
rhythm_name = []

for file in glob.glob("./raw_ecg_data/*.ecg"):

    f = open(file,"r")

    plot = np.fromfile(f, dtype=np.int16)
    ecg_plots.append(plot)

    os.path.splitext(file)
    file = os.path.splitext(file)[0]
    ecg_files_found.append(file)

    episode_files = []
    for episode_file in glob.glob(file+"*"):
        episode_files.append(episode_file)

    for episode in episode_files:
        if "_grp" in episode:
            label_file = open(episode,"r")
            this_label = label_file.read()
            this_label = this_label.strip()

            labels.append(this_label)

for ecg_index,label in enumerate(labels):
    result = re.search("\\[\\{(.*)\\}\\]", label)
    new_label = result.group(0)

    information_array = new_label.split("{")
    information_label.append(information_array)

    this_label_onsets = []
    this_label_offsets = []
    this_label_rhythm_name = []
    for item in information_array:
        match = re.search('"onset": (\d+)', item)
        if match:
            this_label_onsets.append(match.group(1))
        
        match = re.search('"offset": (\d+)',item)
        if match:
            this_label_offsets.append(match.group(1))
        
        match = re.search('"rhythm_name": "(.*)",',item)
        if match:
            this_label_rhythm_name.append(match.group(1))
    
    if len(this_label_onsets) == len(this_label_offsets):
        if len(this_label_offsets) == len(this_label_rhythm_name):
            for index,start_onset in enumerate(this_label_onsets):
                this_ecg_plot = ecg_plots[ecg_index]

                this_offset = this_label_offsets[index]

                ecg_label = this_label_rhythm_name[index]
                ecg_subsection = this_ecg_plot[int(start_onset):int(this_offset)]

                global_file_counter = write_to_file(ecg_subsection, ecg_label, global_file_counter)

recategorise_classes(destination_filename)

print("Complete!")