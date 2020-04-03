"""
Function which pre-processes the raw ECG signal into segments with only one class label
"""

import glob, os
import numpy as np
import re
import scipy
import sys  
#from scipy import signal
#from scipy import pi
#from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np    
#from scipy.signal import butter, lfilter, freqz 

def write_to_file(ecg_plot, rhythm_name, index):
    if not os.path.exists("./processed_data/"+rhythm_name):
        os.makedirs("./processed_data/"+rhythm_name)

    f= open("./processed_data/"+rhythm_name+"/ecg_"+str(index)+".ecg","wb")

    cutoff = 100
    order = 6
    fs = 100000.0
    ecg_plot_filtered = ecg_plot

    for value in ecg_plot_filtered:
        value_int = int(round(value))
        to_write = str(value_int) + " "
        bin = np.int16(to_write)

        f.write(bin)
    f.close()

ecg_files_found = []
ecg_plots = []
labels = []
information_label = []

onsets = []
offsets = []
rhythm_name = []

global_file_written_counter = 0

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
    
    # print()
    # print(str(this_label_onsets))
    # print(str(this_label_offsets))
    # print(str(this_label_rhythm_name))

    if len(this_label_onsets) == len(this_label_offsets):
        if len(this_label_offsets) == len(this_label_rhythm_name):
            for index,start_onset in enumerate(this_label_onsets):
                this_ecg_plot = ecg_plots[ecg_index]

                this_offset = this_label_offsets[index]

                ecg_label = this_label_rhythm_name[index]
                ecg_subsection = this_ecg_plot[int(start_onset):int(this_offset)]

                write_to_file(ecg_subsection, ecg_label, global_file_written_counter)
                global_file_written_counter = global_file_written_counter+1
            

print(ecg_plots[0])
print(labels[0])

print()
print(str(len(ecg_plots)))
print(str(len(labels)))

print("Hello!!")
print(information_label[0])