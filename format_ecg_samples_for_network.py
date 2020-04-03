"""
File which formats the ECG into training and testing datasets
"""

from ecg_feature_extraction import feature_extract_ecg
import random
import os
from glob import glob
import shutil
import numpy as np
from noise_reduction import noise_reduce
from numpy import fft

base_filename = "./network_data_unfiltered"

def read_ecg_data(filename):
    f = open(filename,"r")
    ecg_plot = np.fromfile(f, dtype=np.int16)

    ecg = ecg_plot.tolist()
    #print(ecg[0])

    #if list(ecg_plot).count(list(ecg_plot)[0]) == len(list(ecg_plot)):
    
    #if(len(set(ecg))==1):
    if len(ecg) == 0:
        print(filename)

    if ecg.count(ecg[0]) == len(ecg):
        print("All elements in list are same "+filename)

    return ecg_plot

#Function which sets up the training and validation folders in the current directory
def setup_files():
    if not os.path.exists(base_filename):
        os.makedirs(base_filename)
    if os.path.exists(base_filename+"/validation_set"):
        shutil.rmtree(base_filename+"/validation_set")
    os.makedirs(base_filename+"/validation_set")
    if os.path.exists(base_filename+"/training_set"):
        shutil.rmtree(base_filename+"/training_set")
    os.makedirs(base_filename+"/training_set")

#Function which writes the ECG signal and label to training and validation sets
def write_to_file(data, label, filename, validation_set):
    if filename in validation_set:
        f = open(base_filename+"/validation_set/"+filename+".txt","w")
    else:
        f = open(base_filename+"/training_set/"+filename+".txt","w")
    write_string = ""
    for value in data:
        write_string = write_string + str(value) + " "
    write_string = write_string + "\n"
    f.write(write_string)
    write_string = label + "\n"
    f.write(write_string)

    f.close()

os.chdir("./split_processed_data")
subfolders = [f.name for f in os.scandir('.') if f.is_dir() ] 

data_labels = []
ecg_plot = []
ecg_plot_lengths = []

#Get all the ECGs in each folder (not including network data - this has already been processed)
#Label is the foldername - associate this with the ECG
for folder in subfolders:
    if not folder.startswith("network_data"):
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                filename = str(os.path.join(root, name))
                ecg = read_ecg_data(filename)
                #filtered_ecg = noise_reduce(ecg,filename)
                
                ecg_plot.append(ecg)
                #ecg_plot.append(filtered_ecg)
                #fourier_transform = fft.fft(ecg)
                
                #ecg_plot.append(fourier_transform)
                data_labels.append(["",folder])

for plot in ecg_plot:
    ecg_plot_lengths.append(len(plot))

#Find the maximum length of the ECGs (just a check to make sure I'm using the right folder!)
max_length = max(ecg_plot_lengths)
for i,plot in enumerate(ecg_plot):
    new_plot = np.array(plot).tolist()
    while len(new_plot) < max_length:
        new_plot.append(0)
    ecg_plot[i] = new_plot

ecg_plot_lengths = []
for plot in ecg_plot:
    ecg_plot_lengths.append(len(plot))

ecg_plot_lengths = []
for plot in ecg_plot:
    ecg_plot_lengths.append(len(plot))

for i,plot in enumerate(ecg_plot):
    data_labels[i][0] = plot

#Print the max and min lengths - these should be the same
print("Max ECG length = "+str(max(ecg_plot_lengths)))
print("Min ECG length = "+str(min(ecg_plot_lengths)))

#Randomly split the dataset into training and validation set (25% validation currently set)
validation_set = random.sample(range(0,len(ecg_plot)), len(ecg_plot)//4)
for i,item in enumerate(validation_set):
    validation_set[i] = "ecg_"+str(item)

#Write processed data to file
setup_files()
for i,item in enumerate(data_labels):
    data = item[0]
    label = item[1]
    filename = "ecg_"+str(i)

    write_to_file(data, label, filename, validation_set)