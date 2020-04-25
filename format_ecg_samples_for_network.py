"""
File which formats the ECG into training and testing datasets
"""

from ecg_feature_extraction import feature_extract_ecg
import random
import os
import shutil
import numpy as np
from noise_reduction import noise_reduce
from numpy import fft

#destination_folder = "./network_data_unfiltered"
#base_folder = "./split_processed_data"
destination_folder = "./network_data"
base_folder = "./mit_bih_processed_data_two_leads_subset"
binary_file = 0
leave_one_out_validation = 1

def read_ecg_data(filename, binary_file=1):
    f = open(filename,"r")
    
    if binary_file == 1:
        ecg_plot = np.fromfile(f, dtype=np.int16)
        ecg = ecg_plot.tolist()
    else:
        ecg_plot = f.read()
        ecg_plot = ecg_plot.strip()
        ecg = ecg_plot.split(" ")
        ecg = [int(n) for n in ecg]
    #print(ecg[0])

    #if list(ecg_plot).count(list(ecg_plot)[0]) == len(list(ecg_plot)):
    
    #if(len(set(ecg))==1):
    if len(ecg) == 0:
        print(filename)

    if ecg.count(ecg[0]) == len(ecg):
        print("All elements in list are same "+filename)

    return ecg

#Function which sets up the training and validation folders in the current directory
def setup_files():
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if os.path.exists(destination_folder+"/validation_set"):
        shutil.rmtree(destination_folder+"/validation_set")
    os.makedirs(destination_folder+"/validation_set")
    if os.path.exists(destination_folder+"/training_set"):
        shutil.rmtree(destination_folder+"/training_set")
    os.makedirs(destination_folder+"/training_set")

#Function which writes the ECG signal and label to training and validation sets
def write_to_file(data, label, filename, validation_set):
    if filename in validation_set:
        f = open(destination_folder+"/validation_set/"+filename+".txt","w")
    else:
        f = open(destination_folder+"/training_set/"+filename+".txt","w")
    write_string = ""
    for value in data:
        write_string = write_string + str(value) + " "
    write_string = write_string + "\n"
    f.write(write_string)
    write_string = label + "\n"
    f.write(write_string)

    f.close()

def write_validation_to_file(data, label, filename):
    f = open(destination_folder+"/validation_set/"+filename+".txt","w")
    
    write_string = ""
    for value in data:
        write_string = write_string + str(value) + " "
    write_string = write_string + "\n"
    f.write(write_string)
    write_string = label + "\n"
    f.write(write_string)

    f.close()

def split_samples(base_folder):
    print(os.getcwd())
    os.chdir(base_folder)
    subfolders = [f.name for f in os.scandir('.') if f.is_dir() ] 

    data_labels = []
    ecg_plot = []
    ecg_plot_lengths = []

    #Get all the ECGs in each folder (not including network data - this has already been processed)
    #Label is the foldername - associate this with the ECG
    for folder in subfolders:
        if not folder.startswith("network_"):
            print("Folder "+str(folder))
            for root, dirs, files in os.walk(folder, topdown=False):
                for name in files:
                    filename = str(os.path.join(root, name))
                    ecg = read_ecg_data(filename,binary_file=binary_file)
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

    return ecg_plot, data_labels

ecg_plot, data_labels = split_samples(base_folder)

#Randomly split the dataset into training and validation set (25% validation currently set)
if leave_one_out_validation == 0:
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

else:
    validation_set = {}

    #Write processed data to file
    setup_files()
    for i,item in enumerate(data_labels):
        data = item[0]
        label = item[1]
        filename = "ecg_"+str(i)

        write_to_file(data, label, filename, validation_set)
    
    ecg_plot, data_labels = split_samples("./network_validation/")
    os.chdir("..")
    
    #Write processed data to file
    for i,item in enumerate(data_labels):
        data = item[0]
        label = item[1]
        filename = "ecg_"+str(i)

        write_validation_to_file(data, label, filename)