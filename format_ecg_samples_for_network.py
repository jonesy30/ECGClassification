from ecg_feature_extraction import feature_extract_ecg
import random
import os
from glob import glob
import shutil
import numpy as np

def read_ecg_data(filename):
    f = open(filename,"r")
    ecg_plot = np.fromfile(f, dtype=np.int16)
    return ecg_plot

def setup_files():
    if not os.path.exists("./network_data"):
        os.makedirs("./network_data")
    if os.path.exists("./network_data/validation_set"):
        shutil.rmtree("./network_data/validation_set")
    os.makedirs("./network_data/validation_set")
    if os.path.exists("./network_data/training_set"):
        shutil.rmtree("./network_data/training_set")
    os.makedirs("./network_data/training_set")

def write_to_file(data, label, filename, validation_set):
    if filename in validation_set:
        f = open("./network_data/validation_set/"+filename+".txt","w")
    else:
        f = open("./network_data/training_set/"+filename+".txt","w")
    write_string = ""
    for value in data:
        write_string = write_string + str(value) + " "
    write_string = write_string + "\n"
    f.write(write_string)
    write_string = label + "\n"
    f.write(write_string)

    f.close()

os.chdir("./processed_data")
subfolders = [f.name for f in os.scandir('.') if f.is_dir() ] 

data_labels = []
ecg_plot = []
ecg_plot_lengths = []
for folder in subfolders:
    if folder != "network_data" and folder != "network_data":
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                filename = str(os.path.join(root, name))
                ecg_plot.append(read_ecg_data(filename))
                data_labels.append(["",folder])

for plot in ecg_plot:
    ecg_plot_lengths.append(len(plot))

max_length = max(ecg_plot_lengths)
for i,plot in enumerate(ecg_plot):
    new_plot = np.array(plot).tolist()
    while len(new_plot) < max_length:
        new_plot.append(0)
    ecg_plot[i] = new_plot

ecg_plot_lengths = []
for plot in ecg_plot:
    ecg_plot_lengths.append(len(plot))

print("Min length = "+str(min(ecg_plot_lengths)))

#max_length = max(ecg_plot_lengths)
#max_length = 500
#for i,plot in enumerate(ecg_plot):
    # new_plot = plot
    # if len(new_plot) > max_length:
    #     j = 0
    #     created_arrays = []
    #     label = data_labels[i]

    #     while j < len(new_plot):
    #         n = 0
    #         new_array = [] 
    #         while n < max_length:
    #             if j + n < len(new_plot):
    #                 new_array.append(plot[j+n])
    #             else:
    #                 new_array.append(0)
    #             n = n + 1
    #         created_arrays.append(new_array)
    #         j = j + 1
        
    #     for created in created_arrays:
    #         ecg_plot.append(created)
    #         data_labels.append(label)

    # while len(new_plot) < max_length:
    #     new_plot = np.append(new_plot,0)
    
    # ecg_plot[i] = new_plot

ecg_plot_lengths = []
for plot in ecg_plot:
    ecg_plot_lengths.append(len(plot))

for i,plot in enumerate(ecg_plot):
    data_labels[i][0] = plot

print("Max ECG length = "+str(max(ecg_plot_lengths)))
print("Min ECG length = "+str(min(ecg_plot_lengths)))

validation_set = random.sample(range(0,len(ecg_plot)), len(ecg_plot)//10)
#validation_set = random.sample(range(0, 819), 205)
for i,item in enumerate(validation_set):
    validation_set[i] = "ecg_"+str(item)

setup_files()
for i,item in enumerate(data_labels):
    data = item[0]
    label = item[1]
    filename = "ecg_"+str(i)

    write_to_file(data, label, filename, validation_set)