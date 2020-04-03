"""
Format the feature extracted data into filenames
"""

from ecg_feature_extraction import feature_extract_ecg
import random
import os
from glob import glob
import shutil

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
    for value in data:
        write_string = str(value) + "\n"
        f.write(write_string)
    write_string = label + "\n"
    f.write(write_string)

    f.close()

os.chdir("./split_processed_data")
subfolders = [f.name for f in os.scandir('.') if f.is_dir() ] 

data_labels = []
for folder in subfolders:
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            filename = str(os.path.join(root, name))
            data = feature_extract_ecg(filename)
            if data != None:
                data_labels.append([data,folder,os.path.splitext(name)[0]])

validation_set = random.sample(range(0, 819), 205)
for i,item in enumerate(validation_set):
    validation_set[i] = "ecg_"+str(item)

setup_files()
for item in data_labels:
    data = item[0]
    label = item[1]
    filename = item[2]

    write_to_file(data, label, filename, validation_set)