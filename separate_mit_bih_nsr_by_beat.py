import wfdb
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import os, shutil

beat_replacement = {'/':'P'} #since you can't have / as a foldername!
beat_dict = {'N':'Normal','L':'LBBB','R':'RBBB','A':'APB','a':'AAPB','J':'JUNCTIONAL','S':'SP','V':'VT','r':'RonT','e':'Aesc','j':'Jesc','n':'SPesc','E':'Vesc','/':'Paced'}

# N		Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
# L		Left bundle branch block beat
# R		Right bundle branch block beat
# B		Bundle branch block beat (unspecified)
# A		Atrial premature beat
# a		Aberrated atrial premature beat
# J		Nodal (junctional) premature beat
# S		Supraventricular premature or ectopic beat (atrial or nodal)
# V		Premature ventricular contraction
# r		R-on-T premature ventricular contraction
# F		Fusion of ventricular and normal beat (not included)
# e		Atrial escape beat
# j		Nodal (junctional) escape beat
# n		Supraventricular escape beat (atrial or nodal)
# E		Ventricular escape beat
# /		Paced beat
# f		Fusion of paced and normal beat (not included)

def setup_files(destination_folder="./external_validation_data"):
    if os.path.exists(destination_folder+"/mit_bih_nsr"):
        shutil.rmtree(destination_folder+"/mit_bih_nsr")
    os.makedirs(destination_folder+"/mit_bih_nsr")

def write_to_file(data, label, filename, destination_folder="./external_validation_data/mit_bih_nsr/"):
    
    f = open(destination_folder+"ecg_"+str(filename)+".txt","w")
    write_string = ""
    for value in data:
        new_value = int(round(value*1000))
        write_string = write_string + str(new_value) + " "
    write_string = write_string + "\n"
    f.write(write_string)
    write_string = label + "\n"
    f.write(write_string)

    f.close()

def get_full_ecg(filenumber, base_filename="external_original/mit_bih_normal/"):
    max_length = 1300
    
    annotation = wfdb.rdann(base_filename+str(filenumber), 'atr', sampto=650000)
    ecg_signal, _ = wfdb.srdsamp(base_filename+str(filenumber))

    ecg_lead_1 = ecg_signal[:,0]
    ecg_lead_2 = ecg_signal[:,1]

    unique_labels_df = annotation.get_contained_labels(inplace=False)
    annotation_indices = annotation.sample
    annotation_classes = annotation.symbol
    sampling_frequency = annotation.fs

    for index,beat_class in enumerate(annotation_classes):
        if str(beat_class) not in beat_dict:
            np.delete(annotation_indices, index)
            np.delete(annotation_classes, index)

    complete_beats = []
    beat_indexes = []

    for index,current_beat in enumerate(annotation_indices):
        if index > 0 and index<len(annotation_indices)-1:
            beat_before = annotation_indices[index-1]
            beat_after = annotation_indices[index+1]

            start_recording = ((current_beat-beat_before)//2) + beat_before
            end_recording = ((beat_after-current_beat)//2)+current_beat

            complete_beat_1 = ecg_lead_1[start_recording:end_recording]
            complete_beat_1 = np.pad(complete_beat_1, (0, max_length - len(complete_beat_1)), 'constant')

            complete_beat_2 = ecg_lead_2[start_recording:end_recording]
            complete_beat_2 = np.pad(complete_beat_2, (0, max_length - len(complete_beat_2)), 'constant')

            complete_beat = np.append(complete_beat_1, complete_beat_2)

            #print(complete_beat)
            beat_label = annotation_classes[index]

            beat_indexes.append([start_recording,end_recording,beat_label])
            complete_beats.append([complete_beat,beat_label])
    
    return complete_beats

def process_files(base_filename="external_original/mit_bih_normal"):

    setup_files()

    ecg_files = []
    write_counter = 0
    max_length = 1300

    for file in glob.glob(base_filename+"/*.atr"):
        split, _ = file.split(".")
        _, name = split.split("\\")
        filenumber = name.split("-")

        if len(filenumber) > 1:
            filenumber, _ = filenumber
        else:
            filenumber = filenumber[0]

        ecg_files.append(filenumber)
        print(filenumber)

        complete_beats = get_full_ecg(filenumber)

        for sig_beat in complete_beats:
            signal, beat_label = sig_beat
            write_to_file(signal, beat_label, write_counter)
            write_counter += 1

    print("Complete!")
    print("Files Written = "+str(write_counter))

if __name__ == "__main__":
    process_files()