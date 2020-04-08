import wfdb
import matplotlib.pyplot as plt
import glob, os
import numpy as np

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

def write_to_file(ecg_plot, rhythm_name, index):
    if not os.path.exists("./mit_bih_processed_data/"+rhythm_name):
        os.makedirs("./mit_bih_processed_data/"+rhythm_name)

    f= open("./mit_bih_processed_data/"+rhythm_name+"/ecg_"+str(index)+".txt","w")

    for value in ecg_plot:
        value_int = int(round(value*1000))
        to_write = str(value_int) + " "

        f.write(to_write)
    f.close()

def get_full_ecg(filenumber):
    annotation = wfdb.rdann('mit_bih/'+str(filenumber), 'atr', sampto=650000)
    ecg_signal, _ = wfdb.srdsamp('mit_bih/'+str(filenumber))

    ecg_signal = ecg_signal[:,0]

    unique_labels_df = annotation.get_contained_labels(inplace=False)
    annotation_indices = annotation.sample
    annotation_classes = annotation.symbol
    sampling_frequency = annotation.fs

    for index,beat_class in enumerate(annotation_classes):
        if beat_class not in beat_dict:
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

            complete_beat = ecg_signal[start_recording:end_recording]
            #print(complete_beat)
            beat_label = annotation_classes[index]

            beat_indexes.append([start_recording,end_recording,beat_label])
            complete_beats.append([complete_beat,beat_label])
    
    return complete_beats

def plot_ecg(filenumber):

    record = wfdb.rdsamp('mit_bih/'+str(filenumber), sampto=3000, smoothframes=True)
    annotation = wfdb.rdann('mit_bih/'+str(filenumber), 'atr', sampto=3000)
    #Annotation attributes: record_name, extension, sample (indices), symbol (classes), subtype, chan, num, aux_note, fs, label_store, description, custom_labels, contained_labels

    wfdb.plotrec(record, annotation = annotation, title="Record "+str(filenumber)+" from MIT-BIH Arrhythmia Database", figsize = (10,4), ecggrids = 'all',plotannsym=True)

def process_files():

    ecg_files = []
    write_counter = 0
    max_length = 1300
    total_lengths = set()

    for file in glob.glob("mit_bih/*.atr"):
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
            if len(signal) > max_length:
                max_length = len(signal)
            if beat_label in beat_dict:
                if beat_label in beat_replacement:
                    beat_label = beat_replacement.get(beat_label)

                signal = np.pad(signal, (0, max_length - len(signal)), 'constant')
                total_lengths.add(len(signal))

                write_to_file(signal, beat_label, write_counter)
                write_counter += 1
        print("Max = "+str(max_length))
        print("All lengths = "+str(total_lengths))

    print("Complete!")
    print("Files Written = "+str(write_counter))

if __name__ == "__main__":
    process_files()
    #plot_ecg(232)

    # f = "./mit_bih_processed_data/P/ecg_4134.txt"

    # file = open(f, "r")
    # ecg_string = file.read()
    # ecg_string = ecg_string.strip()
    # ecg = ecg_string.split(" ")

    # ecg = [int(n) for n in ecg]

    # print(len(ecg))

    # plt.plot(ecg)
    # plt.show()