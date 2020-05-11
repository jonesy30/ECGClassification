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

leave_one_out_validation = 0

def write_to_file(ecg_plot, rhythm_name, index, validation_flag = 0, base_filename="./mit_bih_processed_data_two_leads/"):
    
    if validation_flag == 1:
        if not os.path.exists(base_filename+"/network_validation/"+rhythm_name):
            os.makedirs(base_filename+"/network_validation/"+rhythm_name)

        f= open(base_filename+"/network_validation/"+rhythm_name+"/ecg_"+str(index)+".txt","w")
        print("Validation set - "+rhythm_name)

    else:
        if not os.path.exists(base_filename+rhythm_name):
            os.makedirs(base_filename+rhythm_name)

        f= open(base_filename+rhythm_name+"/ecg_"+str(index)+".txt","w")

    for value in ecg_plot:
        value_int = int(round(value*1000))
        to_write = str(value_int) + " "

        f.write(to_write)
    f.close()

def get_full_ecg(filenumber, base_filename="mit_bih/"):

    seconds_per_sample = 2
    samples_per_block = seconds_per_sample * 360 #360Hz sampling rate

    annotation = wfdb.rdann(base_filename+str(filenumber), 'atr', sampto=650000)
    ecg_signal, _ = wfdb.srdsamp(base_filename+str(filenumber))

    ecg_lead_1 = ecg_signal[:,0]
    ecg_lead_2 = ecg_signal[:,1]

    annotation_indices = annotation.sample
    annotation_classes = annotation.symbol
    sampling_frequency = annotation.fs

    for index,beat_class in enumerate(annotation_classes):
        if beat_class not in beat_dict:
            np.delete(annotation_indices, index)
            np.delete(annotation_classes, index)

    complete_beats = []

    for index in range(len(ecg_lead_1)//samples_per_block):
        start_index = index * samples_per_block
        end_index = (index+1) * samples_per_block

        complete_beat_1 = ecg_lead_1[start_index:end_index]
        complete_beat_2 = ecg_lead_2[start_index:end_index]

        complete_beat = np.append(complete_beat_1, complete_beat_2)

        in_range_annotation_indexes = []
        for meta_index,annotation_index in enumerate(annotation_indices):
            if annotation_index > end_index:
                break
            if annotation_index > start_index:
                in_range_annotation_indexes.append(meta_index)

        in_range_annotations = set([annotation_classes[i] for i in in_range_annotation_indexes])
        if len(in_range_annotations) > 0:
            if "N" in in_range_annotations:
                in_range_annotations.remove("N")
            if len(in_range_annotations) == 0:
                complete_beats.append([complete_beat,"N"])
            elif len(in_range_annotations) == 1:
                beat_label = in_range_annotations.pop()
                if beat_label in beat_dict:
                    complete_beats.append([complete_beat,beat_label])
    
    return complete_beats

def plot_ecg(filenumber, base_filename="mit_bih/", title="MIT-BIH Arrhythmia Database"):

    record = wfdb.rdsamp(base_filename+str(filenumber), sampto=3000, smoothframes=True)
    annotation = wfdb.rdann(base_filename+str(filenumber), 'atr', sampto=3000)
    #Annotation attributes: record_name, extension, sample (indices), symbol (classes), subtype, chan, num, aux_note, fs, label_store, description, custom_labels, contained_labels

    wfdb.plotrec(record, annotation = annotation, title="Record "+str(filenumber)+title, figsize = (10,4), ecggrids = 'all',plotannsym=True)

def process_files(base_filename="mit_bih/"):

    ecg_files = []
    write_counter = 0
    current_max = 0
    biggest_file = 0
    biggest_file_class = ""
    total_lengths = set()

    for file in glob.glob(base_filename+"*.atr"):
        split, _ = file.split(".")
        _, name = split.split("\\")
        filenumber = name.split("-")

        if len(filenumber) > 1:
            filenumber, _ = filenumber
        else:
            filenumber = filenumber[0]

        ecg_files.append(filenumber)
        print(filenumber)

        complete_beats = get_full_ecg(filenumber, base_filename=base_filename)

        for sig_beat in complete_beats:
            signal, beat_label = sig_beat

            if beat_label in beat_dict:
                if beat_label in beat_replacement:
                    beat_label = beat_replacement.get(beat_label)

                #signal = np.pad(signal, (0, max_length - len(signal)), 'constant')
                total_lengths.add(len(signal))

                leave_out_array = ["104","208","113","210","119"]
                #leave_out_array = ["124","200","115","207","228","119","113","210","104","208"]
                #leave_out_array = ["124","200","115","207","228"]
                if (leave_one_out_validation == 1) and (any(x in file for x in leave_out_array)):
                    write_to_file(signal, beat_label, write_counter, validation_flag=1)
                else:
                    destination_filename = "mit_bih_two_second_samples/"
                    write_to_file(signal, beat_label, write_counter, base_filename=destination_filename)
                write_counter += 1
        print("Max = "+str(current_max))
        print("Max file = "+str(biggest_file))
        print("Max file class = "+str(biggest_file_class))
        print("All lengths = "+str(total_lengths))

    print("Complete!")
    print("Files Written = "+str(write_counter))

if __name__ == "__main__":
    
    # complete_beats = get_full_ecg(100)
    
    # print(complete_beats[0][:5])
    
    # ecgs = []
    # labels = []
    # for sig_beat in complete_beats:
    #     signal, beat_label = sig_beat
    #     ecgs.append(signal)
    #     labels.append(beat_label)

    # print(len(ecgs))
    # print(set(labels))

    process_files()
    
    
    #process_files(base_filename="external_original/st_petersburg/files/")
    #plot_ecg(232)

    #plot_ecg("I01",base_filename="./external_original/st_petersburg/files/",title="St Petersburg INCART Dataset")

    # for f in glob.glob("./external_validation_data/mit_bih_nsr_subset/cnn/network_incorrect_results/N/j/*.txt"):
    #     #f = "./hannun_validation_data/ecg_63.txt"
    #     # f = "./mit_bih_processed_data/N/ecg_82473.txt"

    #     file = open(f, "r")
    #     ecg_string = file.read()
    #     ecg_string = ecg_string.replace("\nN",'')
    #     ecg_string = ecg_string.strip()
    #     ecg = ecg_string.split(" ")

    #     print(len(ecg))

    #     ecg = [int(n) for n in ecg]
    #     ecg_1 = ecg[:1300]
    #     ecg_2 = ecg[1300:]

    #     fig, axs = plt.subplots(2)
    #     fig.suptitle(str(os.path.basename(f)))
    #     axs[0].plot(ecg_1)
    #     axs[1].plot(ecg_2)

    #     plt.xlabel("Samples (at 360 Hz)")
    #     #plt.title(f)

    #     for ax in axs.flat:
    #         ax.set(ylabel='Microvolts')

    #     # ax = plt.axes()

    #     # axlabels = np.arange(0,1,0.124)
    #     # axlabels = [round(x,1) for x in axlabels]
    #     # ax.set_xticklabels(axlabels)

    #     #plt.plot(ecg)
    #     plt.show()