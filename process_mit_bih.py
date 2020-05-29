import wfdb
import matplotlib.pyplot as plt
import glob, os
import numpy as np
from scipy import signal
from scipy.signal import resample

beat_replacement = {'/':'P','J':'unused','S':'unused'} #since you can't have / as a foldername!
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

leave_one_out_validation = 1
downsample_to_match_incart = 0

def write_to_file(ecg_plot, rhythm_name, index, validation_flag = 0, base_filename="./mit_bih_processed_data_two_leads_leave_out_validation/", r_index=None):
    
    if validation_flag == 1:
        if not os.path.exists(base_filename+"/network_validation/"+rhythm_name):
            os.makedirs(base_filename+"/network_validation/"+rhythm_name)

        f= open(base_filename+"/network_validation/"+rhythm_name+"/ecg_"+str(index)+".txt","w")

    else:
        if not os.path.exists(base_filename+rhythm_name):
            os.makedirs(base_filename+rhythm_name)

        f= open(base_filename+rhythm_name+"/ecg_"+str(index)+".txt","w")

    for value in ecg_plot:
        value_int = int(round(value*1000))
        to_write = str(value_int) + " "

        f.write(to_write)
    if r_index != None:
        f.write(str(r_index))
    f.close()

def get_full_ecg(filenumber, base_filename="mit_bih/"):
    max_length = 430
    
    annotation = wfdb.rdann(base_filename+str(filenumber), 'atr', sampto=650000)
    ecg_signal, _ = wfdb.srdsamp(base_filename+str(filenumber))

    ecg_lead_1 = ecg_signal[:,0]
    ecg_lead_2 = ecg_signal[:,1]

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

            complete_beat_1 = ecg_lead_1[start_recording:end_recording]
            complete_beat_2 = ecg_lead_2[start_recording:end_recording]

            if "st_petersburg" in base_filename and downsample_to_match_incart != 1:
                complete_beat_1 = signal.resample(complete_beat_1,int(round(len(complete_beat_1)*1.401)))
                complete_beat_2 = signal.resample(complete_beat_2,int(round(len(complete_beat_2)*1.401)))
            elif downsample_to_match_incart == 1:
                complete_beat_1 = signal.resample(complete_beat_1,257)
                complete_beat_2 = signal.resample(complete_beat_2,257)

            if len(complete_beat_1) <= max_length:
                complete_beat_1 = np.pad(complete_beat_1, (0, max_length - len(complete_beat_1)), 'constant')
                complete_beat_2 = np.pad(complete_beat_2, (0, max_length - len(complete_beat_2)), 'constant')

                complete_beat = np.append(complete_beat_1, complete_beat_2)

                #print(complete_beat)
                beat_label = annotation_classes[index]

                #beat_indexes.append([start_recording,end_recording,beat_label])
                this_index = current_beat - start_recording
                
                beat_indexes.append(this_index)
                complete_beats.append([complete_beat,beat_label])
    
    return complete_beats, beat_indexes

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

        complete_beats,beat_indexes = get_full_ecg(filenumber, base_filename=base_filename)

        for index,sig_beat in enumerate(complete_beats):
            signal, beat_label = sig_beat
            r_index = beat_indexes[index]
            if len(signal) > current_max:
                #max_length = len(signal)
                current_max = len(signal)
                biggest_file = write_counter
                biggest_file_class = beat_label
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
                    #destination_filename = "external_validation_data/st_petersburg/"
                    #write_to_file(signal, beat_label, write_counter, base_filename=destination_filename)
                    write_to_file(signal, beat_label, write_counter)
                    #write_to_file(signal, beat_label, write_counter, r_index=r_index)
                write_counter += 1
        print("Max = "+str(current_max))
        print("Max file = "+str(biggest_file))
        print("Max file class = "+str(biggest_file_class))
        print("All lengths = "+str(total_lengths))

    print("Complete!")
    print("Files Written = "+str(write_counter))

if __name__ == "__main__":
    process_files()
    #process_files(base_filename="external_original/st_petersburg/files/")
    #plot_ecg(232)

    # for f in glob.glob("./mit_bih_two_second_samples/P/*.txt"):
    # f = "./mit_bih_processed_data_two_leads_downsampled_incart/N/ecg_3.txt"
    # file = open(f,"r")
    # print(f)

    # ecg_plot = file.read()
    # ecg_plot = ecg_plot.strip()
    # ecg = ecg_plot.split(" ")
    # ecg = [int(n) for n in ecg]

    # print(len(ecg))

    # lead_one = ecg[:720]
    # lead_two = ecg[720:]

    # plt.subplot(211)

    # plt.plot(lead_one)
    # plt.title("Lead One")

    # plt.subplot(212)

    # plt.plot(lead_two)
    # plt.title("Lead Two")

    # plt.show()

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