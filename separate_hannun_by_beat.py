import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from scipy import signal
from scipy.signal import resample
import glob,os

def normalize_signal(signal):
    original_signal = signal.copy()
    modified_signal = signal.copy()
    only_the_positives = signal.copy()

    for index,item in enumerate(modified_signal):
        if item < 0:
            modified_signal[index] = -1*item
            only_the_positives[index] = 0.0

    min_value = min(modified_signal)
    max_value = max(modified_signal)

    new_signal = [0]*len(modified_signal)
    for index,item in enumerate(modified_signal):
        normalized_item = (float(item) - min_value)/float(max_value - min_value)
        new_signal[index] = normalized_item
    
    for index,item in enumerate(original_signal):
        if item < 0:
            new_signal[index] = new_signal[index]*-1

    return new_signal

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def write_to_file(ecg_plot, rhythm_name, index):
    
    if not os.path.exists("./hannun_validation_data/"):
        os.makedirs("./hannun_validation_data/")

    f= open("./hannun_validation_data/ecg_"+str(index)+".txt","w")

    for value in ecg_plot:
        value_int = int(round(value))
        to_write = str(value_int) + " "

        f.write(to_write)
    f.write("\n"+rhythm_name)
    f.close()

def get_full_ecg(ecg_signal, r_peaks):
    max_length = 1300
    
    ecg_lead_1 = ecg_signal
    ecg_lead_2 = ecg_signal

    complete_beats = []
    beat_indexes = []

    for index,current_beat in enumerate(r_peaks):
        if index > 0 and index<len(r_peaks)-1:
            beat_before = r_peaks[index-1]
            beat_after = r_peaks[index+1]

            start_recording = ((current_beat-beat_before)//2) + beat_before
            end_recording = ((beat_after-current_beat)//2)+current_beat

            complete_beat_1 = ecg_lead_1[start_recording:end_recording]
            complete_beat_1 = np.pad(complete_beat_1, (0, max_length - len(complete_beat_1)), 'constant')

            complete_beat_2 = ecg_lead_2[start_recording:end_recording]
            complete_beat_2 = np.pad(complete_beat_2, (0, max_length - len(complete_beat_2)), 'constant')

            complete_beat = np.append(complete_beat_1, complete_beat_2)

            #print(complete_beat)
            beat_label = 'N'

            beat_indexes.append([start_recording,end_recording,beat_label])
            complete_beats.append([complete_beat,beat_label])
        elif index == 0 and len(r_peaks) == 1:
            complete_beat_1 = ecg_lead_1
            complete_beat_1 = np.pad(complete_beat_1, (0, max_length - len(complete_beat_1)), 'constant')
            complete_beat = np.append(complete_beat_1, complete_beat_1)

            #print(complete_beat)
            beat_label = 'N'

            complete_beats.append([complete_beat,beat_label])
        elif index == 0:
            beat_before = 0
            beat_after = r_peaks[index+1]

            start_recording = ((current_beat-beat_before)//2) + beat_before
            end_recording = ((beat_after-current_beat)//2)+current_beat

            complete_beat_1 = ecg_lead_1[start_recording:end_recording]
            complete_beat_1 = np.pad(complete_beat_1, (0, max_length - len(complete_beat_1)), 'constant')

            complete_beat_2 = ecg_lead_2[start_recording:end_recording]
            complete_beat_2 = np.pad(complete_beat_2, (0, max_length - len(complete_beat_2)), 'constant')

            complete_beat = np.append(complete_beat_1, complete_beat_2)

            #print(complete_beat)
            beat_label = 'N'

            beat_indexes.append([start_recording,end_recording,beat_label])
            complete_beats.append([complete_beat,beat_label])
    
    return complete_beats

def get_r_peaks(original_signal, plot_flag=0):

    original_signal = normalize_signal(original_signal)

    t = np.linspace(0,len(original_signal)-1,len(original_signal))

    best_fit_linear = np.poly1d(np.polyfit(t,original_signal,1))
    
    for index, item in enumerate(original_signal):
        original_signal[index] = original_signal[index] - best_fit_linear(index)

    #This doesn't work for some reason and I cannot for the life of me work out why
    #original_signal = original_signal - best_fit_linear(np.linspace(0,len(original_signal),len(original_signal)))

    original_signal = butter_highpass_filter(original_signal,0.5,500,5)

    signal_without_r = original_signal.copy()
    final_signal = original_signal.copy()

    sampled_signal = np.abs(original_signal)

    maximum = max(sampled_signal)
    cutoff = maximum/2

    for index,item in enumerate(sampled_signal):
        if item > cutoff:
            signal_without_r[index] = 0

    x = [0] * (len(signal_without_r)//4)
    y = [0] * (len(signal_without_r)//4)
    i = 0
    mean_i = 0

    while mean_i < len(x):
        x[mean_i] = i - 2
        y[mean_i] = np.mean(signal_without_r[i:i+4])
        i+=4
        mean_i += 1

    best_fit = np.poly1d(np.polyfit(x, y, 3))

    final_signal = final_signal - best_fit(np.linspace(0,len(final_signal)-1,len(final_signal)))

    signal_only_r = np.array(final_signal)    

    if max(final_signal) < abs(min(final_signal))/2:
        for index,item in enumerate(signal_only_r):
            signal_only_r[index] = item*-1

    maximum = max(signal_only_r)
    cutoff = maximum/5

    for index,item in enumerate(signal_only_r):
        if item <= cutoff:
            signal_only_r[index] = 0

    r_indices = []
    start_index = 0

    for index,item in enumerate(signal_only_r):
        if index > 0:
            if signal_only_r[index-1] == 0 and item != 0:
                #this means that we've started an R peak
                start_index = index
            elif item == 0 and start_index != 0:
                #we've reached the end of the R peak
                end_index = index - 1
                peak = ((end_index - start_index)//2)+start_index
                r_indices.append(peak)
                start_index = 0

    changed = 1
    while changed == 1:
        changed = 0
        for index, current in enumerate(r_indices):
            if index > 0:
                prev = r_indices[index-1]
                if (current - prev) < 20:
                    changed = 1
                    #r_indices.pop(index)
                    new_midpoint = ((current - prev)//2)+current
                    r_indices[index] = new_midpoint
                    r_indices.pop(index-1)

    if plot_flag == 1:
        plt.plot(final_signal)
        plt.plot(signal_only_r)
        plt.legend(["Original","Only R"])

    if plot_flag == 2:
        plt.plot(final_signal)

    print(r_indices)

    return r_indices


def split_beats():
    write_counter = 0
    for file in glob.glob("./hannun_nsr_test_data/*.ecg"):
        #file_string = "./split_processed_data/NSR/ecg_215.ecg"
        f = open(file, "r")
        
        ecg_old = np.fromfile(f, dtype=np.int16)
        ecg = resample(ecg_old, round(len(ecg_old)*1.8))
        #resample to get from 200Hz to 360Hz

        r_peaks = get_r_peaks(ecg, plot_flag=0)
        complete_beats = get_full_ecg(ecg, r_peaks)
        #plt.title(file)
        #plt.show()

        for sig_beat in complete_beats:
            signal, beat_label = sig_beat
            write_to_file(signal, beat_label, write_counter)
            write_counter += 1
    print(write_counter)

if __name__ == "__main__":
    split_beats()
    

