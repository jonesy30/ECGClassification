
"""
File which extracts the ECG features returns the results
"""

import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from scipy import signal

#NOTE: Highpass/lowpass filtering has already been done in the pre_processing step
#Maybe this is in the wrong place but it's useful for everything so I'm happy with it for now

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_t_p_wave(ecg_plot, r_max_indexes):

    if len(r_max_indexes) < 3:
        return 0,0
    
    sub_plot = ecg_plot[r_max_indexes[1]:r_max_indexes[2]]

    if len(sub_plot) < 20:
        return 0,0

    lower_bound = (len(sub_plot)//10)
    upper_bound = len(sub_plot)-lower_bound

    sub_plot = sub_plot[lower_bound:upper_bound]
    
    sub_plot = butter_highpass_filter(sub_plot, 1.5, 200, 2)

    sub_plot = butter_lowpass_filter(sub_plot, 30, 200, 2)

    for i,item in enumerate(sub_plot):
        if item < 1:
            sub_plot[i] = 0
    
    isnonzero = np.concatenate(([0], (np.asarray(sub_plot) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    nonzero_ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    if len(nonzero_ranges) != 0:
        t_wave = nonzero_ranges[0]
    else:
        t_wave = [0,0]
    p_wave = [0,0]
    if len(nonzero_ranges) > 1:
        p_wave = nonzero_ranges[-1]

    if np.array_equal(t_wave,[0,0]):
        t_wave = t_wave
    else:
        for i,item in enumerate(t_wave):
            t_wave[i] = item + r_max_indexes[1] + lower_bound

    if np.array_equal(p_wave,[0,0]):
        p_wave = p_wave
    else:
        for i,item in enumerate(p_wave):
            p_wave[i] = item + r_max_indexes[1] + lower_bound

    t_wave_length = t_wave[1] - t_wave[0]
    p_wave_length = p_wave[1] - p_wave[0]

    return t_wave_length, p_wave_length

def feature_extract_ecg(file_string):
    
    f = open(file_string,"r")

    ecg_plot = np.fromfile(f, dtype=np.int16)
    if len(ecg_plot) < 50:
        return None
    #ecg_plot = ecg_plot[:256]

    #Five point filtering
    five_point_array = []
    div = 1.0/(8.0*256.0)   
    for i,element in enumerate(ecg_plot):
        y = 0
        y = y + (2.0 * ecg_plot[i])
        y = y + ecg_plot[i-1]
        y = y - ecg_plot[i-3]
        y = y - (2.0 * ecg_plot[i-4])

        y = y/8.0

        five_point_array.append(y)

    squared = []
    for element in five_point_array:
        squared_element = element ** 2.0
        squared.append(squared_element)

    average_squared = []
    for i,element in enumerate(squared):
        averaged_element = element
        if i > 2 and i < len(squared) - 2:
            averaged_element = squared[i-2] + squared[i-1] + element + squared[i+1] + squared[i+2]
            averaged_element = averaged_element / 5.0

        average_squared.append(averaged_element)

    N = 35
    moving_window_integration = []
    element_index = 0
    while element_index < len(ecg_plot):

        inner = 0
        for i in range(1,N+1):
            inner = inner + average_squared[(element_index - (N-i))]
        
        y = (1.0/N) * inner
        element_index = element_index + 1

        moving_window_integration.append(y)

    derivation = np.gradient(moving_window_integration)
    max = np.amax(derivation)
    zero_upper_bound = max / 10.0
    zero_lower_bound = zero_upper_bound * -1.0

    for index,item in enumerate(derivation):
        if item > 0 and item < zero_upper_bound:
            derivation[index] = 0
        #elif item > zero_lower_bound and item < 0:
        elif item < 0:
            derivation[index] = 0

    #Finding the QRS features
    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (np.asarray(derivation) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    qrs_starts = []
    qrs_ends = []

    qrs_length = []
    for qrs_block in ranges:
        qrs_starts.append(qrs_block[0])
        qrs_ends.append(qrs_block[1])
        length = qrs_block[1] - qrs_block[0]
        qrs_length.append(length)

    # print("Ranges = ")
    # print(str(ranges))
    # print("Lengths = ")
    # print(str(qrs_length))
    # print()

    #Finding RR features
    max = np.amax(average_squared)
    zero_upper_bound = max / 10.0
    zero_lower_bound = zero_upper_bound * -1.0

    average_squared_zeroed = average_squared.copy()
    for index,item in enumerate(average_squared):
        if item > 0 and item < zero_upper_bound:
            average_squared_zeroed[index] = 0
    
    for index,item in enumerate(average_squared_zeroed[1:-1]):
        if average_squared_zeroed[index-1] != 0 and average_squared_zeroed[index+1] !=0 and average_squared_zeroed[index] == 0:
            average_squared_zeroed[index] = average_squared_zeroed[index-1]
        
    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (np.asarray(average_squared_zeroed) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    r_maximums = []
    r_max_indexes = []
    for qrs_block in ranges:
        r_block = average_squared_zeroed[int(qrs_block[0]):int(qrs_block[1])]
        r_max_index = np.argmax(r_block)
        qrs_max = r_block[r_max_index]

        r_max_index = r_max_index + int(qrs_block[0])

        r_maximums.append(qrs_max)
        r_max_indexes.append(r_max_index)

    # print(str(r_maximums))
    # print(str(r_max_indexes))

    max_r = np.amax(r_maximums)
    min_r = np.amin(r_maximums)
    # print("Rmax = "+str(max_r))
    # print("Rmin = "+str(min_r))

    r_distances = []
    for i in range(0,len(r_max_indexes)-1):
        r_distance = r_max_indexes[i+1] - r_max_indexes[i]
        r_distances.append(r_distance)

    if len(r_distances) != 0:
        mean_rr = np.mean(r_distances)
        variance_rr = np.var(r_distances)
    else:
        mean_rr = 0.0
        variance_rr = 0.0

    # print("R distances = "+str(r_distances))
    # print("Mean RR = "+str(mean_rr))
    # print("RR Variance = "+str(variance_rr))

    if len(qrs_length) != 0:
        min_qrs_length = np.amin(qrs_length)
        max_qrs_length = np.amax(qrs_length)
        mean_qrs_length = np.mean(qrs_length)
        variance_qrs_length = np.var(qrs_length)
    else:
        min_qrs_length = 0.0
        max_qrs_length = 0.0
        mean_qrs_length = 0.0
        variance_qrs_length = 0.0
    
    t_wave_length, p_wave_length = get_t_p_wave(ecg_plot, r_max_indexes)

    if __name__ == "__main__":
        return ecg_plot, five_point_array, average_squared, moving_window_integration, derivation, r_max_indexes, qrs_starts, qrs_ends

    return max_r, min_r, mean_rr, variance_rr, max_qrs_length, min_qrs_length, mean_qrs_length, variance_qrs_length, t_wave_length, p_wave_length

    # print("QRS max = "+str(max_qrs_length))
    # print("QRS min = "+str(min_qrs_length))
    # print("Mean QRS length = "+str(mean_qrs_length))
    # print("Variance QRS length = "+str(variance_qrs_length))

if __name__ == "__main__":

    #Magic Numbers
    ECG_SELECTION = 168
    file_string = "processed_data/NSR/ecg_" + str(ECG_SELECTION) + ".ecg"
    [ecg_plot, five_point_array, average_squared, moving_window_integration, derivation, r_max_indexes, qrs_starts, qrs_ends] = feature_extract_ecg(file_string)

    plt.plot(ecg_plot,'r',label="unfiltered ECG")
    plt.title("ECG Signal")
    plt.plot(five_point_array,'g--',label="five point derivation")
    for max in r_max_indexes:
        plt.axvline(x=max,label="Max "+str(max))

    for qrs_start in qrs_starts:
        plt.plot(qrs_start,0,'r+')
    
    for qrs_end in qrs_ends:
        plt.plot(qrs_end,0,'g+')

    plt.legend(loc='upper left')
    plt.figure()
    plt.plot(average_squared,'r',label="squared five point derivation")
    plt.plot(moving_window_integration,'g',label="moving window integration")
    plt.plot(derivation,label="derivation")
    plt.title("Squared Five Point Analysis")
    plt.legend(loc='upper left')
    plt.show()