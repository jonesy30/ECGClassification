
"""
File which extracts the ECG features returns the results
"""

import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from scipy import signal
import glob

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

def get_little_wave(sub_plot, p_wave = 0):
    for i,item in enumerate(sub_plot):
        if item < 0:
            sub_plot[i] = 0
    
    isnonzero = np.concatenate(([0], (np.asarray(sub_plot) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    nonzero_ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    if len(nonzero_ranges) > 1:
        if p_wave == 0:
            wave = nonzero_ranges[-2]
        else:
            wave = nonzero_ranges[-1]
    else:
        wave = [0,0]

    return wave

def get_t_p_wave(ecg_plot, r_max_indexes):

    # if len(r_max_indexes) < 3:
    #     print(r_max_indexes)
    #     sub_plot = ecg_plot[:r_max_indexes[0]]
    # else:
    #     sub_plot = ecg_plot[r_max_indexes[1]:r_max_indexes[2]]

    #lower_bound = (len(sub_plot)//10)
    #upper_bound = len(sub_plot)-lower_bound

    #sub_plot = sub_plot[lower_bound:upper_bound]
    
    ecg_plot = butter_highpass_filter(ecg_plot, 1.5, 200, 2)
    ecg_plot = butter_lowpass_filter(ecg_plot, 10, 200, 2)

    post_filter_max = np.argmax(ecg_plot)

    #plt.plot(ecg_plot)
    
    #plt.figure()

    print("P wave")
    p_wave = get_little_wave(ecg_plot[:post_filter_max],0)
    print("T wave")
    #plt.plot(ecg_plot)
    #plt.figure()
    t_wave = get_little_wave(ecg_plot[post_filter_max:],1)

    print("P wave = "+str(p_wave))
    print("T wave = "+str(t_wave))
    if np.array_equal(t_wave,[0,0]):
        t_wave = t_wave
    else:
        for i,item in enumerate(t_wave):
            t_wave[i] = item + post_filter_max

    t_wave_length = t_wave[1] - t_wave[0]
    p_wave_length = p_wave[1] - p_wave[0]

    return p_wave[0],p_wave[1],t_wave[0],t_wave[1],p_wave_length,t_wave_length

def read_file(file_string):
    f = open(file_string, "r")
    found_data = []
    for i,line in enumerate(f):
        line = line.replace("\n","")
        #ECG signal stored in first line separated by spaces
        if i < 1:
            line_segments = line.split()
            line_segments = [float(x) for x in line_segments]

            for item in line_segments:
                found_data.append(item)
    f.close()

    found_data_lead_1 = found_data[:430]
    found_data_lead_2 = found_data[430:]

    found_data_lead_1 = np.trim_zeros(found_data_lead_1)
    found_data_lead_2 = np.trim_zeros(found_data_lead_2)

    return found_data_lead_1,found_data_lead_2

def feature_extract_ecg(ecg_plot):
    
    r_max = np.argmax(ecg_plot)
    r_max_indexes = [r_max]

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
    zero_upper_bound = max / 15.0
    zero_lower_bound = zero_upper_bound * -1.0

    for index,item in enumerate(derivation):
        if item > 0 and item < zero_upper_bound:
            derivation[index] = 0
        #elif item > zero_lower_bound and item < 0:
        elif item < 0:
            derivation[index] = 0
    
    #derivation = np.gradient(derivation)

    before_r = derivation[:r_max]
    after_r = derivation[r_max:]

    q_point = 0
    for index,i in enumerate(reversed(before_r)):
        if round(i) == 0:
            q_point = len(before_r) - index
            break

    s_point = 0
    for index,i in enumerate(after_r):
        if round(i) == 0:
            s_point = index + r_max
            break
    
    #Finding the QRS features
    #Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    # isnonzero_q = np.concatenate(([0], (np.asarray(before_r) != 0).view(np.int8), [0]))
    # print(isnonzero_q)
    
    # absdiff_q = np.abs(np.diff(isnonzero_q))
    # print(absdiff_q)
    # # Runs start and end where absdiff is 1.
    # ranges_q = np.where(absdiff_q == 1)[0].reshape(-1, 2)

    # isnonzero_s = np.concatenate(([0], (np.asarray(after_r) != 0).view(np.int8), [0]))
    # absdiff_s = np.abs(np.diff(isnonzero_s))
    # # Runs start and end where absdiff is 1.
    # ranges_s = np.where(absdiff_s == 1)[0].reshape(-1, 2)

    # print("Q = "+str(ranges_q))
    # print("S = "+str(ranges_s))

    # qrs_starts = []
    # qrs_ends = []

    # qrs_length = []
    # for qrs_block in ranges:
    #     qrs_starts.append(qrs_block[0])
    #     qrs_ends.append(qrs_block[1])
    #     length = qrs_block[1] - qrs_block[0]
    #     qrs_length.append(length)

    # print("Ranges = ")
    # print(str(ranges))
    # print("Lengths = ")
    # print(str(qrs_length))
    # print()

    # #Finding RR features
    # max = np.amax(average_squared)
    # zero_upper_bound = max / 10.0
    # zero_lower_bound = zero_upper_bound * -1.0

    # average_squared_zeroed = average_squared.copy()
    # for index,item in enumerate(average_squared):
    #     if item > 0 and item < zero_upper_bound:
    #         average_squared_zeroed[index] = 0
    
    # for index,item in enumerate(average_squared_zeroed[1:-1]):
    #     if average_squared_zeroed[index-1] != 0 and average_squared_zeroed[index+1] !=0 and average_squared_zeroed[index] == 0:
    #         average_squared_zeroed[index] = average_squared_zeroed[index-1]
        
    # # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    # isnonzero = np.concatenate(([0], (np.asarray(average_squared_zeroed) != 0).view(np.int8), [0]))
    # absdiff = np.abs(np.diff(isnonzero))
    # # Runs start and end where absdiff is 1.
    # ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    # r_maximums = []
    # r_max_indexes = []
    # for qrs_block in ranges:
    #     r_block = average_squared_zeroed[int(qrs_block[0]):int(qrs_block[1])]
    #     r_max_index = np.argmax(r_block)
    #     qrs_max = r_block[r_max_index]

    #     r_max_index = r_max_index + int(qrs_block[0])

    #     r_maximums.append(qrs_max)
    #     r_max_indexes.append(r_max_index)

    # # print(str(r_maximums))
    # # print(str(r_max_indexes))

    # max_r = np.amax(r_maximums)
    # min_r = np.amin(r_maximums)
    # # print("Rmax = "+str(max_r))
    # # print("Rmin = "+str(min_r))

    # r_distances = []
    # for i in range(0,len(r_max_indexes)-1):
    #     r_distance = r_max_indexes[i+1] - r_max_indexes[i]
    #     r_distances.append(r_distance)

    # if len(r_distances) != 0:
    #     mean_rr = np.mean(r_distances)
    #     variance_rr = np.var(r_distances)
    # else:
    #     mean_rr = 0.0
    #     variance_rr = 0.0

    # # print("R distances = "+str(r_distances))
    # # print("Mean RR = "+str(mean_rr))
    # # print("RR Variance = "+str(variance_rr))

    # if len(qrs_length) != 0:
    #     min_qrs_length = np.amin(qrs_length)
    #     max_qrs_length = np.amax(qrs_length)
    #     mean_qrs_length = np.mean(qrs_length)
    #     variance_qrs_length = np.var(qrs_length)
    # else:
    #     min_qrs_length = 0.0
    #     max_qrs_length = 0.0
    #     mean_qrs_length = 0.0
    #     variance_qrs_length = 0.0

    p_wave_start,p_wave_end,t_wave_start,t_wave_end,p_wave_length,t_wave_length = get_t_p_wave(ecg_plot, r_max_indexes)

    #if __name__ == "__main__":
    return ecg_plot, p_wave_start, p_wave_end, q_point, r_max_indexes, s_point, t_wave_start, t_wave_end

    #return max_r, min_r, mean_rr, variance_rr, max_qrs_length, min_qrs_length, mean_qrs_length, variance_qrs_length, t_wave_length, p_wave_length

    # print("QRS max = "+str(max_qrs_length))
    # print("QRS min = "+str(min_qrs_length))
    # print("Mean QRS length = "+str(mean_qrs_length))
    # print("Variance QRS length = "+str(variance_qrs_length))

if __name__ == "__main__":

    #Magic Numbers
    for f in glob.glob("./mit_bih_processed_data_two_leads/V/*.txt"):
        #file_string = "./mit_bih_processed_data_two_leads/N/ecg_10.txt"
        print(f)

        ecg_1, ecg_2 = read_file(f)
        [ecg_plot, p_wave_start, p_wave_end, q_point, r_max_indexes, s_point, t_wave_start, t_wave_end] = feature_extract_ecg(ecg_1)

        plt.subplot(211)    

        plt.plot(ecg_1,'r',label="unfiltered ECG")
        plt.title("ECG Signal - lead 1")
        #plt.plot(five_point_array,'g--',label="five point derivation")
        for max in r_max_indexes:
            plt.axvline(x=max,label="Max "+str(max))

        plt.plot(q_point,0,'r+')
        plt.plot(s_point,0,'r+')

        plt.plot(p_wave_start,0,'g+')
        plt.plot(p_wave_end,0,'g+')

        plt.plot(t_wave_start,0,'b+')
        plt.plot(t_wave_end,0,'b+')

        #plt.legend(loc='upper left')
        # plt.plot(average_squared,'r',label="squared five point derivation")
        # plt.plot(moving_window_integration,'g',label="moving window integration")
        # plt.plot(derivation,label="derivation")
        # plt.title("Squared Five Point Analysis")
        # plt.legend(loc='upper left')
        # plt.figure()

        #[ecg_plot, five_point_array, average_squared, moving_window_integration, derivation, r_max_indexes, qrs_starts, qrs_ends, p_wave_start, p_wave_end, t_wave_start, t_wave_end] = feature_extract_ecg(ecg_2)

        plt.subplot(212)  

        plt.plot(ecg_2,'r',label="unfiltered ECG")
        plt.title("ECG Signal - lead 2")
        #plt.plot(five_point_array,'g--',label="five point derivation")
        for max in r_max_indexes:
            plt.axvline(x=max,label="Max "+str(max))
        
        plt.plot(q_point,0,'r+')
        plt.plot(s_point,0,'r+')

        plt.plot(p_wave_start,0,'g+')
        plt.plot(p_wave_end,0,'g+')

        plt.plot(t_wave_start,0,'b+')
        plt.plot(t_wave_end,0,'b+')

        #plt.legend(loc='upper left')
        # plt.plot(average_squared,'r',label="squared five point derivation")
        # plt.plot(moving_window_integration,'g',label="moving window integration")
        # plt.plot(derivation,label="derivation")
        # plt.title("Squared Five Point Analysis")
        # plt.legend(loc='upper left')

        plt.show()