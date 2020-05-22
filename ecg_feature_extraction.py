
"""
File which extracts the ECG features returns the results
"""

import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from scipy import signal
import glob

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

    ecg_plot = butter_highpass_filter(ecg_plot, 2, 360, 2)
    ecg_plot = butter_lowpass_filter(ecg_plot, 50, 360, 2)

    # if len(r_max_indexes) < 3:
    #     print(r_max_indexes)
    #     sub_plot = ecg_plot[:r_max_indexes[0]]
    # else:
    #     sub_plot = ecg_plot[r_max_indexes[1]:r_max_indexes[2]]

    #lower_bound = (len(sub_plot)//10)
    #upper_bound = len(sub_plot)-lower_bound

    #sub_plot = sub_plot[lower_bound:upper_bound]

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

def read_file(file_string, r_index=1):
    f = open(file_string, "r")
    found_data = []
    r_value = 0

    for i,line in enumerate(f):
        line = line.replace("\n","")
        #ECG signal stored in first line separated by spaces
        if i < 1:
            line_segments = line.split()

            if r_index == 1:
                r_value = line_segments[-1]
                del line_segments[-1]

            line_segments = [float(x) for x in line_segments]

            for item in line_segments:
                found_data.append(item)
    f.close()

    found_data_lead_1 = found_data[:430]
    found_data_lead_2 = found_data[430:]

    found_data_lead_1 = np.trim_zeros(found_data_lead_1)
    found_data_lead_2 = np.trim_zeros(found_data_lead_2)

    if r_index == 1:
        return found_data_lead_1,found_data_lead_2, r_value

    return found_data_lead_1,found_data_lead_2

def feature_extract_ecg(ecg_plot, r_index):

    ecg_plot = butter_highpass_filter(ecg_plot, 2, 360, 2)
    #ecg_plot = butter_lowpass_filter(ecg_plot, 15, 360, 2)

    r_max = r_index
    r_max_indexes = [r_max]

    r_index = int(r_index)

    sampling_rate = 360

    ecg_plot = butter_highpass_filter(ecg_plot, 2, 360, 2)

    before_r = ecg_plot[:r_index]
    after_r = ecg_plot[r_index:]

    # plt.plot(ecg_plot)
    # plt.figure()
    # plt.plot(before_r)
    # plt.figure()
    # plt.plot(after_r)

    #q-wave
    start_index = r_index - int(sampling_rate*0.1)
    if start_index < 0:
        start_index = 0
    q_wave_subplot = before_r[start_index:]
    q_point = np.argmin(q_wave_subplot) + start_index

    #s-wave
    end_index = int(sampling_rate*0.1)
    s_wave_subplot = after_r[:end_index]
    s_point = np.argmin(s_wave_subplot) + r_index

    ecg_plot_p = butter_highpass_filter(ecg_plot, 5, 360, 2)
    ecg_plot_p = butter_lowpass_filter(ecg_plot_p, 50, 360, 2)

    #p-wave
    p_wave_block = ecg_plot_p[:q_point]
    p_wave_max = np.argmax(p_wave_block)

    # plt.figure()
    # plt.plot(p_wave_block)
    # plt.show()    

    p_wave_start = p_wave_max - int(sampling_rate*0.1)
    p_wave_start_block = ecg_plot[p_wave_start:p_wave_max]
    
    if len(p_wave_start_block) == 0:
        p_wave_start = 0
    else:
        p_wave_start = np.argmin(p_wave_start_block) + p_wave_start

    p_wave_interval = int(sampling_rate*0.05)
    p_wave_distance = q_point - p_wave_max

    if p_wave_interval < p_wave_distance:
        p_wave_end = p_wave_max + p_wave_interval
    else:
        p_wave_end = q_point

    p_wave_end_block = ecg_plot_p[p_wave_max:p_wave_end]
    p_wave_end = np.argmin(p_wave_end_block) + p_wave_max

    #t-wave
    print(s_point)
    t_wave_block = ecg_plot[s_point:]

    #t_wave_block = butter_lowpass_filter(t_wave_block, 50, 360, 5)

    t_wave_max = np.argmax(t_wave_block) + s_point

    t_wave_start = t_wave_max - int(sampling_rate*0.2)
    t_wave_start_block = ecg_plot[t_wave_start:t_wave_max]

    if len(t_wave_start_block) == 0:
        t_wave_start = s_point
    else:
        t_wave_start = np.argmin(t_wave_start_block) + t_wave_start

    t_wave_end = t_wave_max + int(sampling_rate*0.05)
    t_wave_end_block = ecg_plot[t_wave_max:t_wave_end]
    t_wave_end = np.argmin(t_wave_end_block) + t_wave_max

    # print("Q")
    # print(q_point)
    # print("S")
    # print(s_point)
    # print("P")
    # print(p_wave_max)
    # print(p_wave_start)
    # print(p_wave_end)
    # print("T")
    # print(t_wave_max)
    # print(t_wave_start)
    # print(t_wave_end)

    #plt.show()

    #if __name__ == "__main__":
    return ecg_plot, p_wave_start, p_wave_end, q_point, r_index, s_point, t_wave_start, t_wave_end

def feature_extract_ecg_old(ecg_plot, r_index):

    print(r_index)
    plt.plot(ecg_plot)
    plt.show()

    ecg_plot = butter_highpass_filter(ecg_plot, 2, 360, 2)
    #ecg_plot = butter_lowpass_filter(ecg_plot, 15, 360, 2)

    r_max = r_index
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
    zero_upper_bound = max / 60.0
    zero_lower_bound = zero_upper_bound * -1.0
    
    #derivation = np.gradient(derivation)
    for index,item in enumerate(derivation):
        if item > 0 and item < zero_upper_bound:
            derivation[index] = 0
        #elif item > zero_lower_bound and item < 0:
        elif item < 0:
            derivation[index] = 0 

    before_r = derivation[:r_max]
    after_r = derivation[r_max:]   

    print(r_max)

    plt.plot(ecg_plot)
    plt.figure()
    plt.plot(before_r)
    plt.figure()
    plt.plot(after_r)
    plt.show()

    q_point = 0
    for index,i in enumerate(reversed(before_r)):
        if round(i) <= 0:
            q_point = len(before_r) - index - 1
            break

    s_point = 0
    for index,i in enumerate(after_r):
        if round(i) == 0:
            s_point = index + r_max
            break
    
    p_wave_start,p_wave_end,t_wave_start,t_wave_end,p_wave_length,t_wave_length = get_t_p_wave(ecg_plot, r_max_indexes)

    #if __name__ == "__main__":
    return ecg_plot, p_wave_start, p_wave_end, q_point, r_max_indexes[0], s_point, t_wave_start, t_wave_end

if __name__ == "__main__":

    #Magic Numbers
    for f in glob.glob("./mit_bih_processed_data_two_leads_r_marker/N/*.txt"):
        #file_string = "./mit_bih_processed_data_two_leads/N/ecg_10.txt"
        print(f)

        ecg_1, ecg_2, r_index = read_file(f)

        r_index = int(r_index)

        [ecg_plot, p_wave_start, p_wave_end, q_point, r_index, s_point, t_wave_start, t_wave_end] = feature_extract_ecg(ecg_1, r_index)

        plt.subplot(211)    

        plt.plot(ecg_1,'r',label="unfiltered ECG")
        plt.title("ECG Signal - lead 1")
        #plt.plot(five_point_array,'g--',label="five point derivation")
        #for max in r_max_indexes:
        plt.axvline(x=r_index,label="Max "+str(r_index))

        plt.axvline(x=q_point,c='red')
        plt.axvline(x=s_point,c='red')

        plt.axvline(x=p_wave_start,c='green')
        plt.axvline(x=p_wave_end,c='green')

        plt.axvline(x=t_wave_start,c='blue')
        plt.axvline(x=t_wave_end,c='blue')

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
        #for max in r_max_indexes:
        plt.axvline(x=r_index,label="Max "+str(r_index))
        
        plt.axvline(x=q_point,c='red')
        plt.axvline(x=s_point,c='red')

        plt.axvline(x=p_wave_start,c='green')
        plt.axvline(x=p_wave_end,c='green')

        plt.axvline(x=t_wave_start,c='blue')
        plt.axvline(x=t_wave_end,c='blue')

        #plt.legend(loc='upper left')
        # plt.plot(average_squared,'r',label="squared five point derivation")
        # plt.plot(moving_window_integration,'g',label="moving window integration")
        # plt.plot(derivation,label="derivation")
        # plt.title("Squared Five Point Analysis")
        # plt.legend(loc='upper left')

        plt.show()