"""
File which noise reduces the incoming ECG signal
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from scipy import signal

#define highpass filters
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

#define lowpass filters - unused as high frequency components are important and low-frequency is generally noise, but included for completeness
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

#reduce low frequency signal using a method given to me by Manlio Tassieri
def reduce_low_frequency(original, filename, plot):

    #Plot a 1st order polynomial of the data to get low-frequency and remove this from the signal at each point
    t = np.linspace(0,len(original),len(original))

    best_fit_linear = np.poly1d(np.polyfit(t,original, 1))
    for index, item in enumerate(original):
        original[index] = original[index] - best_fit_linear(index)

    #pass through a high pass filter
    original = butter_highpass_filter(original,0.5,500,5)

    #remove the R peaks from the signal, find a 3rd order polynomial which best fits the result, and remove this from the signal
    signal_without_r = original.copy()
    filtered = original.copy()

    original = np.abs(original)

    maximum = max(original)
    cutoff = maximum//2

    for index,item in enumerate(original):
        if item > cutoff:
            signal_without_r[index] = 0

    mean_values = []
    sum = 0
    counter = 1
    for index, item in enumerate(signal_without_r):
        if counter % 4 == 0:
            sum = sum + item
            mean = sum/counter
            mean_values.append([index-5, mean])
            counter = 1
            sum = 0
        else:
            counter += 1
            sum = sum + item

    if counter != 0:
        sum = sum + item
        mean = sum/counter
        mean_values.append([index-(counter//2), mean])

    x = []
    y = []
    for item in mean_values:
        [x_item, y_item] = item
        x.append(x_item)
        y.append(y_item)

    best_fit = np.poly1d(np.polyfit(x, y, 3))

    for index, item in enumerate(filtered):
        filtered[index] = filtered[index] - best_fit(index)

    signal_only_r = filtered.copy()
    signal_only_r = np.array(signal_only_r)    

    if max(filtered) < abs(min(filtered))/2:
        for index,item in enumerate(signal_only_r):
            signal_only_r[index] = item*-1

    maximum = max(signal_only_r)
    cutoff = maximum/10

    for index,item in enumerate(signal_only_r):
        if item <= cutoff:
            signal_only_r[index] = 0

    #Plot the signal with only the R peaks (only used for testing)
    if plot == 2:
        plt.plot(signal_only_r)
        plt.title("Only r")
        plt.show()

    #Plot if flag set (only used for testing)
    if plot == 1:
        plt.plot(original, label="Original")
        plt.plot(filtered, label="Filtered")
        plt.title(filename + " (manlio method)")
        plt.legend()
        plt.show()

    return filtered

#Pass ECG signal through a highpass filter (and plot if flag set - only for testing)
def highpass_reduce(ecg, filename,plot):
    filtered = butter_highpass_filter(ecg, 1.5, 200, 2)

    if plot == 1:
        plt.plot(ecg, label='Original')
        plt.plot(filtered, label='Filtered')
        plt.title(filename + " (highpass only)")
        plt.legend()
        plt.show()

    return filtered

#main method - noise reduce the signal
def noise_reduce(ecg, filename, plot=0):
    
    #manlio method is method given to me by Manlio Tassieri - two different options, Manlio method or ordinary highpass filter
    current_method = "manlio"
    
    if current_method == "manlio":
        return reduce_low_frequency(ecg, filename, plot)
    else:
        return highpass_reduce(ecg, filename, plot)