
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import freqz
from scipy.signal import butter, lfilter

#Magic Numbers
ECG_SELECTION = 0
N = 35

file_string = "raw_ecg_data/NSR/ecg_" + str(ECG_SELECTION) + ".ecg"
f = open(file_string,"r")

ecg_plot = np.fromfile(f, dtype=np.int16)
#ecg_plot = ecg_plot[:256]

plt.plot(ecg_plot,label="unfiltered ECG")
plt.title("ECG Signal")

#Band pass filter

#Low pass
# low_passed_ecg = []
# for i,element in enumerate(ecg_plot):
#     y = ecg_plot[i]
#     if i > 2:
#         #y = y + 2.0 * low_passed_ecg[i-1]
#         #y = y - low_passed_ecg[i-2]
#         if i > 6:
#             y = y - 2.0 * ecg_plot[i-6]
#         if i > 12:
#             y = y + ecg_plot[i-12]
    
#     low_passed_ecg.append(y)

# high_passed_ecg = []
# for i,element in enumerate(low_passed_ecg):
#     y = ecg_plot[i]
#     if i > 1:
#         y = y + high_passed_ecg[i-1]
#     if i > 32:
#         y = y - ecg_plot[i-32]

#     high_passed_ecg.append(y)

# plt.plot(low_passed_ecg,label="low passed ECG")
# plt.plot(high_passed_ecg,label="high passed ECG")

# band_pass_filtered_ecg = []
# for i, element in enumerate(ecg_plot):
#     p = 0
#     y = ecg_plot[i]
#     if i > 32:
#         y = y - ecg_plot[i-32]
#     y = y / 32.0
#     if i > 16:
#         p = ecg_plot[i-16] - y

#     band_pass_filtered_ecg.append(p)

# plt.plot(band_pass_filtered_ecg,label="band passed ECG")

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

plt.plot(five_point_array,label="five point derivation")

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

qrs_length = []
for qrs_block in ranges:
    length = qrs_block[1] - qrs_block[0]
    qrs_length.append(length)

print("Ranges = ")
print(str(ranges))
print("Lengths = ")
print(str(qrs_length))
print()

#Finding RR features
max = np.amax(average_squared)
zero_upper_bound = max / 10.0
zero_lower_bound = zero_upper_bound * -1.0

average_squared_zeroed = average_squared.copy()
for index,item in enumerate(average_squared):
    if item > 0 and item < zero_upper_bound:
        average_squared_zeroed[index] = 0
    
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


print(str(r_maximums))
print(str(r_max_indexes))

max_r = np.amax(r_maximums)
min_r = np.amin(r_maximums)
print("Rmax = "+str(max_r))
print("Rmin = "+str(min_r))

r_distances = []
for i in range(0,len(r_max_indexes)-1):
    r_distance = r_max_indexes[i+1] - r_max_indexes[i]
    r_distances.append(r_distance)

mean_rr = np.mean(r_distances)
variance_rr = np.var(r_distances)

print("R distances = "+str(r_distances))
print("Mean RR = "+str(mean_rr))
print("RR Variance = "+str(variance_rr))

min_qrs_length = np.amin(qrs_length)
max_qrs_length = np.amax(qrs_length)
mean_qrs_length = np.mean(qrs_length)
variance_qrs_length = np.var(qrs_length)

print("QRS max = "+str(max_qrs_length))
print("QRS min = "+str(min_qrs_length))
print("Mean QRS length = "+str(mean_qrs_length))
print("Variance QRS length = "+str(variance_qrs_length))

#print(str(derivation[:300]))

plt.legend(loc='upper left')
plt.figure()
plt.plot(average_squared,'r',label="squared five point derivation")
plt.plot(moving_window_integration,'g',label="moving window integration")
plt.plot(derivation,label="derivation")
plt.title("Squared Five Point Analysis")
plt.legend(loc='upper left')
plt.show()