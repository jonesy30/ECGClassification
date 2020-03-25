
import matplotlib.pyplot as plt 
import numpy as np
import scipy
import sys  
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np   
from numpy import fft
import os 

LENGTH = 400

# def normalize(ecg_signal):
#     max_value = max(ecg_signal)
#     min_value = min(ecg_signal)

#     range_values = max_value - min_value

#     return [(x - min_value)/range_values for x in ecg_signal]

# overly_noisy_ecgs = [112, 123, 136, 137, 144, 149, 197, 214, 238, 301, 311, 422, 476, 493, 535, 539, 569, 62, 620, 621, 653, 59, 705, 734, 76]
# overly_noisy_filenames = ["ecg_"+str(n)+".ecg" for n in overly_noisy_ecgs]

# valid_ecgs = []
# really_not_valid_ecgs = []

# split_ecgs = 0

# for path, subdirs, files in os.walk("./processed_data/"):
#     for name in files:
#         if name not in overly_noisy_filenames:
#             file_path = os.path.join(path, name)
#             f = open(file_path, "r")
#             ecg = np.fromfile(f, dtype=np.int16)
#             if len(ecg) >= LENGTH:
#                 valid_ecgs.append(file_path)
#                 complete_chunks = len(ecg)//LENGTH

#                 for i in range(complete_chunks):
#                     start_index = i * LENGTH
#                     end_index = start_index + LENGTH

#                     new_ecg = ecg[start_index:end_index]
#                     new_file = path.replace("processed_data","split_processed_data") + "/ecg_" + str(split_ecgs) + ".ecg"
#                     new_f = open(new_file, "wb")

#                     new_f.write(bytes(new_ecg))
#                     new_f.close()
#                     split_ecgs += 1

#         elif len(ecg) == 0:
#             really_not_valid_ecgs.append(file_path)

# print("Valid ecgs: "+str(len(valid_ecgs)))
# print("Invalid ecgs: "+str(len(really_not_valid_ecgs)))
# print("Split ecgs: "+str(split_ecgs))

for path, subdirs, files in os.walk("./split_processed_data/"):
    for name in files:
        file_path = os.path.join(path, name)
        f = open(file_path, "r")
        ecg = np.fromfile(f, dtype=np.int16)
        plt.plot(ecg)
        plt.title(name)
        plt.show()

file_string = "./split_processed_data/NSR/ecg_3305.ecg"
f = open(file_string, "r")
#a = np.fromfile(f, dtype=np.uint16)
normal2 = np.fromfile(f, dtype=np.int16)
#normal2 = normal2[:LENGTH]

#plt.plot(afib)
plt.plot(normal2)
plt.title("ecg_1628 - NORMAL")
# fourier_transform = fft.fft(normal2)

# plt.figure()
# plt.plot(fourier_transform)

#plt.plot(afib2)
#plt.plot(normal2)
#plt.legend(["Unfiltered", "Filtered"])
#plt.plot(afl1)
#plt.plot(afl2)
plt.show()