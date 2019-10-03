
import matplotlib.pyplot as plt 
import numpy as np
import scipy
import sys  
from scipy import signal
from scipy import pi
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np    
from scipy.signal import butter, lfilter, freqz 

LENGTH = 1000

file_string = "./processed_data/AFIB/ecg_60.ecg"
f = open(file_string, "r")
#a = np.fromfile(f, dtype=np.uint16)
afib = np.fromfile(f, dtype=np.int16)
afib = afib[:LENGTH]

file_string = "./processed_data/AFIB/ecg_378.ecg"
f = open(file_string, "r")
#a = np.fromfile(f, dtype=np.uint16)
afib2 = np.fromfile(f, dtype=np.int16)
afib2 = afib2[:LENGTH]

file_string = "./processed_data/AFIB/ecg_1065.ecg"
f = open(file_string, "r")
#a = np.fromfile(f, dtype=np.uint16)
normal = np.fromfile(f, dtype=np.int16)
normal = normal[:LENGTH]

file_string = "./processed_data/NSR/ecg_0.ecg"
f = open(file_string, "r")
#a = np.fromfile(f, dtype=np.uint16)
normal2 = np.fromfile(f, dtype=np.int16)
normal2 = normal2[:LENGTH]

file_string = "./processed_data/AFL/ecg_210.ecg"
f = open(file_string, "r")
#a = np.fromfile(f, dtype=np.uint16)
afl1 = np.fromfile(f, dtype=np.int16)
afl1 = afl1[:LENGTH]

file_string = "./processed_data/AFL/ecg_240.ecg"
f = open(file_string, "r")
#a = np.fromfile(f, dtype=np.uint16)
afl2 = np.fromfile(f, dtype=np.int16)
afl2 = afl2[:LENGTH]

#plt.plot(afib)
plt.plot(normal)
#plt.plot(afib2)
#plt.plot(normal2)
#plt.legend(["Unfiltered", "Filtered"])
#plt.plot(afl1)
#plt.plot(afl2)
plt.legend(['Afib1','Normal1','Afib2','Normal2','Afl1','Afl2'])
plt.show()