import wfdb
import matplotlib.pyplot as plt
import glob, os
import numpy as np
from scipy import signal
from scipy import fftpack
import sys

mean_values_l1_incart = []
mean_values_l2_incart = []

for f in glob.glob("./external_validation_data/st_petersburg/V/*.txt"):
    file = open(f, "r")

    ecg_plot = file.read()
    ecg_plot = ecg_plot.strip()
    ecg = ecg_plot.split(" ")
    ecg = [int(n) for n in ecg]

    lead_one = ecg[:430]
    lead_two = ecg[430:]

    lead_one_mean = np.mean(lead_one)
    lead_two_mean = np.mean(lead_two)

    mean_values_l1_incart.append(lead_one_mean)
    mean_values_l2_incart.append(lead_two_mean)

average_l1_incart = np.mean(mean_values_l1_incart)
average_l2_incart = np.mean(mean_values_l2_incart)

print("Lead 1 INCART " + str(average_l1_incart))
print("Lead 2 INCART " + str(average_l2_incart))

mean_values_l1_mit = []
mean_values_l2_mit = []

for f in glob.glob("./mit_bih_processed_data_two_leads/V/*.txt"):
    file = open(f, "r")

    ecg_plot = file.read()
    ecg_plot = ecg_plot.strip()
    ecg = ecg_plot.split(" ")
    ecg = [int(n) for n in ecg]

    lead_one = ecg[:430]
    lead_two = ecg[430:]

    lead_one_mean = np.mean(lead_one)
    lead_two_mean = np.mean(lead_two)

    mean_values_l1_mit.append(lead_one_mean)
    mean_values_l2_mit.append(lead_two_mean)

average_l1_mit = np.mean(mean_values_l1_mit)
average_l2_mit = np.mean(mean_values_l2_mit)

print("Lead 1 MIT-BIH " + str(average_l1_mit))
print("Lead 2 MIT-BIH " + str(average_l2_mit))

sys.exit()

fft_ecgs_incart = []

for f in glob.glob("./external_validation_data/st_petersburg/V/*.txt"):
    #f = "./mit_bih_two_second_samples/N/ecg_10007.txt"
    file = open(f,"r")

    ecg_plot = file.read()
    ecg_plot = ecg_plot.strip()
    ecg = ecg_plot.split(" ")
    ecg = [int(n) for n in ecg]

    lead_one = ecg[:430]

    f_s = 360

    fft_ecg = fftpack.fft(lead_one)

    fft_ecgs_incart.append(fft_ecg)

mean_fft_incart = []
for i in range(len(fft_ecgs_incart[0])):
    #i_values = [fft_ecg[j][i] for j in fft_ecgs_incart]

    i_values = []
    for ecg in fft_ecgs_incart:
        i_values.append(ecg[i])

    mean = np.mean(i_values)
    mean_fft_incart.append(mean)

fft_ecgs_mit = []

for f in glob.glob("./mit_bih_processed_data_two_leads/V/*.txt"):
    file = open(f,"r")

    ecg_plot = file.read()
    ecg_plot = ecg_plot.strip()
    ecg = ecg_plot.split(" ")
    ecg = [int(n) for n in ecg]

    lead_one = ecg[:430]

    f_s = 360

    fft_ecg = fftpack.fft(lead_one)

    fft_ecgs_mit.append(fft_ecg)

mean_fft_mit = []
for i in range(len(fft_ecgs_mit[0])):
    #i_values = [fft_ecg[j][i] for j in fft_ecgs_incart]

    i_values = []
    for ecg in fft_ecgs_mit:
        i_values.append(ecg[i])

    mean = np.mean(i_values)
    mean_fft_mit.append(mean)

# plt.plot(mean_fft_incart)
# plt.plot(mean_fft_mit, linestyle="--")
# plt.xlabel("Frequency   ")
# plt.legend(['INCART','MIT-BIH'])
# plt.title("Frequency Distribution in Premature Ventricular Contractions in the MIT-BIH and Incart Datasets")
# plt.show()

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

for f in glob.glob("./external_validation_data/st_petersburg/V/*.txt"):
    #f = "./mit_bih_two_second_samples/N/ecg_10007.txt"
    file = open(f,"r")
    print(f)

    ecg_plot = file.read()
    ecg_plot = ecg_plot.strip()
    ecg = ecg_plot.split(" ")
    ecg = [int(n) for n in ecg]

    print(len(ecg))

    lead_one = ecg[:430]
    lead_two = ecg[430:]

    for g in glob.glob("./mit_bih_processed_data_two_leads/V/*.txt"):
        
        file_2 = open(g, "r")
        print(g)

        mit_ecg_plot = file_2.read()
        mit_ecg_plot = mit_ecg_plot.strip()
        mit_ecg = mit_ecg_plot.split(" ")
        mit_ecg = [int(n) for n in mit_ecg]

        mit_lead_one = mit_ecg[:430]
        mit_lead_two = mit_ecg[430:]

        plt.subplot(211)

        plt.plot(lead_one)
        plt.plot(mit_lead_one)
        plt.title("Lead One")
        plt.legend(['INCART','MIT-BIH'])

        plt.subplot(212)

        plt.plot(lead_two)
        plt.plot(mit_lead_two)
        plt.title("Lead Two")
        plt.legend(['INCART','MIT-BIH'])

        plt.suptitle("Dataset Comparison - Ventricular Contraction")

        plt.show()