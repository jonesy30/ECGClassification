import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import os

#Function which normalizes the ECG signal
def normalize(ecg_signal, filename):
    max_value = max(ecg_signal)
    min_value = min(ecg_signal)

    range_values = max_value - min_value

    if range_values == 0:
        print(max_value)
        print(min_value)
        print(filename)

    if range_values == 0:
        return ecg_signal

    normalised = [(x - min_value)/range_values for x in ecg_signal]

    return [np.float32(a) for a in normalised]

#Function which reads ECG data and labels from each file in folder
def read_data(file):
    
    labels = []
    unnormalised = []

    #for each file in corresponding folder
    f = open(str(file), "r")
    found_data = []
    label = ""
    for i,line in enumerate(f):
        line = line.replace("\n","")
        line_segments = line.split()

        line_segments = [float(x) for x in line_segments]

        for item in line_segments:
            found_data.append(item)
    f.close()

    normalized_data = normalize(found_data, file)

    return normalized_data

base_filename = "./saved_ecg_classifications/lstm/"

plt.suptitle("Correct Classifications - L")
counter = 1
chosen_files = set()
while (len(chosen_files) < 20):
    random_file=random.choice(os.listdir(base_filename+"/correct_predictions/L/"))
    chosen_files.add(random_file)

for file in chosen_files:
    ecg = read_data(base_filename+"/correct_predictions/L/"+file)

    # i = (counter//7)+1
    # j = counter % 7

    # print(i)
    # print(j)

    # ax = plt.subplot2grid((7,3), (i,j))

    plt.subplot(5,4,counter)

    plt.plot(ecg)
    #plt.title(file)

    counter += 1
    if counter >= 21:
        break

plt.figure()

plt.suptitle("Correct Classifications - A")
counter = 1
chosen_files = set()
while (len(chosen_files) < 20):
    random_file=random.choice(os.listdir(base_filename+"/correct_predictions/A/"))
    chosen_files.add(random_file)

for file in chosen_files:
    ecg = read_data(base_filename+"/correct_predictions/A/"+file)

    # i = (counter//7)+1
    # j = counter % 7

    # print(i)
    # print(j)

    # ax = plt.subplot2grid((7,3), (i,j))

    plt.subplot(5,4,counter)

    plt.plot(ecg)
    #plt.title(file)

    counter += 1
    if counter >= 21:
        break

plt.figure()

plt.suptitle("Correct Classifications - V")
counter = 1
chosen_files = set()
while (len(chosen_files) < 20):
    random_file=random.choice(os.listdir(base_filename+"/correct_predictions/V/"))
    chosen_files.add(random_file)

for file in chosen_files:
    ecg = read_data(base_filename+"/correct_predictions/V/"+file)

    # i = (counter//7)+1
    # j = counter % 7

    # print(i)
    # print(j)

    # ax = plt.subplot2grid((7,3), (i,j))

    plt.subplot(5,4,counter)

    plt.plot(ecg)
    #plt.title(file)

    counter += 1
    if counter >= 21:
        break

plt.figure()

plt.suptitle("Incorrect Classifications - L")
chosen_files = set()
counter = 1
while (len(chosen_files) < 20):
    random_file=random.choice(os.listdir(base_filename+"/incorrect_predictions/V/"))
    if "_L_" in random_file:
        chosen_files.add(random_file)

for file in chosen_files:
    ecg = read_data(base_filename+"/incorrect_predictions/V/"+file)

    # i = (counter//7)+1
    # j = counter % 7

    # print(i)
    # print(j)

    # ax = plt.subplot2grid((7,3), (i,j))

    plt.subplot(5,4,counter)

    plt.plot(ecg)
    #plt.title(file)

    counter += 1
    if counter >= 21:
        break

plt.figure()

plt.suptitle("Incorrect Classifications - A")
chosen_files = set()
counter = 1
while (len(chosen_files) < 20):
    random_file=random.choice(os.listdir(base_filename+"/incorrect_predictions/V/"))
    if "_A_" in random_file:
        chosen_files.add(random_file)

for file in chosen_files:
    ecg = read_data(base_filename+"/incorrect_predictions/V/"+file)

    # i = (counter//7)+1
    # j = counter % 7

    # print(i)
    # print(j)

    # ax = plt.subplot2grid((7,3), (i,j))

    plt.subplot(5,4,counter)

    plt.plot(ecg)
    #plt.title(file)

    counter += 1
    if counter >= 21:
        break

plt.show()



