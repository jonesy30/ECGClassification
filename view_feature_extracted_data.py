import numpy as np
import os
import matplotlib.pyplot as plt

def read_data(filename):
    
    data = []
    labels = []
    for file in os.listdir(filename):
        f = open(str(filename+file), "r")
        found_data = []
        label = ""
        label_text = ""
        write_string = []
        for i,line in enumerate(f):
            line = line.replace("\n","")
            if i < 8:
                write_string.append(line)
            else:
                label = line
        found_data = write_string
        f.close()

        if label != "":
            data.append(found_data)
            labels.append(label)
    
    return data, labels

#max_r, min_r, mean_rr, variance_rr, max_qrs_length, min_qrs_length, mean_qrs_length, variance_qrs_length
(training_data, training_labels) = read_data("./processed_data/network_data/training_set/")
print(str(training_labels))

unpacked_data = []

for i,data in enumerate(training_data):
    this_unpacked = []
    for j,item in enumerate(data):
        this_unpacked.append(item)
    
    print("Training label = "+str(training_labels[i]))
    this_unpacked.append(training_labels[i])
    unpacked_data.append(this_unpacked)

class_names = ['AFIB', 'AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NSR', 'SUDDEN_BRADY', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']
variances_rr = [0]*len(class_names)
mean_rr = [0]*len(class_names)

for i,item in enumerate(unpacked_data[:-1]):
    if item[-1] in class_names:
        variances_rr[class_names.index(item[-1])] = item[3]
        mean_rr[class_names.index(item[-1])] = item[2]

print("Variances = "+str(variances_rr))
print("Means = "+str(mean_rr))

cumulative_variances = []
cumulative_means = []
variances_rr = np.array(variances_rr).astype(np.float)
mean_rr = np.array(mean_rr).astype(np.float)

for label in variances_rr:
    cumulative_variances.append(np.mean(label))
    
for label in mean_rr: 
    cumulative_means.append(np.mean(label))

print(str(cumulative_variances))
plt.figure()
plt.title("Variances")
plt.bar(class_names, cumulative_variances, color = 'r')
plt.xticks(class_names, fontsize=7, rotation=30)
plt.xlabel("Condition")

plt.figure()
plt.title("Means")
plt.bar(class_names, cumulative_means, color = 'b')
plt.xticks(class_names, fontsize=7, rotation=30)
plt.xlabel("Condition")
plt.show()

