# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os
import wfdb
import numpy as np

beat_dict = {'N':'Normal','L':'LBBB','R':'RBBB','A':'APB','a':'AAPB','J':'JUNCTIONAL','S':'SP','V':'VT','r':'RonT','e':'Aesc','j':'Jesc','n':'SPesc','E':'Vesc','/':'Paced'}
beat_lengths = []
base_filename = "./external_original/st_petersburg/files/"
#base_filename = "./mit_bih/"

above_430 = 0
within_430 = 0

max_length = 0
max_file = ""

for file in glob.glob(base_filename+"*.atr"):
    
    print(file)

    _, split, _ = file.split(".")
    filepath = split.split("\\")
    name = filepath[-1]
    filenumber = name.split("-")

    if len(filenumber) > 1:
        filenumber, _ = filenumber
    else:
        filenumber = filenumber[0]

    annotation = wfdb.rdann(base_filename+str(filenumber), 'atr', sampto=650000)
    ecg_signal, _ = wfdb.srdsamp(base_filename+str(filenumber))

    ecg = ecg_signal[:,0]

    annotation_indices = annotation.sample
    annotation_classes = annotation.symbol
    sampling_frequency = annotation.fs

    for index,beat_class in enumerate(annotation_classes):
        if beat_class not in beat_dict:
            np.delete(annotation_indices, index)
            np.delete(annotation_classes, index)

    complete_beats = []

    for index,current_beat in enumerate(annotation_indices):
        if index > 0 and index<len(annotation_indices)-1:
            beat_before = annotation_indices[index-1]
            beat_after = annotation_indices[index+1]

            start_recording = ((current_beat-beat_before)//2) + beat_before
            end_recording = ((beat_after-current_beat)//2)+current_beat

            complete_beats = ecg[start_recording:end_recording]
            beat_lengths.append(len(complete_beats))
            if len(complete_beats) > 307:
                above_430 += 1
            else:
                within_430 += 1
            if len(complete_beats) > max_length:
                max_length = len(complete_beats)
                max_file = file + ", beat "+str(index)


if "st_petersburg" in base_filename:
    beat_lengths = [1.401 * x for x in beat_lengths]

# matplotlib histogram
plt.hist(beat_lengths, color = 'blue', edgecolor = 'black',
         bins = int(180/5))

# seaborn histogram
sns.distplot(beat_lengths, hist=True, kde=False, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of ECG Lengths')
plt.xlabel('Lengths')

plt.figure()

plt.boxplot(beat_lengths)
plt.title("Boxplot of ECG Lengths")

print("Samples above 430 = "+str(above_430))
print("Samples within 430 = "+str(within_430))
print("Max length = "+str(max_length))
print("This belongs to "+str(max_file))
print()
print("Mean = "+str(np.mean(beat_lengths)))
print("Standard Deviation = "+str(np.std(beat_lengths)))
print()
print("Median = "+str(np.median(beat_lengths)))
print("Interquartile range = "+str(np.percentile(beat_lengths, 75, interpolation = 'midpoint')))
print("Interquartile range = "+str(np.percentile(beat_lengths, 25, interpolation = 'midpoint')))

plt.show()