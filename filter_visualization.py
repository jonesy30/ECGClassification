
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from analyse_ml_results import analyse_results
from tensorflow.keras.models import Model
from numpy import expand_dims
from vis.visualization import visualize_saliency
from vis.visualization import visualize_activation
from tensorflow.keras import activations
from vis.utils import utils
from scipy.signal import butter, lfilter, freqz
from scipy import signal

class_names = ['A','E','j','L','N','P','R','V']
two_leads = 0

#Function which normalizes the ECG signal
def normalize(ecg_signal):
    max_value = max(ecg_signal)
    min_value = min(ecg_signal)

    range_values = max_value - min_value

    if range_values == 0:
        return ecg_signal

    normalised = [(x - min_value)/range_values for x in ecg_signal]

    return [a * 50 for a in normalised]
    #return [np.float32(a) for a in normalised]

def standardise(ecg_signal):
    standard_dev = np.std(ecg_signal)
    mean = np.mean(ecg_signal)

    standardised = [(x-mean)/standard_dev for x in ecg_signal]
    
    #return standardised
    return [a * 50 for a in standardised]

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

def read_data(foldername,save_unnormalised=False):
    
    data = []
    labels = []
    unnormalised = []

    #for each file in corresponding folder
    for file in os.listdir(foldername):
        f = open(str(foldername+file), "r")
        ecg = []
        label = ""
        for i,line in enumerate(f):
            line = line.replace("\n","")
            #ECG signal stored in first line separated by spaces
            if i < 1:
                line_segments = line.split()
                if two_leads == 0:
                    line_segments = line_segments[:430]
                for i,item in enumerate(line_segments):
                    line_segments[i] = float(item)

                for item in line_segments:
                    ecg.append(item)
            #label stored on second line
            else:
                if str(line) not in class_names:
                    label = ""
                else:
                    index = class_names.index(line)
                    label = index
        f.close()

        #if label exists, store in trainng validation data
        if label != "":
            unnormalised.append(ecg)
            normalized_data = normalize(ecg, file)
            data.append(normalized_data)
            labels.append(label)
    
    if save_unnormalised == True:
        return [data, unnormalised], labels

    return data, labels

model_location = 'saved_models\\cnn\\cnn_model'
model = tf.keras.models.load_model(model_location)

print(model.summary())

for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    filters, biases = layer.get_weights()
    print(str(layer.name)  + ": " + str(filters.shape))
    print(biases)

filters, biases = model.layers[0].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min)/(f_max - f_min)

n_filters, ix = 4, 1
for i in range(n_filters):
    f = filters[:,:,i]
    ax = plt.subplot(2,n_filters,ix)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(i)

    plt.imshow(f, cmap='gray')
    ix += 1

filters, biases = model.layers[4].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min)/(f_max - f_min)

for i in range(n_filters):
    f = filters[:,:,i]
    ax = plt.subplot(2,n_filters,ix)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(i)

    plt.imshow(f, cmap='gray')
    ix += 1

plt.suptitle("Feature Visualisation of Conv Layers")
plt.figure()
#plt.show()

for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue
    print(str(i) + ": " + layer.name + " - " + str(layer.output.shape))

ixs = [0,1,2,4,5,6]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)

f = open("./mit_bih_processed_data_two_leads/network_data/training_set/ecg_77001.txt", "r")
ecg = []
for i,line in enumerate(f):
    line = line.replace("\n","")
    #ECG signal stored in first line separated by spaces
    if i < 1:
        line_segments = line.split()
        for i,item in enumerate(line_segments):
            line_segments[i] = float(item)

        for item in line_segments:
            ecg.append(item)
f.close()

#make one lead
#ecg = ecg[:430]

cutoff = 90
order = 5
fs = 360.0

#ecg_plot_filtered = butter_highpass_filter(ecg, cutoff, fs, order)
ecg_plot_filtered = butter_lowpass_filter(ecg, cutoff, fs, order)
ecg_plot_filtered = normalize(ecg_plot_filtered)

plt.plot(ecg)
plt.plot(ecg_plot_filtered)
plt.title("Raw ECG")
plt.legend(['Original','Filtered'])
plt.figure()

ecg = [np.asarray(item) for item in ecg]
ecg_plot_filtered = np.array(ecg_plot_filtered)

ecg = np.array(ecg)

ecg = ecg[:, np.newaxis]
ecg = expand_dims(ecg, axis=0)

feature_maps = model.predict(ecg)
square = 20
for map_index,fmap in enumerate(feature_maps):
    plt.suptitle("Feature Map, Layer "+str(ixs[map_index]))
    ix = 1
    for _ in range(square):
        ax = plt.subplot(square,1,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(fmap[:,ix-1],cmap='gray')
        ix += 1
    plt.figure()

grads1 = visualize_saliency(model, 5, filter_indices=None, seed_input=ecg,keepdims=True)


#grads1 = grads1[2]
grads1 = np.rot90(grads1,k=1)
fig, ax = plt.subplots(figsize=(18,2))
ax.plot(ecg_plot_filtered)
ax.imshow(grads1, cmap='jet',interpolation='nearest',aspect='auto')
plt.title("Saliency Map - Normal")
plt.grid(which='both')
plt.minorticks_on()
#ax.set_aspect(aspect=0.2)
plt.figure()

#Class Activation Mapping
model.layers[4].activation = activations.linear
model = utils.apply_modifications(model)

c_a_m = visualize_activation(model, 4, filter_indices=1, max_iter=10000,verbose=False)
plt.title("Class Activation Mapping: AFIB")
plt.plot(c_a_m)
plt.figure()

c_a_m = visualize_activation(model, 4, filter_indices=5, max_iter=10000,verbose=False)
plt.title("Class Activation Mapping: Normal")
plt.plot(c_a_m)

plt.show()