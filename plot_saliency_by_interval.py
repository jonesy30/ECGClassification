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
from ecg_feature_extraction import feature_extract_ecg
import sys
import glob
import time
import gc
import pandas as pd

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

class_names = ['A','E','j','L','N','P','R','V']

def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

#Function which normalizes the ECG signal
def normalize(ecg_signal):
    max_value = max(ecg_signal)
    min_value = min(ecg_signal)

    range_values = max_value - min_value

    if range_values == 0:
        return ecg_signal

    normalised = [(x - min_value)/range_values for x in ecg_signal]

    return [a * -50 for a in normalised]
    #return [np.float32(a) for a in normalised]

model_location = 'saved_models\\cnn_hannun\\cnn_model'
model = tf.keras.models.load_model(model_location)
#model = tf.keras.models.load_model(model_location, custom_objects={'focal_loss_fixed': focal_loss()})

print(model.summary())

two_leads = 1

filename = "./mit_bih_processed_data_two_leads_r_marker/A/ecg_7.txt"
f = open(filename, "r")
ecg = []
for i,line in enumerate(f):
    line = line.replace("\n","")
    #ECG signal stored in first line separated by spaces
    if i < 1:
        line_segments = line.split()

        r_value = line_segments[-1]
        r_value = int(r_value)
        del line_segments[-1]

        if two_leads == 0:
            line_segments = line_segments[:430]
        line_segments = [float(x) for x in line_segments]

        for item in line_segments:
            ecg.append(item)
f.close()

ecg_original = ecg.copy()

#make one lead
#ecg = ecg[:430]

cutoff = 90
order = 5
fs = 360.0

#ecg_plot_filtered = butter_highpass_filter(ecg, cutoff, fs, order)
#ecg_plot_filtered = butter_lowpass_filter(ecg, cutoff, fs, order)
ecg_plot_filtered = normalize(ecg)

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

# feature_maps = model.predict(ecg)
# square = 20
# for map_index,fmap in enumerate(feature_maps):
#     plt.suptitle("Feature Map, Layer "+str(ixs[map_index]))
#     ix = 1
#     for _ in range(square):
#         ax = plt.subplot(square,1,ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.imshow(fmap[:,ix-1],cmap='gray')
#         ix += 1
#     plt.figure()

fig, ax = plt.subplots(figsize=(18,2))

segment_names = ["-A12","-A11","-A10","-A9","-A8","-A7","-A6","-A5","-A4","-A3","-A2","-A1","+A1","+A2","+A3","+A4","+A5","+A6","+A7","+A8","+A9","+A10","+A11","+A12","-B12","-B11","-B10","-B9","-B8","-B7","-B6","-B5","-B4","-B3","-B2","-B1","+B1","+B2","+B3","+B4","+B5","+B6","+B7","+B8","+B9","+B10","+B11","+B12"]

df_file = pd.read_csv("C:/Users/yolaj/Dropbox/PhD/My Papers/IEEE BIBE/csv files/segments_j.csv")
mean_gradients = df_file.iloc[-3]
lower_quantile = df_file.iloc[-2]
upper_quantile = df_file.iloc[-1]

mean_gradients = mean_gradients.values.tolist()
del mean_gradients[0]
mean_gradients = [float(x) for x in mean_gradients]

lower_quantile = lower_quantile.values.tolist()
del lower_quantile[0]
lower_quantile = [float(x) for x in lower_quantile]

upper_quantile = upper_quantile.values.tolist()
del upper_quantile[0]
upper_quantile = [float(x) for x in upper_quantile]

plt.plot(mean_gradients, color="#4a86e8")
plt.fill_between(np.arange(0,len(mean_gradients),1), lower_quantile, upper_quantile, alpha=0.1, facecolor="#4a86e8")

df_file = pd.read_csv("C:/Users/yolaj/Dropbox/PhD/My Papers/IEEE BIBE/csv files/segments_correct_j_predictions.csv")
mean_gradients = df_file.iloc[-3]
lower_quantile = df_file.iloc[-2]
upper_quantile = df_file.iloc[-1]

mean_gradients = mean_gradients.values.tolist()
del mean_gradients[0]
mean_gradients = [float(x) for x in mean_gradients]

lower_quantile = lower_quantile.values.tolist()
del lower_quantile[0]
lower_quantile = [float(x) for x in lower_quantile]

upper_quantile = upper_quantile.values.tolist()
del upper_quantile[0]
upper_quantile = [float(x) for x in upper_quantile]

plt.plot(mean_gradients, color="#9900ff")
plt.fill_between(np.arange(0,len(mean_gradients),1), lower_quantile, upper_quantile, alpha=0.1, facecolor="#9900ff")

df_file = pd.read_csv("C:/Users/yolaj/Dropbox/PhD/My Papers/IEEE BIBE/csv files/segments_incorrect_j_predictions.csv")
counts = df_file.iloc[-4]
mean_gradients = df_file.iloc[-3]
lower_quantile = df_file.iloc[-2]
upper_quantile = df_file.iloc[-1]

counts = counts.values.tolist()
del counts[0]
counts = [int(x) for x in counts]

mean_gradients = mean_gradients.values.tolist()
del mean_gradients[0]
mean_gradients = [float(x) for x in mean_gradients]

lower_quantile = lower_quantile.values.tolist()
del lower_quantile[0]
lower_quantile = [float(x) for x in lower_quantile]

upper_quantile = upper_quantile.values.tolist()
del upper_quantile[0]
upper_quantile = [float(x) for x in upper_quantile]

plt.plot(mean_gradients, color="#00ffff")
plt.fill_between(np.arange(0,len(mean_gradients),1), lower_quantile, upper_quantile, alpha=0.1, facecolor="#00ffff")

plt.legend(["Total","Correct","Incorrect"])

label_names = []
for index, item in enumerate(segment_names):
    label_names.append(str(item) +"\n"+ str(counts[index]))

plt.xticks(np.arange(0,len(mean_gradients),1),segment_names)
ax.tick_params(axis="x",labelsize=8)
plt.title("Median Values and 25% - 75% Quartile Per Segment, Junctional Escape Beats")
plt.figure()

fig, ax = plt.subplots(figsize=(18,2))
plt.plot(mean_gradients, color="#00ffff")
plt.fill_between(np.arange(0,len(mean_gradients),1), lower_quantile, upper_quantile, alpha=0.1, facecolor="#00ffff")

label_names = []
for index, item in enumerate(segment_names):
    label_names.append(str(item) +"\n"+ str(counts[index]))

plt.xticks(np.arange(0,len(mean_gradients),1),segment_names)
ax.tick_params(axis="x",labelsize=8)
plt.title("Median Values and 25% - 75% Quartile Per Segment, Incorrect Junctional Escape Beats vs Normal Beats")

df_file = pd.read_csv("C:/Users/yolaj/Dropbox/PhD/My Papers/IEEE BIBE/csv files/segments_N.csv")
mean_gradients = df_file.iloc[-3]
lower_quantile = df_file.iloc[-2]
upper_quantile = df_file.iloc[-1]

mean_gradients = mean_gradients.values.tolist()
del mean_gradients[0]
mean_gradients = [float(x) for x in mean_gradients]

lower_quantile = lower_quantile.values.tolist()
del lower_quantile[0]
lower_quantile = [float(x) for x in lower_quantile]

upper_quantile = upper_quantile.values.tolist()
del upper_quantile[0]
upper_quantile = [float(x) for x in upper_quantile]

plt.plot(mean_gradients, color="#9900ff")
plt.fill_between(np.arange(0,len(mean_gradients),1), lower_quantile, upper_quantile, alpha=0.1, facecolor="#9900ff")

plt.legend(["Incorrectly Identified Junctional Escape Beats","Normal Beats"])
plt.show()

interval = int(0.1 * 360) #0.1 seconds * sampling rate  

mean_values_a = []
segments_a = []

end_index = r_value
start_column = "-A"
i = 1
while end_index > 0:

    column = start_column + str(i)
    mean_value = mean_gradients[column]
    mean_value = float(mean_value)

    mean_values_a.append(mean_value)
    
    start_index = end_index - interval

    if start_index < 0:
        start_index = 0
    segments_a.append([start_index, end_index])

    end_index = start_index

    i += 1

mean_values_a.reverse()
segments_a.reverse()

start_index = r_value
start_column = "+A"
i = 1
while start_index < 430:

    column = start_column + str(i)
    mean_value = mean_gradients[column]
    mean_value = float(mean_value)

    mean_values_a.append(mean_value)
    
    end_index = start_index + interval

    if end_index > 430:
        end_index = 430
    segments_a.append([start_index, end_index])

    start_index = end_index

    i += 1

mean_values_b = []
segments_b = []

end_index = r_value
start_column = "-B"
i = 1
while end_index > 0:

    column = start_column + str(i)
    mean_value = mean_gradients[column]
    mean_value = float(mean_value)

    mean_values_b.append(mean_value)
    
    start_index = end_index - interval

    if start_index < 0:
        start_index = 0
    segments_b.append([start_index+430, end_index+430])

    end_index = start_index

    i += 1

mean_values_b.reverse()
segments_b.reverse()

start_index = r_value
start_column = "+B"
i = 1
while start_index < 430:

    column = start_column + str(i)
    mean_value = mean_gradients[column]
    mean_value = float(mean_value)

    mean_values_a.append(mean_value)
    
    end_index = start_index + interval

    if end_index > 430:
        end_index = 430
    segments_b.append([start_index+430, end_index+430])

    start_index = end_index

    i += 1

for item in mean_values_b:
    mean_values_a.append(item)

for item in segments_b:
    segments_a.append(item)

upsampling_factor = 25

plot_grads = []
for index,item in enumerate(segments_a):

    mean_value = mean_values_a[index]

    start = item[0]
    end = item[1]

    if end == 860:
        end = 859
        #to make the plotting line segments happy, when I get to the end it plots to 860 and doesn't overflow

    print(start)
    print(end)
    print()

    plt.plot(np.arange(start,end+1,1),ecg_original[start:end+1],color=plt.cm.nipy_spectral(mean_value))

cmap = plt.cm.ScalarMappable(cmap=plt.cm.nipy_spectral)
fig.colorbar(cmap)

plt.title("Saliency Map - Average Contribution Per Feature (Premature Ventricular Contraction)")
plt.grid(which='both')
plt.minorticks_on()

plt.show()