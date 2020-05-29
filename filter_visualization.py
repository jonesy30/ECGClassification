
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

def read_file_for_feature_extraction(file_string, r_index=1):
    f = open(file_string, "r")
    found_data = []
    r_value = 0

    for i,line in enumerate(f):
        line = line.replace("\n","")
        #ECG signal stored in first line separated by spaces
        if i < 1:
            line_segments = line.split()

            if r_index == 1:
                r_value = line_segments[-1]
                del line_segments[-1]

            line_segments = [float(x) for x in line_segments]

            for item in line_segments:
                found_data.append(item)
    f.close()

    found_data_lead_1 = found_data[:430]
    found_data_lead_2 = found_data[430:]

    found_data_lead_1 = np.trim_zeros(found_data_lead_1)
    found_data_lead_2 = np.trim_zeros(found_data_lead_2)

    if r_index == 1:
        return found_data_lead_1,found_data_lead_2, r_value

    return found_data_lead_1,found_data_lead_2

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

model_location = 'saved_models\\cnn_hannun\\cnn_model'
model = tf.keras.models.load_model(model_location)
#model = tf.keras.models.load_model(model_location, custom_objects={'focal_loss_fixed': focal_loss()})

print(model.summary())

# for layer in model.layers:
#     if 'conv' not in layer.name:
#         continue
#     filters, biases = layer.get_weights()
#     print(str(layer.name)  + ": " + str(filters.shape))
#     print(biases)

# filters, biases = model.layers[0].get_weights()
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min)/(f_max - f_min)

# n_filters, ix = 4, 1
# for i in range(n_filters):
#     f = filters[:,:,i]
#     ax = plt.subplot(2,n_filters,ix)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(i)

#     plt.imshow(f, cmap='gray')
#     ix += 1

# filters, biases = model.layers[4].get_weights()
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min)/(f_max - f_min)

# for i in range(n_filters):
#     f = filters[:,:,i]
#     ax = plt.subplot(2,n_filters,ix)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(i)

#     plt.imshow(f, cmap='gray')
#     ix += 1

# plt.suptitle("Feature Visualisation of Conv Layers")
# plt.figure()
# #plt.show()

# for i in range(len(model.layers)):
#     layer = model.layers[i]
#     if 'conv' not in layer.name:
#         continue
#     print(str(i) + ": " + layer.name + " - " + str(layer.output.shape))

# ixs = [0,1,2,4,5,6]
# outputs = [model.layers[i].output for i in ixs]
# model = Model(inputs=model.inputs, outputs=outputs)

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

grads1 = visualize_saliency(model, 40, filter_indices=None, seed_input=ecg,keepdims=True)

#grads1 = grads1[2]
#grads1 = np.rot90(grads1,k=1)
#fig, ax = plt.subplots(figsize=(18,2))
#ax.plot(ecg_plot_filtered)
#ax.imshow(grads1, cmap='jet',interpolation='nearest',aspect='auto')

ecg_lead_1, ecg_lead_2, r_value = read_file_for_feature_extraction(filename)
ecg_plot, p_wave_start, p_wave_end, q_point, r_max_index, s_point, t_wave_start, t_wave_end = feature_extract_ecg(ecg_lead_1, r_value)

def get_saliency_feature_value(start, end, grads, complete_gradient,ecg,plt):
    grads_subset = grads[start:end]

    mean_grad = np.mean(grads_subset)

    for i in range(len(grads_subset)):
        complete_gradient.append(mean_grad)

    #plt.plot(ecg[start:end],color=plt.cm.rainbow(mean_grad))

    return complete_gradient

complete_gradient = []

#before p wave
complete_gradient = get_saliency_feature_value(0,p_wave_start,grads1,complete_gradient,ecg_original,plt)
#p-wave
complete_gradient = get_saliency_feature_value(p_wave_start,p_wave_end,grads1,complete_gradient,ecg_original,plt)
#p - q
complete_gradient = get_saliency_feature_value(p_wave_end,q_point,grads1,complete_gradient,ecg_original,plt)
#q - r
complete_gradient = get_saliency_feature_value(q_point,r_max_index,grads1,complete_gradient,ecg_original,plt)
#r - s
complete_gradient = get_saliency_feature_value(r_max_index,s_point,grads1,complete_gradient,ecg_original,plt)
#s-t segment
complete_gradient = get_saliency_feature_value(s_point,t_wave_start,grads1,complete_gradient,ecg_original,plt)
#t wave
complete_gradient = get_saliency_feature_value(t_wave_start,t_wave_end,grads1,complete_gradient,ecg_original,plt)
#after t-wave
complete_gradient = get_saliency_feature_value(t_wave_end,430,grads1,complete_gradient,ecg_original,plt)

if two_leads == 1:
    #second lead....

    #before p wave
    complete_gradient = get_saliency_feature_value(430,p_wave_start+430,grads1,complete_gradient,ecg_original,plt)
    #p-wave
    complete_gradient = get_saliency_feature_value(p_wave_start+430,p_wave_end+430,grads1,complete_gradient,ecg_original,plt)
    #p - q
    complete_gradient = get_saliency_feature_value(p_wave_end+430,q_point+430,grads1,complete_gradient,ecg_original,plt)
    #q - r
    complete_gradient = get_saliency_feature_value(q_point+430,r_max_index+430,grads1,complete_gradient,ecg_original,plt)
    #r - s
    complete_gradient = get_saliency_feature_value(r_max_index+430,s_point+430,grads1,complete_gradient,ecg_original,plt)
    #s-t segment
    complete_gradient = get_saliency_feature_value(s_point+430,t_wave_start+430,grads1,complete_gradient,ecg_original,plt)
    #t wave
    complete_gradient = get_saliency_feature_value(t_wave_start+430,t_wave_end+430,grads1,complete_gradient,ecg_original,plt)
    #after t-wave
    complete_gradient = get_saliency_feature_value(t_wave_end+430,860,grads1,complete_gradient,ecg_original,plt)

upsampling_factor = 25

plot_grads = []
for item in complete_gradient:
    for i in range(upsampling_factor):
        plot_grads.append(item)

ecg_upsampled = signal.resample(ecg_original,upsampling_factor*860)

#plt.plot(ecg_original,zorder=1)
sc = ax.scatter(np.arange(0,860,1/upsampling_factor),ecg_upsampled,c=plot_grads,s=5,zorder=2)

plt.colorbar(sc)

ax.axvspan(p_wave_start, p_wave_end, alpha=0.5, color='coral')
ax.axvspan(p_wave_start+430, p_wave_end+430, alpha=0.5, color='coral')

ax.axvspan(q_point, r_max_index, alpha=0.5, color='yellow')
ax.axvspan(q_point+430, r_max_index+430, alpha=0.5, color='yellow')

ax.axvspan(r_max_index, s_point, alpha=0.5, color='lightgreen')
ax.axvspan(r_max_index+430, s_point+430, alpha=0.5, color='lightgreen')

ax.axvspan(t_wave_start, t_wave_end, alpha=0.5, color='lightseagreen')
ax.axvspan(t_wave_start+430, t_wave_end+430, alpha=0.5, color='lightseagreen')

_, top = ax.get_ylim()
label_height = 650

plt.annotate("P-wave",xy=(((p_wave_end-p_wave_start)/2)+p_wave_start, label_height), ha='center', color='coral')
plt.annotate("Q-R",xy=(((r_max_index-q_point)/2)+q_point, label_height), ha='right', color='goldenrod')
plt.annotate("R-S",xy=(((s_point-r_max_index)/2)+r_max_index, label_height), color='darkgreen')
plt.annotate("T-wave",xy=(((t_wave_end-t_wave_start)/2)+t_wave_start, label_height), ha='center', color='lightseagreen')

plt.annotate("P-wave",xy=(((p_wave_end-p_wave_start)/2)+p_wave_start+430, label_height), ha='center', color='coral')
plt.annotate("Q-R",xy=(((r_max_index-q_point)/2)+q_point+430, label_height), ha='right', color='goldenrod')
plt.annotate("R-S",xy=(((s_point-r_max_index)/2)+r_max_index+430, label_height), color='darkgreen')
plt.annotate("T-wave",xy=(((t_wave_end-t_wave_start)/2)+t_wave_start+430, label_height), ha='center', color='lightseagreen')

plt.title("Saliency Map - Abonormal (Atrial Premature Beat)")
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