
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

model_location = 'saved_models\\lstm_two_leads\\lstm_model'
#model = tf.keras.models.load_model(model_location)
model = tf.keras.models.load_model(model_location, custom_objects={'focal_loss_fixed': focal_loss()})

print(model.summary())

two_leads = 1

filename = "./just_testing/E/pred_A_278.txt"
#filename = "./mit_bih_processed_data_two_leads/A/ecg_7.txt"
f = open(filename, "r")
ecg = []
for i,line in enumerate(f):
    line = line.replace("\n","")
    #ECG signal stored in first line separated by spaces
    if i < 1:
        line_segments = line.split()

        if two_leads == 0:
            line_segments = line_segments[:430]
        line_segments = [float(x) for x in line_segments]

        for item in line_segments:
            ecg.append(item)
f.close()

ecg_original = ecg.copy()

ecg_plot_filtered = normalize(ecg)

ecg = [np.asarray(item) for item in ecg]
ecg_plot_filtered = np.array(ecg_plot_filtered)

ecg = np.array(ecg)

ecg = ecg[:, np.newaxis]
ecg = expand_dims(ecg, axis=0)

fig, ax = plt.subplots(figsize=(18,2))

grads1 = visualize_saliency(model, 0, filter_indices=None, seed_input=ecg,keepdims=True)

upsampling_factor = 25

plot_grads = []
for item in grads1:
    for i in range(upsampling_factor):
        plot_grads.append(item)

ecg_upsampled = signal.resample(ecg_original,upsampling_factor*860)

#plt.plot(ecg_original,zorder=1)
sc = ax.scatter(np.arange(0,860,1/upsampling_factor),ecg_upsampled,c=plot_grads,s=5,zorder=2)

plt.colorbar(sc)

plt.title("Saliency Map - Abonormal (Atrial Premature Beat)")
plt.grid(which='both')
plt.minorticks_on()

plt.show()