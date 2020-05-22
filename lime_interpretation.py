import skimage.segmentation
import numpy as np
import skimage
from skimage import io
import keras
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import copy
import sklearn.metrics
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.utils import to_categorical
import sys
import random
from keras.models import Model

import lime
import lime.lime_tabular
import lime.lime_image
from ecg_feature_extraction import feature_extract_ecg

class_names = ['A','E','j','L','N','P','R','V']
two_leads = 0


def predict_fn(incoming_permutation):
    # ecg = [np.asarray(item) for item in ecg]
    # ecg = np.array(ecg)

    # ecg = ecg[:, np.newaxis]
    # ecg = np.expand_dims(ecg, axis=0)

    # print(len(incoming_permutation[0]))

    new_ecgs = []

    for permutation in incoming_permutation:
        new_ecg = ecg_original.copy()
        for permutation_index,item in enumerate(permutation):
            
            if item == 0:
                start_index = permutation_index*block_length
                end_index = (permutation_index+1)*block_length
                #start_index, end_index = masked_blocks[permutation_index]

                new_ecg[start_index:end_index] = [0]*(end_index-start_index)
        new_ecgs.append(new_ecg)

    new_ecgs = [np.asarray(item) for item in new_ecgs]
    new_ecgs = np.array(new_ecgs)

    new_ecgs = new_ecgs[:, :, np.newaxis]

    predictions = model.predict_proba(new_ecgs)
    for index,prediction in enumerate(predictions):
        predictions[index] = [int(value) for value in prediction]

    return predictions

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

model_location = 'saved_models\\cnn\\cnn_model'
model = tf.keras.models.load_model(model_location)

print(model.summary())

#old normal file: 77001
#old arrthmia file: 306
filename = "./mit_bih_processed_data_two_leads_r_marker/network_data/training_set/ecg_227.txt"
f = open(filename, "r")
ecg = []
for i,line in enumerate(f):
    line = line.replace("\n","")
    #ECG signal stored in first line separated by spaces
    if i < 1:
        line_segments = line.split()

        if "r_marker" in filename:
            r_value = line_segments[-1]
            del line_segments[-1]

        for i,item in enumerate(line_segments):
            line_segments[i] = float(item)

        for item in line_segments:
            ecg.append(item)
f.close()

# plt.show()
ecg_original = ecg.copy()


ecg = [np.asarray(item) for item in ecg]
ecg = np.array(ecg)

ecg = ecg[:, np.newaxis]
ecg = np.expand_dims(ecg, axis=0)

block_length = 20
#num_blocks = 860//block_length
num_blocks = 43
print(num_blocks)

masked_blocks = []
for i in range(num_blocks):
    start_index = random.randint(-10, 870) #start_index -43 and end_index + 43 to allow for the starts/ends to be covered
    end_index = start_index + 43
    if start_index < 0:
        start_index = 0
    if end_index > 860:
        end_index = 860
    masked_blocks.append([start_index, end_index])

number_of_permutations = 150

permutation_blocks = []
permutation_ecgs = []

for index in range(number_of_permutations):
    
    this_permutation = [1]*num_blocks
    new_ecg = ecg_original.copy()
    
    number_of_samples = random.randint(0, num_blocks)

    off_list = random.sample(range(0, num_blocks), number_of_samples)

    print(off_list)

    for item in off_list:
        this_permutation[item] = 0

    #this is just visualisation
    # for permutation_index,item in enumerate(this_permutation):
    #     if item == 0:
    #         #here, this is where I create the blocks
    #         start_index = permutation_index*block_length
    #         #end_index = (permutation_index+1)*block_length
    #         start_index, end_index = masked_blocks[permutation_index]

    #         new_ecg[start_index:end_index] = [0]*(end_index-start_index)

    # print(this_permutation)
    permutation_blocks.append(this_permutation)
    # permutation_ecgs.append(new_ecg)

permutation_blocks = [np.asarray(item) for item in permutation_blocks]
permutation_blocks = np.array(permutation_blocks)

# for index, new_ecg in enumerate(permutation_ecgs):
#     permutation = permutation_blocks[index]
#     plt.plot(new_ecg)
#     plt.title(permutation)
#     plt.show()

ecg_original = [np.asarray(item) for item in ecg_original]
ecg_original = np.array(ecg_original)

# explainer = lime.lime_image.LimeImageExplainer()
# explanation = explainer.explain_instance(ecg, model.predict_proba, top_labels=5, hide_color=1, num_samples=1000)

print(len(permutation_blocks))

explainer = lime.lime_tabular.LimeTabularExplainer(permutation_blocks, feature_names=np.arange(0,num_blocks,1), class_names=[0]*len(permutation_blocks), discretize_continuous=True)
exp = explainer.explain_instance(np.ones(num_blocks), predict_fn, num_features=num_blocks)

print(exp)

print("List")
print(exp.as_list())

print("Map")

this_map = exp.as_map()
results_list = this_map[1]

fig, ax = plt.subplots()
plt.plot(ecg_original)
plt.title("Abnormal ECG (atrial premature beat)")
plt.hlines(0,xmin=0,xmax=860)
plt.grid()
for i in range(num_blocks-1):
    ax.annotate("|",[block_length*(i+1),0])

for rank,item in enumerate(results_list):
    [index, importance] = item
    start = index*block_length
    end = (index+1)*block_length
    midpoint = (end-start)/2 + start

    if importance > 0:
        ax.annotate(rank+1, [midpoint, 10], color='red')
    else:
        ax.annotate(rank+1, [midpoint, 10], color='green')

#plot_features(exp, ncol = 1)

# for index, item in enumerate(masked_blocks):
#     print(str(index)+": "+str(item))


ecg_lead_1, ecg_lead_2, r_value = read_file_for_feature_extraction(filename)
ecg_plot, p_wave_start, p_wave_end, q_point, r_max_index, s_point, t_wave_start, t_wave_end = feature_extract_ecg(ecg_lead_1, r_value)

ax.axvspan(p_wave_start, p_wave_end, alpha=0.5, color='coral')
ax.axvspan(p_wave_start+430, p_wave_end+430, alpha=0.5, color='coral')

ax.axvspan(q_point, r_max_index, alpha=0.5, color='yellow')
ax.axvspan(q_point+430, r_max_index+430, alpha=0.5, color='yellow')

ax.axvspan(r_max_index, s_point, alpha=0.5, color='lightgreen')
ax.axvspan(r_max_index+430, s_point+430, alpha=0.5, color='lightgreen')

ax.axvspan(t_wave_start, t_wave_end, alpha=0.5, color='lightseagreen')
ax.axvspan(t_wave_start+430, t_wave_end+430, alpha=0.5, color='lightseagreen')

plt.annotate("P-wave",xy=(((p_wave_end-p_wave_start)/2)+p_wave_start, 600), ha='center', color='coral')
plt.annotate("Q-R",xy=(((r_max_index-q_point)/2)+q_point, 600), ha='right', color='goldenrod')
plt.annotate("R-S",xy=(((s_point-r_max_index)/2)+r_max_index, 600), color='darkgreen')
plt.annotate("T-wave",xy=(((t_wave_end-t_wave_start)/2)+t_wave_start, 600), ha='center', color='lightseagreen')

plt.annotate("P-wave",xy=(((p_wave_end-p_wave_start)/2)+p_wave_start+430, 600), ha='center', color='coral')
plt.annotate("Q-R",xy=(((r_max_index-q_point)/2)+q_point+430, 600), ha='right', color='goldenrod')
plt.annotate("R-S",xy=(((s_point-r_max_index)/2)+r_max_index+430, 600), color='darkgreen')
plt.annotate("T-wave",xy=(((t_wave_end-t_wave_start)/2)+t_wave_start+430, 600), ha='center', color='lightseagreen')

exp.as_pyplot_figure()

plt.show()

#explainer = lime_image.LimeImageExplainer()
#explanation = explainer.explain_instance(images[0], inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)
