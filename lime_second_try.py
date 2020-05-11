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

class_names = ['A','E','j','L','N','P','R','V']
two_leads = 0

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
def read_data(foldername,save_unnormalised=False):
    
    data = []
    labels = []
    unnormalised = []

    #for each file in corresponding folder
    for file in os.listdir(foldername):
        f = open(str(foldername+file), "r")
        found_data = []
        label = ""
        label_text = ""
        for i,line in enumerate(f):
            line = line.replace("\n","")
            #ECG signal stored in first line separated by spaces
            if i < 1:
                line_segments = line.split()

                if two_leads == 0:
                    line_segments = line_segments[:430]
                line_segments = [float(x) for x in line_segments]

                for item in line_segments:
                    found_data.append(item)
            #label stored on second line
            else:
                label_text = line
                #if label_text != "OTHER":
                index = class_names.index(line)
                label = index
        f.close()

        #if label exists, store in trainng validation data
        if label != "":
            unnormalised.append(found_data)
            normalized_data = normalize(found_data, file)
            data.append(normalized_data)
            labels.append(label)
    
    if save_unnormalised == True:
        return [data, unnormalised], labels

    return data, labels

model_location = 'saved_models\\cnn\\cnn_model'
model = tf.keras.models.load_model(model_location)

print(model.summary())

#old normal file: 77001
f = open("./mit_bih_processed_data_two_leads/network_data/training_set/ecg_306.txt", "r")
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

# plt.show()
ecg_original = ecg.copy()

ecg = [np.asarray(item) for item in ecg]
ecg = np.array(ecg)

ecg = ecg[:, np.newaxis]
ecg = np.expand_dims(ecg, axis=0)

block_length = 43
num_blocks = 860//block_length
print(num_blocks)

number_of_permutations = 150

permutation_blocks = []
permutation_ecgs = []

for index in range(number_of_permutations):
    
    this_permutation = [1]*num_blocks
    new_ecg = ecg_original.copy()
    
    number_of_samples = random.randint(0, num_blocks)

    off_list = random.sample(range(0, num_blocks), number_of_samples)
    for item in off_list:
        this_permutation[item] = 0

    for permutation_index,item in enumerate(this_permutation):
        if item == 0:
            start_index = permutation_index*block_length
            end_index = (permutation_index+1)*block_length

            new_ecg[start_index:end_index] = [0]*(end_index-start_index)
    
    print(this_permutation)
    permutation_blocks.append(this_permutation)
    permutation_ecgs.append(new_ecg)

permutation_blocks = [np.asarray(item) for item in permutation_blocks]
permutation_blocks = np.array(permutation_blocks)

# for index, new_ecg in enumerate(permutation_ecgs):
#     permutation = permutation_blocks[index]
#     plt.plot(new_ecg)
#     plt.title(permutation)
#     plt.show()

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

                new_ecg[start_index:end_index] = [0]*(end_index-start_index)
        new_ecgs.append(new_ecg)

    new_ecgs = [np.asarray(item) for item in new_ecgs]
    new_ecgs = np.array(new_ecgs)

    new_ecgs = new_ecgs[:, :, np.newaxis]

    predictions = model.predict_proba(new_ecgs)
    for index,prediction in enumerate(predictions):
        predictions[index] = [int(value) for value in prediction]
    
    return predictions

ecg_original = [np.asarray(item) for item in ecg_original]
ecg_original = np.array(ecg_original)

# explainer = lime.lime_image.LimeImageExplainer()
# explanation = explainer.explain_instance(ecg, model.predict_proba, top_labels=5, hide_color=1, num_samples=1000)

print(len(permutation_blocks))

explainer = lime.lime_tabular.LimeTabularExplainer(permutation_blocks, feature_names=np.arange(0,block_length,1), class_names=[0]*len(permutation_blocks), discretize_continuous=True)
exp = explainer.explain_instance(np.ones(num_blocks), predict_fn, num_features=num_blocks)

print(exp)

print("List")
print(exp.as_list())

print("Map")

this_map = exp.as_map()
results_list = this_map[1]

fig, ax = plt.subplots()
plt.plot(ecg_original)
plt.title("Abnormal ECG (AFIB)")
plt.hlines(0,xmin=0,xmax=860)
plt.grid()
for i in range(num_blocks-1):
    ax.annotate("|",[block_length*(i+1),0])

for rank,item in enumerate(results_list):
    [index, importance] = item
    start = index*43
    end = (index+1)*43
    midpoint = (end-start)/2 + start

    if importance > 0:
        ax.annotate(rank+1, [midpoint, 10], color='red')
    else:
        ax.annotate(rank+1, [midpoint, 10], color='green')

#plot_features(exp, ncol = 1)
exp.as_pyplot_figure()
plt.show()

#explainer = lime_image.LimeImageExplainer()
#explanation = explainer.explain_instance(images[0], inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)
