"""
File which uses a fully connected deep neural network to classify ECG signals (or feature extracted data) into classes of heart condition
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from keras.constraints import maxnorm
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import itertools
from plot_confusion_matrix import plot_confusion_matrix
import time
from classification_report import plot_classification_report
from visualise_incorrect_predictions import save_incorrect_predictions
from analyse_ml_results import analyse_results

#class_names = ['AFIB_AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'NSR', 'SVT', 'TRIGEMINY', 'WENCKEBACH']
mode = "ECG" #or FEATURE

class_names = ['A','E','j','L','N','P','R','V']
labels = ["APB","Vesc","Jesc","LBBB","Normal","Paced","RBBB","VT"]

#Function which normalizes the ECG signal
def normalize(ecg_signal, filename):
    max_value = max(ecg_signal)
    min_value = min(ecg_signal)

    range_values = max_value - min_value

    if range_values == 0:
        #print(max_value)
        #print(min_value)
        print(filename)

    if range_values == 0:
        return ecg_signal

    return [(x - min_value)/range_values for x in ecg_signal]

#Function which reads the data from each file in directory and splits into data/labels
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
                for i,item in enumerate(line_segments):
                    line_segments[i] = float(item)

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

start_time = time.time()

#get the training data and labels

base_filename = "./mit_bih_processed_data_two_leads_subset/"
(training_data, training_labels) = read_data(base_filename+"network_data/validation_set/")

#format the training data into a numpy array of numpy arrays
training_data = [np.asarray(item) for item in training_data]
training_data = np.array(training_data)

#format the training labels into a numpy array
training_labels = [np.asarray(item) for item in training_labels]
training_labels = np.array(training_labels)

#Upsample the data to amplify lesser classes
df = pd.DataFrame(training_data)
df['label'] = training_labels

#Find the class with the maximum number of samples
max_label = df['label'].value_counts().idxmax()
max_number = df['label'].value_counts()[max_label]

#Create dataframe to store oversampled data, and set initially to the largest class
df_oversampled = df[df['label']==max_label]

#for each lesser class, upsample to match size of largest class and append to oversampled dataframe
for value in range(df['label'].nunique()):
    if value != max_label:
        df_class = df[df['label']==value]
        df_class_over = df_class.sample(max_number, replace=True)
        df_class_over = pd.DataFrame(df_class_over)
        df_oversampled = pd.concat([df_oversampled, df_class_over])

#Split oversampled dataframe into training data and labels
training_labels = df_oversampled['label'].tolist()
training_labels = np.array(training_labels)

training_data = df_oversampled.drop(columns='label').to_numpy()
#Aaaaand we're done upsampling! Hooray!

input_size = 2600
input_shape = training_data[0].shape

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=input_shape)
])

model.add(keras.layers.Dense(256, activation=tf.nn.relu, input_shape=input_shape, kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

#Add dense layers
for i in range(15):
    model.add(keras.layers.Dense(256, activation=tf.nn.relu, kernel_constraint=maxnorm(3)))
    #model.add(Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(len(class_names), activation=tf.nn.softmax))

#add optimizer and compile the model
adam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer="adagrad", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

input_shape = training_data[0].shape  
model.build(input_shape)

epochs = 100

#train the model with the training data and labels
history = model.fit(training_data, training_labels, validation_split=0.1, epochs=epochs)
#history = model.fit(training_data, training_labels, epochs=100)

print("History Keys")
print(history.history.keys())

print("Model Summary")
print(model.summary())

del training_data
del training_labels

([validation_data,unnormalised_validation], validation_labels) = read_data(base_filename + "network_data/validation_set/",save_unnormalised=True)

print("Finished reading validation, evaluating...")

#format the validation data into a numpy array of numpy arrays
validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#format the validation labels into a numpy array
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

# make a prediction
test_loss, test_acc = model.evaluate(validation_data, validation_labels)
predicted_labels = model.predict(validation_data)
# show the inputs and predicted outputs

predicted_encoded = np.argmax(predicted_labels, axis=1)

end_time = time.time()
print("Time for "+str(epochs)+" epochs = "+str(end_time-start_time))

if not os.path.exists("./saved_models/"):
    os.makedirs("./saved_models/")
if not os.path.exists("./saved_models/fully_connected/"):
    os.makedirs("./saved_models/fully_connected/")

model.save(".\\saved_models\\fully_connected\\fully_connected_model")

analyse_results(history, validation_data, validation_labels, predicted_labels, "fully_connected", base_filename, unnormalised_validation, test_acc)