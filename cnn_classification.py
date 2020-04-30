"""
File which uses a convolutional neural network to classify ECG signals into classes of heart condition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from plot_confusion_matrix import plot_confusion_matrix
import time
from classification_report import plot_classification_report
from visualise_incorrect_predictions import save_incorrect_predictions
from analyse_ml_results import analyse_results

#class_names = ['AFIB_AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'NSR', 'SVT', 'TRIGEMINY', 'WENCKEBACH']
class_names = ['A','E','j','L','N','P','R','V']
labels = ["APB","Vesc","Jesc","LBBB","Normal","Paced","RBBB","VT"]

label_names = {'N':'Normal','L':'LBBB','R':'RBBB','A':'APB','a':'AAPB','J':'JUNCTIONAL','S':'SP','V':'VT','r':'RonT','e':'Aesc','j':'Jesc','n':'SPesc','E':'Vesc','P':'Paced'}

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

#Read the training and validation data and labels and store in arrays
#(training_data, training_labels) = read_data("./split_processed_data/network_data_unfiltered/training_set/")
#(validation_data, validation_labels) = read_data("./split_processed_data/network_data_unfiltered/validation_set/")

start_time = time.time()

base_filename = "./mit_bih_processed_data_two_leads/"

(training_data, training_labels) = read_data(base_filename + "network_data/training_set/")

#Turn each training data array into numpy arrays of numpy arrays
training_data = [np.asarray(item) for item in training_data]
training_data = np.array(training_data)

#Turn training labels into element arrays of 1x1 element arrays (each containing a label)
training_labels = [np.asarray(item) for item in training_labels]
training_labels = np.array(training_labels)

#Upsample the data to amplify lesser classes
df = pd.DataFrame(training_data)
df['label'] = training_labels

#Get the size of the largest class (so I know how much to upsample by)
max_label = df['label'].value_counts().idxmax()
max_number = df['label'].value_counts()[max_label]

#Create an upsampling space, initially fill it with the largest category
df_oversampled = df[df['label']==max_label]

#For each smaller class, oversample it and add to the oversampling space
for value in range(df['label'].nunique()):
    if value != max_label:
        df_class = df[df['label']==value]
        df_class_over = df_class.sample(max_number, replace=True)
        df_class_over = pd.DataFrame(df_class_over)
        df_oversampled = pd.concat([df_oversampled, df_class_over])

#Convert the upsampled data to training and labelled data
training_labels = df_oversampled['label'].tolist()
training_data = df_oversampled.drop(columns='label').to_numpy()
#Aaaaand we're done upsampling! Hooray!

#Resize training data to fit CNN input layer and convert labels to one-hot encoding
training_data = training_data[:, :, np.newaxis]
training_labels = to_categorical(training_labels)

input_size = 860 #400 for other dataset

#Build the intial model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[input_size,1])
    #keras.layers.Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32))
    #Dropout(0.2)
    #keras.layers.Conv1D(kernel_size=10, filters=128, strides=4, use_bias=True, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_initializer='VarianceScaling'),
    #keras.layers.AveragePooling1D(pool_size=2, strides=1,padding="same")
])
#Add extra labels

model.add(keras.layers.Conv1D(kernel_size=10, filters=64, strides=4, input_shape=(input_size,1), use_bias=True, kernel_initializer='VarianceScaling'))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.AveragePooling1D(pool_size=2, strides=1, padding="same"))
model.add(Dropout(0.5))

model.add(keras.layers.Conv1D(kernel_size=10, filters=64, strides=4, use_bias=True, kernel_initializer='VarianceScaling'))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.AveragePooling1D(pool_size=2, strides=1, padding="same"))
model.add(Dropout(0.5))

#model.add(keras.layers.AveragePooling1D(pool_size=2, strides=1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(len(class_names), activation='softmax'))

#MAGIC NUMBERS
verbose = 1
epochs = 30
batch_size = 100

#Build and fit the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(training_data,training_labels,epochs=epochs,batch_size=batch_size,validation_split=0.1,verbose=verbose)
#history = model.fit(training_data,training_labels,epochs=epochs,batch_size=batch_size,verbose=verbose)

print("History Keys")
print(history.history.keys())

print("Model Summary")
print(model.summary())

print("Evaluating....")

#Clear training data - we don't need this any more
del training_data
del training_labels

([validation_data,unnormalised_validation], validation_labels) = read_data(base_filename + "network_data/validation_set/",save_unnormalised=True)
print("Finished reading validation file")

#Turn each validation data array into numpy arrays of numpy arrays
validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#Turn validation labels into element arrays of 1x1 element arrays (each containing a label)
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

#Resize validation data to fit CNN input layer and convert labels to one-hot encoding
validation_data = validation_data[:, :, np.newaxis]
validation_labels = to_categorical(validation_labels)

test_loss, test_acc = model.evaluate(validation_data, validation_labels)
predicted_labels = model.predict(validation_data)
    
end_time = time.time()
print("Time for "+str(epochs)+" epochs = "+str(end_time-start_time))

if not os.path.exists("./saved_models/"):
    os.makedirs("./saved_models/")
if not os.path.exists("./saved_models/CNN/"):
    os.makedirs("./saved_models/CNN/")

model.save(".\\saved_models\\cnn\\cnn_model")

analyse_results(history, validation_data, validation_labels, predicted_labels, "cnn", base_filename, unnormalised_validation, test_acc)
