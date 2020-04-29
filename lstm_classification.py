"""
Initial experiment with LSTMs, keeping file for archiving purposes
"""

import random
import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
import pandas as pd
from tensorflow.keras.utils import to_categorical
from classification_report import plot_classification_report

class_names = ['A','E','j','L','N','P','R','V']

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

    return [(x - min_value)/range_values for x in ecg_signal]

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
                if str(line) not in class_names:
                    label = ""
                else:
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

base_filename = "./mit_bih_subset/network_data/"
(training_data, training_labels) = read_data(base_filename + "training_set/",save_unnormalised=False)
(validation_data, validation_labels) = read_data(base_filename + "validation_set/",save_unnormalised=False)

#Turn each training data array into numpy arrays of numpy arrays
training_data = [np.asarray(item) for item in training_data]
training_data = np.array(training_data)

#Turn training labels into element arrays of 1x1 element arrays (each containing a label)
training_labels = [np.asarray(item) for item in training_labels]
training_labels = np.array(training_labels)

#Resize training data to fit CNN input layer and convert labels to one-hot encoding
training_data = training_data[:, :, np.newaxis]
training_labels = to_categorical(training_labels,num_classes=len(class_names))

#NOTE: Still to do oversampling

#Turn each validation data array into numpy arrays of numpy arrays
validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#Turn validation labels into element arrays of 1x1 element arrays (each containing a label)
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

#Resize validation data to fit CNN input layer and convert labels to one-hot encoding
validation_data = validation_data[:, :, np.newaxis]
validation_labels = to_categorical(validation_labels,num_classes=len(class_names))

#embedding_vecor_length = 500
model = Sequential()
#model.add(Embedding(200, embedding_vecor_length, input_length=max_length))
model.add(Bidirectional(LSTM(100, return_sequences=False, input_shape=(2600, 1))))
# model.add(Dropout(0.2))
# model.add(LSTM(60))
# model.add(Dropout(0.2))
model.add(Dense(len(class_names), activation='sigmoid'))

epochs = 1

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_data, training_labels, epochs=epochs, batch_size=200)
print(model.summary())
# Final evaluation of the model
scores = model.evaluate(validation_data, validation_labels, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

predictions = model.predict(validation_data)

one_hot_predictions = []
for value in predictions:
    one_hot_predictions.append(round(value[0]))

df = pd.DataFrame(one_hot_predictions)
print(df[0].value_counts())

predicted_encoded = np.argmax(predictions, axis=1)
actual_encoded = np.argmax(validation_labels, axis=1)

recall, precision, f1_score = plot_classification_report(actual_encoded, predicted_encoded, classes=class_names)
print("Recall = "+str(recall))
print("Precision = "+str(precision))
print("F1 score = "+str(f1_score))