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
from analyse_ml_results import analyse_results
import tensorflow as tf

class_names = ['A','E','j','L','N','P','R','V']
two_leads = 0

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

def normalize(ecg_signal, filename):
    max_value = max(ecg_signal)
    min_value = min(ecg_signal)

    range_values = max_value - min_value

    # if range_values == 0:
    #     print(max_value)
    #     print(min_value)
    #     print(filename)

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

base_filename = "./mit_bih_processed_data_two_leads/network_data/"
(training_data, training_labels) = read_data(base_filename + "training_set/",save_unnormalised=False)

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

if two_leads == 0:
    input_shape = 430
else:
    input_shape = 860

#embedding_vecor_length = 500
model = Sequential()
#model.add(Embedding(200, embedding_vecor_length, input_length=max_length))
#model.add(Bidirectional(LSTM(64, return_sequences=False, input_shape=(input_shape, 1))))
model.add(LSTM(64, return_sequences=False, input_shape=(input_shape, 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(60))
# model.add(Dropout(0.2))
model.add(Dense(32, activation=tf.nn.relu))
model.add(Dense(len(class_names), activation='softmax'))

epochs = 40
batch_size = 128

print("Epochs = "+str(epochs))

model.compile(loss=focal_loss(alpha=1), optimizer='nadam', metrics=['accuracy'])
history = model.fit(training_data, training_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

print("History Keys")
print(history.history.keys())

print("Model Summary")
print(model.summary())

print("Evaluating....")

del training_data
del training_labels

([validation_data,unnormalised_validation], validation_labels) = read_data(base_filename + "/validation_set/",save_unnormalised=True)
print("Finished reading validation file")

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
    
if not os.path.exists("./saved_models/"):
    os.makedirs("./saved_models/")
if not os.path.exists("./saved_models/lstm/"):
    os.makedirs("./saved_models/lstm/")

model.save(".\\saved_models\\lstm\\lstm_model")

analyse_results(history, validation_data, validation_labels, predicted_labels, "lstm", base_filename, unnormalised_validation, test_acc)
