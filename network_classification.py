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
def read_data(filename):
    data = []
    labels = []
    for file in os.listdir(filename):
        f = open(str(filename+file), "r")
        found_data = []
        label = ""
        label_text = ""
        for i,line in enumerate(f):
            line = line.replace("\n","")
            #Function supports two modes - raw signal classification and feature extracted classification. This is raw ECG
            if mode == "ECG":
                #The ECG signal is stored on line 1 separated by spaces, split data and save
                if i < 1:
                    line_segments = line.split()
                    for i,item in enumerate(line_segments):
                        line_segments[i] = float(item)

                    for item in line_segments:
                        found_data.append(item)
                #Second line is the label
                else:
                    label_text = line
                    #if label_text != "OTHER":
                    index = class_names.index(line)
                    label = index
            #Function supports two modes - raw signal classification and feature extracted classification. This is feature extraction
            elif mode == "FEATURE":
                #features stored in first 10 lines
                if i < 10: #this number might be wrong! Check!
                    line_segments = line.split()
                    for i,item in enumerate(line_segments):
                        line_segments[i] = float(item)
                        found_data.append(float(line))

                    for item in line_segments:
                        found_data.append(item)
                #final line is the label
                else:
                    label_text = line
                    #if label_text != "OTHER":
                    index = class_names.index(line)
                    label = index
        f.close()

        #If label exists, add data to the data/labels array
        if label != "":
            normalized_data = normalize(found_data, file)
            data.append(normalized_data)
            labels.append(label)
    
    return data, labels

start_time = time.time()

#get the training data and labels
(training_data, training_labels) = read_data("./mit_bih_processed_data/network_data/training_set/")
(validation_data, validation_labels) = read_data("./mit_bih_processed_data/network_data/validation_set/")

#format the training data into a numpy array of numpy arrays
training_data = [np.asarray(item) for item in training_data]
training_data = np.array(training_data)

#format the validation data into a numpy array of numpy arrays
validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#format the training labels into a numpy array
training_labels = [np.asarray(item) for item in training_labels]
training_labels = np.array(training_labels)

#format the validation labels into a numpy array
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

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

#Function which creates the model and returns it
def baseline_model():
    #Create model
    model = keras.Sequential()
    model.add(Dropout(0.2))
    model.add(keras.layers.Dense(256, input_dim=14, activation=tf.nn.relu, kernel_constraint=maxnorm(3)))

    #Add dense layers
    for i in range(15):
        model.add(keras.layers.Dense(256, activation=tf.nn.relu, kernel_constraint=maxnorm(3)))
        #model.add(Dropout(0.2))
    model.add(keras.layers.Dense(len(class_names), activation=tf.nn.softmax))

    #add optimizer and compile the model
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer="adagrad", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = baseline_model()

epochs = 100

#train the model with the training data and labels
history = model.fit(training_data, training_labels, validation_split=0.1, epochs=epochs)
#history = model.fit(training_data, training_labels, epochs=100)

print("History Keys")
print(history.history.keys())

print("Model Summary")
print(model.summary())

# make a prediction
predicted_labels = model.predict(validation_data)
# show the inputs and predicted outputs

predicted_encoded = np.argmax(predicted_labels, axis=1)

#Create a confusion matrix and display
matrix = confusion_matrix(validation_labels, predicted_encoded, normalize='all')
plot_confusion_matrix(matrix, classes=labels, normalize=True, title="Confusion Matrix (fully connected)")

plt.figure()

#plot accuracy for accuracy
if 'accuracy' in history.history.keys():
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
elif 'acc' in history.history.keys():
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
plt.title('Model Accuracy (fully connected)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.figure()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (fully connected)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.figure()

#Sort the predictions into correctly and incorrectly identified to find the accuracy of each class
tested = 0
correct = 0
correct_predictions = [0]*len(class_names)
incorrect_predictions = [0]*len(class_names)

incorrectly_identified = []

for i in range(len(validation_data)):
    predicted = np.where(predicted_labels[i] == np.amax(predicted_labels[i]))
    predicted_value = predicted[0][0]

    tested = tested + 1
    if validation_labels[i] == predicted_value:
        correct = correct + 1
        correct_predictions[validation_labels[i]] = correct_predictions[validation_labels[i]] + 1
    else:
        incorrect_predictions[validation_labels[i]] = incorrect_predictions[validation_labels[i]] + 1
        incorrectly_identified.append([validation_data[i],i])

#Print, format and plot results
accuracy = correct/tested
print("Accuracy = "+str(accuracy))

accuracy_of_predictions = [0]*len(class_names)
for i,item in enumerate(correct_predictions):
    total_labels = correct_predictions[i] + incorrect_predictions[i]
    if total_labels!=0:
        accuracy_of_predictions[i] = correct_predictions[i]/total_labels*100
    else:
        accuracy_of_predictions[i] = np.nan

accuracy_of_predictions.append(accuracy*100)

for i,item in enumerate(class_names):
    total = correct_predictions[i] + incorrect_predictions[i]
    class_names[i] = class_names[i] + " ("+str(total)+")"
class_names.append("TOTAL")

test_loss, test_acc = model.evaluate(validation_data, validation_labels)
print("Test accuracy: "+str(test_acc))
end_time = time.time()
print("Time for "+str(epochs)+" epochs = "+str(end_time-start_time))

plt.bar(class_names, accuracy_of_predictions)
plt.xticks(class_names, fontsize=7, rotation=30)
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100))
plt.title("Fully Connected NN\nOverall Accuracy = "+str(round(test_acc*100,2))+"%")
plt.ylabel("Accuracy of predictions (%)")
plt.xlabel("Condition")
plt.show()