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
#Function which creates the model and returns it
def baseline_model():
    #Create model
    model = keras.Sequential()
    keras.layers.InputLayer(input_shape=[input_size,1])
    model.add(Dropout(0.2))
    model.add(keras.layers.Dense(256, input_shape = (2600,1), activation=tf.nn.relu, kernel_constraint=maxnorm(3)))

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

predicted_values = []
incorrectly_identified_ecgs = []
incorrectly_identified_predicted_labels = []
incorrectly_identified_true_labels = []

#Format the predictions into incorrect and correct predictions
for i in range(len(validation_data)):
    predicted = np.where(predicted_labels[i] == np.amax(predicted_labels[i]))
    predicted_value = predicted[0][0]
    predicted_values.append(predicted_value)

    actual = validation_labels[i]

    tested = tested + 1
    if actual == predicted_value:
        correct = correct + 1
        correct_predictions[actual] = correct_predictions[actual] + 1
    else:
        incorrect_predictions[actual] = incorrect_predictions[actual] + 1
        
        incorrectly_identified_ecgs.append(unnormalised_validation[i])
        incorrectly_identified_predicted_labels.append(class_names[predicted_value])
        incorrectly_identified_true_labels.append(class_names[actual])

save_incorrect_predictions(incorrectly_identified_ecgs, incorrectly_identified_predicted_labels, incorrectly_identified_true_labels, base_filename+"/fully_connected/")

#Print evaluations
accuracy = correct/tested
print("Accuracy = "+str(accuracy))
print("Correct matrix = ")
print(correct_predictions)
print("Incorrect matrix = ")
print(incorrect_predictions)

predicted_values = pd.DataFrame(predicted_values,columns=['predicted'])
print("Predicted Value Count Check")
print(predicted_values['predicted'].value_counts())
print()

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

print("Test accuracy: "+str(test_acc))
end_time = time.time()
print("Time for "+str(epochs)+" epochs = "+str(end_time-start_time))

print(predicted_labels[:5])
print(validation_labels[:5])

predicted_encoded = np.argmax(predicted_labels, axis=1)
#actual_encoded = np.argmax(validation_labels, axis=1)

#Plot the confusion matrix of the expected and predicted classes
matrix = confusion_matrix(validation_labels, predicted_encoded, normalize='all')
plot_confusion_matrix(matrix, classes=labels, normalize=True, title="Confusion Matrix (fully connected), Accuracy = "+str(round(test_acc*100,2))+"%")

plot_classification_report(validation_labels, predicted_encoded, labels, show_plot=False)

plt.figure()

if not os.path.exists("./saved_models/"):
    os.makedirs("./saved_models/")
if not os.path.exists("./saved_models/fully_connected/"):
    os.makedirs("./saved_models/fully_connected/")

model.save(".\\saved_models\\fully_connected\\fully_connected_model")

#Plot prediction accuracy percentages
plt.bar(class_names, accuracy_of_predictions)
plt.xticks(class_names, fontsize=7, rotation=30)
plt.title("Fully Connected\nOverall Accuracy = "+str(round(test_acc*100,2))+"%")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100))
plt.ylabel("Accuracy of predictions (%)")
plt.xlabel("Condition")
plt.show()