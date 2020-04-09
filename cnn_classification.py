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

#class_names = ['AFIB_AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'NSR', 'SVT', 'TRIGEMINY', 'WENCKEBACH']
class_names = ['A','E','j','L','N','P','R','V']

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

    return [(x - min_value)/range_values for x in ecg_signal]

#Function which reads ECG data and labels from each file in folder
def read_data(foldername):
    
    data = []
    labels = []
    
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
            normalized_data = normalize(found_data, file)
            data.append(normalized_data)
            labels.append(label)
    
    return data, labels

#Read the training and validation data and labels and store in arrays
#(training_data, training_labels) = read_data("./split_processed_data/network_data_unfiltered/training_set/")
#(validation_data, validation_labels) = read_data("./split_processed_data/network_data_unfiltered/validation_set/")

(training_data, training_labels) = read_data("./mit_bih_processed_data/network_data/training_set/")
(validation_data, validation_labels) = read_data("./mit_bih_processed_data/network_data/validation_set/")

#Turn each training data array into numpy arrays of numpy arrays
training_data = [np.asarray(item) for item in training_data]
training_data = np.array(training_data)

#Turn each validation data array into numpy arrays of numpy arrays
validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#Turn training labels into element arrays of 1x1 element arrays (each containing a label)
training_labels = [np.asarray(item) for item in training_labels]
training_labels = np.array(training_labels)

#Turn validation labels into element arrays of 1x1 element arrays (each containing a label)
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

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

#Resize validation data to fit CNN input layer and convert labels to one-hot encoding
validation_data = validation_data[:, :, np.newaxis]
validation_labels = to_categorical(validation_labels)

input_size = 1300 #400 for other dataset

#Build the intial model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[1300,1])
    #keras.layers.Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32))
    #Dropout(0.2)
    #keras.layers.Conv1D(kernel_size=10, filters=128, strides=4, use_bias=True, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_initializer='VarianceScaling'),
    #keras.layers.AveragePooling1D(pool_size=2, strides=1,padding="same")
])

#Add extra labels
for i in range(2):
    model.add(keras.layers.Conv1D(kernel_size=10, filters=64, strides=4, use_bias=True, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_initializer='VarianceScaling'))
    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=1, padding="same"))
    model.add(Dropout(0.5))
    #model.add(keras.layers.UpSampling1D(size=5))

#model.add(keras.layers.AveragePooling1D(pool_size=2, strides=1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(len(class_names), activation='softmax'))

#MAGIC NUMBERS
verbose = 1
epochs = 20
batch_size = 100

#Build and fit the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(training_data,training_labels,epochs=epochs,batch_size=batch_size,validation_split=0.25,verbose=verbose)
#history = model.fit(training_data,training_labels,epochs=epochs,batch_size=batch_size,verbose=verbose)

print("History Keys")
print(history.history.keys())

print("Model Summary")
print(model.summary())

print("Evaluating....")

test_loss, test_acc = model.evaluate(validation_data, validation_labels)
predicted_labels = model.predict(validation_data)
# show the inputs and predicted outputs

predicted_encoded = np.argmax(predicted_labels, axis=1)
actual_encoded = np.argmax(validation_labels, axis=1)

#Plot the confusion matrix of the expected and predicted classes
matrix = confusion_matrix(actual_encoded, predicted_encoded, normalize='all')
plot_confusion_matrix(matrix, classes=class_names, normalize=True, title="Confusion Matrix (CNN)")

plt.figure()

#plot accuracy for loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy (CNN)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.figure()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (CNN)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.figure()

tested = 0
correct = 0
correct_predictions = [0]*len(class_names)
incorrect_predictions = [0]*len(class_names)

predicted_values = []
incorrectly_identified = []

#Format the predictions into incorrect and correct predictions
for i in range(len(validation_data)):
    predicted = np.where(predicted_labels[i] == np.amax(predicted_labels[i]))
    predicted_value = predicted[0][0]
    predicted_values.append(predicted_value)

    actual = list(validation_labels[i]).index(1)

    tested = tested + 1
    if actual == predicted_value:
        correct = correct + 1
        correct_predictions[actual] = correct_predictions[actual] + 1
    else:
        incorrect_predictions[actual] = incorrect_predictions[actual] + 1
        incorrectly_identified.append([validation_data[i],i])

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

for item in incorrectly_identified:
    [data, index] = item

#Plot prediction accuracy percentages
plt.bar(class_names, accuracy_of_predictions)
plt.xticks(class_names, fontsize=7, rotation=30)
plt.title("CNN\nOverall Accuracy = "+str(round(test_acc*100,2))+"%")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100))
plt.ylabel("Accuracy of predictions (%)")
plt.xlabel("Condition")
plt.show()
