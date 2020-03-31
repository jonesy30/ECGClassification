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

class_names = ['AFIB_AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'NSR', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']
#class_names = ['AFIB', 'AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NSR', 'SUDDEN_BRADY', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']
#class_names = ['AFIB', 'AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'SUDDEN_BRADY', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']
#class_names = ['AFIB', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'NSR', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']
#class_names = ['AFIB', 'NOISE', 'NSR']
#class_names = ['AFIB', 'NSR']
mode = "ECG" #or FEATURE

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
            if mode == "ECG":
                if i < 1:
                    line_segments = line.split()
                    for i,item in enumerate(line_segments):
                        line_segments[i] = float(item)

                    for item in line_segments:
                        found_data.append(item)
                else:
                    label_text = line
                    #if label_text != "OTHER":
                    index = class_names.index(line)
                    label = index
            elif mode == "FEATURE":
                if i < 10: #this number might be wrong! Check!
                    line_segments = line.split()
                    for i,item in enumerate(line_segments):
                        line_segments[i] = float(item)
                        found_data.append(float(line))

                    for item in line_segments:
                        found_data.append(item)
                else:
                    label_text = line
                    #if label_text != "OTHER":
                    index = class_names.index(line)
                    label = index
        f.close()

        if label != "":
            data.append(found_data)
            labels.append(label)
    
    return data, labels

(training_data, training_labels) = read_data("./split_processed_data/network_data_unfiltered/training_set/")
(validation_data, validation_labels) = read_data("./split_processed_data/network_data_unfiltered/validation_set/")

# for item in validation_data:
#     training_data.append(item)
# for item in validation_labels:
#     training_labels.append(item)

training_data = [np.asarray(item) for item in training_data]
training_data = np.array(training_data)

validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

training_labels = [np.asarray(item) for item in training_labels]
training_labels = np.array(training_labels)

validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

#This is the upsampling section
df = pd.DataFrame(training_data)
df['label'] = training_labels

max_label = df['label'].value_counts().idxmax()
max_number = df['label'].value_counts()[max_label]

df_oversampled = df[df['label']==max_label]

for value in range(df['label'].nunique()):
    if value != max_label:
        df_class = df[df['label']==value]
        df_class_over = df_class.sample(max_number, replace=True)
        df_class_over = pd.DataFrame(df_class_over)
        df_oversampled = pd.concat([df_oversampled, df_class_over])

training_labels = df_oversampled['label'].tolist()
training_labels = np.array(training_labels)

training_data = df_oversampled.drop(columns='label').to_numpy()
#training_labels = to_categorical(training_labels)   
#Aaaaand we're done upsampling! Hooray!

# model = keras.Sequential([
#     #keras.layers.Dense(6, activation=tf.nn.sigmoid),
#     #keras.layers.Dense(12, activation=tf.nn.sigmoid),
#     keras.layers.Dense(32, activation=tf.nn.relu),
#     keras.layers.Dense(32, activation=tf.nn.relu),
#     #keras.layers.Dense(32, activation=tf.nn.relu),   
#     keras.layers.Dense(len(class_names), activation
#     =tf.nn.softmax)
# ])

#opt = SGD(lr=0.1, momentum=0.9, decay=0.01)
#rms = keras.optimizers.RMSprop(learning_rate=0.1, rho=0.9)

def baseline_model():
    model = keras.Sequential()
    model.add(Dropout(0.2))

    model.add(keras.layers.Dense(256, input_dim=14, activation=tf.nn.relu, kernel_constraint=maxnorm(3)))

    for i in range(15):
        model.add(keras.layers.Dense(256, activation=tf.nn.relu, kernel_constraint=maxnorm(3)))
        #model.add(Dropout(0.2))
    model.add(keras.layers.Dense(len(class_names), activation=tf.nn.softmax))

    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer="adagrad", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=100, verbose=1)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, training_data, training_labels, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model = baseline_model()
model.fit(training_data, training_labels, epochs=100)

print("Model Summary")
print(model.summary())

# make a prediction
predicted_labels = model.predict(validation_data)
# show the inputs and predicted outputs

#validation_encoded = to_categorical(validation_labels)  

predicted_encoded = np.argmax(predicted_labels, axis=1)

matrix = confusion_matrix(validation_labels, predicted_encoded, normalize='all')
#disp = plot_confusion_matrix(model, validation_labels, predicted_encoded,display_labels=class_names)
plot_confusion_matrix(matrix, classes=class_names, normalize=True, title="Confusion Matrix (fully connected)")

plt.figure()

tested = 0
correct = 0
correct_predictions = [0]*len(class_names)
incorrect_predictions = [0]*len(class_names)

incorrectly_identified = []

for i in range(len(validation_data)):
    predicted = np.where(predicted_labels[i] == np.amax(predicted_labels[i]))
    predicted_value = predicted[0][0]

    #print("i = "+str(i)+" Label = "+str(validation_labels[i]) + ", Predicted = "+str(predicted_value))

    tested = tested + 1
    if validation_labels[i] == predicted_value:
        correct = correct + 1
        correct_predictions[validation_labels[i]] = correct_predictions[validation_labels[i]] + 1
    else:
        incorrect_predictions[validation_labels[i]] = incorrect_predictions[validation_labels[i]] + 1
        incorrectly_identified.append([validation_data[i],i])

accuracy = correct/tested
print("Accuracy = "+str(accuracy))
print("Correct matrix = "+str(correct_predictions))
print("Incorrect matrix = "+str(incorrect_predictions))

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

#print("Accuracy of predictions = "+str(accuracy_of_predictions))
plt.bar(class_names, accuracy_of_predictions)
plt.xticks(class_names, fontsize=7, rotation=30)
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100))
plt.title("Fully Connected NN\nOverall Accuracy = "+str(round(test_acc*100,2))+"%")
plt.ylabel("Accuracy of predictions (%)")
plt.xlabel("Condition")
plt.show()