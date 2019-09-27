from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import SGD

#class_names = ['AFIB', 'AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'NSR', 'SUDDEN_BRADY', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']
class_names = ['AFIB', 'AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NSR', 'SUDDEN_BRADY', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']
#class_names = ['AFIB', 'AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'SUDDEN_BRADY', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']
class_names = ['AFIB', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']


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
            if i < 1:
                line_segments = line.split()
                for i,item in enumerate(line_segments):
                    line_segments[i] = float(item)

                for item in line_segments:
                    found_data.append(item)
            else:
                label_text = line
                if label_text != "AFL" and label_text != "SUDDEN_BRADY" and label_text != "NSR":
                    index = class_names.index(line)
                    label = index
        f.close()

        if label != "":
            data.append(found_data)
            labels.append(label)
    
    return data, labels

(training_data, training_labels) = read_data("./processed_data/network_data_ecg_samples/training_set/")
(validation_data, validation_labels) = read_data("./processed_data/network_data_ecg_samples/validation_set/")

# for item in validation_data:
#     training_data.append(item)
# for item in validation_labels:
#     training_labels.append(item)

for i,item in enumerate(training_data):
    training_data[i] = np.asarray(item)

training_data = np.array(training_data)

for i,item in enumerate(validation_data):
    validation_data[i] = np.asarray(item)

validation_data = np.array(validation_data)

for i,item in enumerate(training_labels):
    training_labels[i] = np.asarray(item)

training_labels = np.array(training_labels)

for i,item in enumerate(validation_labels):
    validation_labels[i] = np.asarray(item)

validation_labels = np.array(validation_labels)

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])

rms = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=1000)

# make a prediction
predicted_data = model.predict(validation_data)
# show the inputs and predicted outputs

tested = 0
correct = 0
correct_predictions = [0]*len(class_names)
incorrect_predictions = [0]*len(class_names)

for i in range(len(validation_data)):
    predicted = np.where(predicted_data[i] == np.amax(predicted_data[i]))
    predicted_value = predicted[0][0]

    print("i = "+str(i)+" Label = "+str(validation_labels[i]) + ", Predicted = "+str(predicted_value))

    tested = tested + 1
    if validation_labels[i] == predicted_value:
        correct = correct + 1
        correct_predictions[validation_labels[i]] = correct_predictions[validation_labels[i]] + 1
    else:
        incorrect_predictions[validation_labels[i]] = incorrect_predictions[validation_labels[i]] + 1

accuracy = correct/tested
print("Accuracy = "+str(accuracy))
print("Correct matrix = "+str(correct_predictions))
print("Incorrect matrix = "+str(incorrect_predictions))

accuracy_of_predictions = [0]*len(class_names)
for i,item in enumerate(correct_predictions):
    total_labels = correct_predictions[i] + incorrect_predictions[i]
    accuracy_of_predictions[i] = correct_predictions[i]/total_labels*100

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
plt.ylabel("Accuracy of predictions (%)")
plt.xlabel("Condition")
plt.show()