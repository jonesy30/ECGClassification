import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from plot_confusion_matrix import plot_confusion_matrix

class_names = ['AFIB_AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'NSR', 'SVT', 'TRIGEMINY', 'VT', 'WENCKEBACH']

def normalize(ecg_signal):
    max_value = max(ecg_signal)
    min_value = min(ecg_signal)

    range_values = max_value - min_value

    return [(x - min_value)/range_values for x in ecg_signal]

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
                #if label_text != "OTHER":
                index = class_names.index(line)
                label = index
        f.close()

        if label != "":
            normalized_data = normalize(found_data)
            data.append(normalized_data)
            labels.append(label)
    
    return data, labels

(training_data, training_labels) = read_data("./split_processed_data/network_data_unfiltered/training_set/")
(validation_data, validation_labels) = read_data("./split_processed_data/network_data_unfiltered/validation_set/")

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
training_data = df_oversampled.drop(columns='label').to_numpy()
#Aaaaand we're done upsampling! Hooray!

training_data = training_data[:, :, np.newaxis]
training_labels = to_categorical(training_labels)

validation_data = validation_data[:, :, np.newaxis]
validation_labels = to_categorical(validation_labels)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[400,1]),
    #Dropout(0.2)
    #keras.layers.Conv1D(kernel_size=10, filters=128, strides=4, use_bias=True, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_initializer='VarianceScaling'),
    #keras.layers.AveragePooling1D(pool_size=2, strides=1,padding="same")
])

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

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(training_data,training_labels,epochs=epochs,batch_size=batch_size,verbose=verbose)

print("Model Summary")
print(model.summary())

print("Evaluating....")

test_loss, test_acc = model.evaluate(validation_data, validation_labels)
predicted_labels = model.predict(validation_data)
# show the inputs and predicted outputs

predicted_encoded = np.argmax(predicted_labels, axis=1)
actual_encoded = np.argmax(validation_labels, axis=1)

print(validation_labels[:50])
print(predicted_encoded[:50])

matrix = confusion_matrix(actual_encoded, predicted_encoded, normalize='all')
#disp = plot_confusion_matrix(model, validation_labels, predicted_encoded,display_labels=class_names)
plot_confusion_matrix(matrix, classes=class_names, normalize=True, title="Confusion Matrix (CNN)")

plt.figure()

tested = 0
correct = 0
correct_predictions = [0]*len(class_names)
incorrect_predictions = [0]*len(class_names)

predicted_values = []

incorrectly_identified = []

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

#print("Accuracy of predictions = "+str(accuracy_of_predictions))
plt.bar(class_names, accuracy_of_predictions)
plt.xticks(class_names, fontsize=7, rotation=30)
plt.title("CNN\nOverall Accuracy = "+str(round(test_acc*100,2))+"%")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100))
plt.ylabel("Accuracy of predictions (%)")
plt.xlabel("Condition")
plt.show()
