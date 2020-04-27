
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
#from cnn_classification import read_data

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

model_location = 'saved_models\\cnn\\cnn_model'
new_model = tf.keras.models.load_model(model_location)

print(new_model.summary())

base_filename = "./external_validation_data/mit_bih_nsr_subset/"
#base_filename + "network_data/validation_set/"
#"hannun_validation_data/""
(validation_data, validation_labels) = read_data(base_filename,save_unnormalised=False)
#print(len(validation_data))
#validation_data = np.reshape(validation_data,(2600,len(validation_data)))

validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#Turn validation labels into element arrays of 1x1 element arrays (each containing a label)
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

#Resize validation data to fit CNN input layer and convert labels to one-hot encoding

if "cnn" in model_location:
    validation_data = validation_data[:, :, np.newaxis]
    validation_labels = to_categorical(validation_labels,num_classes=len(class_names))

loss, acc = new_model.evaluate(validation_data,  validation_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

#print(new_model.predict(test_images).shape)