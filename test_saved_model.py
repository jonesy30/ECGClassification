
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from analyse_ml_results import analyse_results
#from cnn_classification import read_data

class_names = ['A','E','j','L','N','P','R','V']
two_leads = 1

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
        for i,line in enumerate(f):
            line = line.replace("\n","")
            #ECG signal stored in first line separated by spaces
            if i < 1:
                line_segments = line.split()
                if two_leads == 0:
                    line_segments = line_segments[:430]
                for i,item in enumerate(line_segments):
                    line_segments[i] = float(item)

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

model_location = 'saved_models\\lstm\\lstm_model'
new_model = tf.keras.models.load_model(model_location, custom_objects={'focal_loss_fixed': focal_loss()})

print(new_model.summary())

#base_filename = "./external_validation_data/st_petersburg_downsampled/network_data/complete_set/"

base_filename = "./mit_bih_processed_data_two_leads/network_data/validation_set/"

#base_filename + "network_data/validation_set/"
#"hannun_validation_data/""
([validation_data,unnormalised_validation], validation_labels) = read_data(base_filename,save_unnormalised=True)
#print(len(validation_data))
#validation_data = np.reshape(validation_data,(2600,len(validation_data)))

print("Finished reading validation data")

validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#Turn validation labels into element arrays of 1x1 element arrays (each containing a label)
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

#Resize validation data to fit CNN input layer and convert labels to one-hot encoding

if ("cnn" in model_location) or ("lstm" in model_location):
    validation_data = validation_data[:, :, np.newaxis]
    validation_labels = to_categorical(validation_labels,num_classes=len(class_names))

print("Finished converting")

loss, acc = new_model.evaluate(validation_data,  validation_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

predicted_labels = new_model.predict(validation_data)

print(validation_labels[:50])
if "cnn" in model_location:
    analyse_results(None, validation_data, validation_labels, predicted_labels, "cnn", base_filename, unnormalised_validation, acc)
elif "lstm" in model_location:
    analyse_results(None, validation_data, validation_labels, predicted_labels, "lstm", base_filename, unnormalised_validation, acc)
elif "fully_connected" in model_location:
    analyse_results(None, validation_data, validation_labels, predicted_labels, "fully_connected", base_filename, unnormalised_validation, acc)
else:
    print("NEW MODEL - UPDATE SCRIPT TO MATCH")

#print(new_model.predict(test_images).shape)
