"""
File which uses a convolutional neural network to classify ECG signals into classes of heart condition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import itertools
import time
from analyse_ml_results import analyse_results
from sklearn.utils import resample
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

#class_names = ['AFIB_AFL', 'AVB_TYPE2', 'BIGEMINY', 'EAR', 'IVR', 'JUNCTIONAL', 'NOISE', 'NSR', 'SVT', 'TRIGEMINY', 'WENCKEBACH']
class_names = ['A','E','j','L','N','P','R','V']
labels = ["APB","Vesc","Jesc","LBBB","Normal","Paced","RBBB","VT"]

label_names = {'N':'Normal','L':'LBBB','R':'RBBB','A':'APB','a':'AAPB','J':'JUNCTIONAL','S':'SP','V':'VT','r':'RonT','e':'Aesc','j':'Jesc','n':'SPesc','E':'Vesc','P':'Paced'}

two_leads = 1

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

    normalised = [(x - min_value)/range_values for x in ecg_signal]

    return [np.float32(a) for a in normalised]

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

#Read the training and validation data and labels and store in arrays
#(training_data, training_labels) = read_data("./split_processed_data/network_data_unfiltered/training_set/")
#(validation_data, validation_labels) = read_data("./split_processed_data/network_data_unfiltered/validation_set/")

start_time = time.time()

base_filename = "./mit_bih_processed_data_two_leads/"

(training_data, training_labels) = read_data(base_filename + "network_data/training_set/")

#Turn each training data array into numpy arrays of numpy arrays
training_data = [np.asarray(item) for item in training_data]
training_data = np.array(training_data)

#Turn training labels into element arrays of 1x1 element arrays (each containing a label)
training_labels = [np.asarray(item) for item in training_labels]
training_labels = np.array(training_labels)

upsampling_flag = 1
if upsampling_flag == 1:
    #Upsample the data to amplify lesser classes
    df = pd.DataFrame(training_data)
    df['label'] = training_labels

    #Get the size of the largest class (so I know how much to upsample by)
    print("Downsampling")
    print(df['label'].value_counts())

    max_label = df['label'].value_counts().idxmax()

    downsampled_df = df[df['label']!=max_label]
    second_max_class = downsampled_df['label'].value_counts().max()
    df_majority_downsampled = resample(df[df['label']==max_label], replace=False, n_samples=second_max_class, random_state=123)

    downsampled_df = pd.concat([df_majority_downsampled, downsampled_df])
    print(downsampled_df['label'].value_counts())

    max_number = downsampled_df['label'].value_counts()[max_label]

    #Create an upsampling space, initially fill it with the largest category
    df_oversampled = downsampled_df[downsampled_df['label']==max_label]

    #For each smaller class, oversample it and add to the oversampling space
    for value in range(downsampled_df['label'].nunique()):
        if value != max_label:
            df_class = downsampled_df[downsampled_df['label']==value]
            df_class_over = df_class.sample(max_number, replace=True)
            df_class_over = pd.DataFrame(df_class_over)
            df_oversampled = pd.concat([df_oversampled, df_class_over])

    print(df_oversampled['label'].value_counts())

    #Convert the upsampled data to training and labelled data
    training_labels = df_oversampled['label'].tolist()
    training_data = df_oversampled.drop(columns='label').to_numpy()
    #Aaaaand we're done upsampling! Hooray!

#Resize training data to fit CNN input layer and convert labels to one-hot encoding
training_data = training_data[:, :, np.newaxis]

training_labels = to_categorical(training_labels, num_classes=len(class_names))

if two_leads == 0:
    input_shape = 430
else:
    input_shape = 860

#Build the intial model
# model = keras.Sequential([
#     keras.layers.InputLayer(input_shape=[input_shape,1])
#     #keras.layers.Lambda(lambda v: tf.cast(tf.spectral.fft(tf.cast(v,dtype=tf.complex64)),tf.float32))
#     #Dropout(0.2)
#     #keras.layers.Conv1D(kernel_size=10, filters=128, strides=4, use_bias=True, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_initializer='VarianceScaling'),
#     #keras.layers.AveragePooling1D(pool_size=2, strides=1,padding="same")
# ])
# #Add extra labels

# model.add(keras.layers.Conv1D(kernel_size=10, filters=64, strides=4, input_shape=(input_shape,1), use_bias=True, kernel_initializer='VarianceScaling'))
# model.add(keras.layers.LeakyReLU(alpha=0.3))
# model.add(keras.layers.AveragePooling1D(pool_size=2, strides=1, padding="same"))
# model.add(Dropout(0.5))

# model.add(keras.layers.Conv1D(kernel_size=10, filters=64, strides=4, use_bias=True, kernel_initializer='VarianceScaling'))
# model.add(keras.layers.LeakyReLU(alpha=0.3))
# model.add(keras.layers.AveragePooling1D(pool_size=2, strides=1, padding="same"))
# model.add(Dropout(0.5))

# #model.add(keras.layers.AveragePooling1D(pool_size=2, strides=1))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(len(class_names), activation='softmax'))

#Structure from Hannun et al
# model = keras.Sequential()

# # #block 1
# model.add(keras.layers.InputLayer(input_shape=[input_shape,1]))
# model.add(keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LeakyReLU(alpha=0.3))

input = keras.layers.Input(shape=(input_shape,1,))
x = keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling')(input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)

#block 2
# model.add(keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LeakyReLU(alpha=0.3))
# model.add(keras.layers.Dropout(0.2))

x = keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = keras.layers.Dropout(0.2)(x)

# input2 = tf.keras.layers.Input(shape=(415,24))
# x2 = tf.keras.layers.Dense(32, activation='relu')(input2)
# # equivalent to `added = tf.keras.layers.add([x1, x2])`
# added = tf.keras.layers.Add()([x2, shortcut])
# out = tf.keras.layers.Dense(4)(added)
# new_model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

#block 3
for i in range(4):

    #filters = 32*(2**(i//4))
    # filters = 64 * ((i//2)+1)
    # print("Filter size = "+str(filters))
    # model.add(keras.layers.Conv1D(kernel_size=16, filters=filters, strides=1, use_bias=True, kernel_initializer='VarianceScaling'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.LeakyReLU(alpha=0.3))
    # #x2 = keras.layers.Activation(activations.relu)(input)

    # #this should be strides = 4 to subsample but I think my data is too small for that
    # model.add(keras.layers.Conv1D(kernel_size=16, filters=filters, strides=1, use_bias=True, kernel_initializer='VarianceScaling'))
    # #layer = keras.layers.BatchNormalization()(layer)
    # model.add(keras.layers.LeakyReLU(alpha=0.3))
    # model.add(keras.layers.Dropout(0.2))

    shortcut = MaxPooling1D(pool_size=1)(x)

    filters = 64 * ((i//2)+1)
    print("Filter size = "+str(filters))
    x = keras.layers.Conv1D(kernel_size=16, filters=filters, strides=1, use_bias=True, padding="same", kernel_initializer='VarianceScaling')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, padding="same", kernel_initializer='VarianceScaling')(x)
        #layer = keras.layers.BatchNormalization()(layer)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = keras.layers.Dropout(0.2)(x)
    #x2 = keras.layers.Activation(activations.relu)(input)

    x = tf.keras.layers.Add()([x, shortcut])

    #test = keras.Sequential()

    #layer = keras.layers.Add()([test, test_1])
    #model.add(layer)

#block 4
# model.add(keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LeakyReLU(alpha=0.3))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(len(class_names), activation='softmax'))

x = keras.layers.Conv1D(kernel_size=16, filters=32, strides=1, use_bias=True, kernel_initializer='VarianceScaling')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(len(class_names), activation='softmax')(x)
model = tf.keras.models.Model(inputs=[input], outputs=out)

#MAGIC NUMBERS
verbose = 1
epochs = 30
batch_size = 128

opt = Adam(learning_rate=0.001)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=0.001 * 0.001)

#Build and fit the model
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history = model.fit(training_data,training_labels,epochs=epochs,batch_size=batch_size,validation_split=0.1,verbose=verbose, callbacks=[reduce_lr])
#history = model.fit(training_data,training_labels,epochs=epochs,batch_size=batch_size,verbose=verbose)

print("History Keys")
print(history.history.keys())

print("Model Summary")
print(model.summary())

if not os.path.exists("./saved_models/"):
    os.makedirs("./saved_models/")
if not os.path.exists("./saved_models/cnn_hannun/"):
    os.makedirs("./saved_models/cnn_hannun/")

model.save(".\\saved_models\\cnn_hannun\\cnn_model")

print("Evaluating....")

#Clear training data - we don't need this any more
del training_data
del training_labels

([validation_data,unnormalised_validation], validation_labels) = read_data(base_filename + "network_data/validation_set/",save_unnormalised=True)
print("Finished reading validation file")

#Turn each validation data array into numpy arrays of numpy arrays
validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#Turn validation labels into element arrays of 1x1 element arrays (each containing a label)
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

#Resize validation data to fit CNN input layer and convert labels to one-hot encoding
validation_data = validation_data[:, :, np.newaxis]
validation_labels = to_categorical(validation_labels, num_classes=len(class_names))

test_loss, test_acc = model.evaluate(validation_data, validation_labels)
predicted_labels = model.predict(validation_data)
    
end_time = time.time()
print("Time for "+str(epochs)+" epochs = "+str(end_time-start_time))

analyse_results(history, validation_data, validation_labels, predicted_labels, "cnn", base_filename, unnormalised_validation, test_acc)
