from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from analyse_ml_results import analyse_results
import sys
from tensorflow import keras
from tensorflow.keras.layers import Conv1D

#code taken (and adapted) from https://github.com/facundodeza/transfomer-audio-classification/blob/master/audio_classification_transformer.ipynb

class_names = ['A','E','j','L','N','P','R','V']
labels = ["APB","Vesc","Jesc","LBBB","Normal","Paced","RBBB","VT"]

label_names = {'N':'Normal','L':'LBBB','R':'RBBB','A':'APB','a':'AAPB','J':'JUNCTIONAL','S':'SP','V':'VT','r':'RonT','e':'Aesc','j':'Jesc','n':'SPesc','E':'Vesc','P':'Paced'}

two_leads = 1

#this is taken directly from tensorflow website (text transformer tutorial)
#and the transformer chatbot code
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        print("D Model = "+str(d_model))

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs

#taken from tensorflow chatbot tutorial
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

#also taken from tensorflow chatbot tutorial
#includes residual connection
def encoder_layer(units, d_model, num_heads, dropout,name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None,d_model ), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)
    
def encoder(num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    
    projection = tf.identity(inputs)
    projection *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    
    #1D input
    projection = PositionalEncoding(1, d_model)(projection)

    outputs = tf.keras.layers.Dropout(rate=dropout)(projection)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])
    
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def transformer(num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                output_size,
                name="transformer"):
    
    inputs = tf.keras.Input(shape=(None,d_model), name="inputs")

    # conv = keras.layers.Conv1D(kernel_size=(10), filters=1, strides=10, activation='relu', kernel_initializer='VarianceScaling', name='conv')(inputs)
    # conv = keras.layers.Conv1D(kernel_size=(5), filters=1, strides=5, activation='relu', kernel_initializer='VarianceScaling')(conv)

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(tf.dtypes.cast(
            
        #Like our input has a dimension of length X d_model but the masking is applied to a vector
        # We get the sum for each row and result is a vector. So, if result is 0 it is because in that position was masked      
        tf.math.reduce_sum(
        inputs,
        axis=2,
        keepdims=False,
        name=None), tf.int32))

    #conv = keras.layers.Reshape(target_shape=(1,))(conv)

    #d_model = 32

    #enc_padding_mask = tf.keras.layers.Embedding(input_dim=d_model, mask_zero=True)(inputs)

    #enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1,1,None))
    #enc_padding_mask = tf.dtypes.cast(tf.math.reduce_sum(inputs,axis=2,keepdims=False), tf.int32)

    enc_outputs = encoder(
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='encoder'
    )(inputs=[inputs, enc_padding_mask])

    #We reshape for feeding our FC in the next step
    outputs=tf.reshape(enc_outputs,(-1,d_model))
    
    #We predict our class
    outputs = tf.keras.layers.Dense(units=output_size,use_bias=True,activation='softmax', name="outputs")(outputs)

    return tf.keras.Model(inputs=[inputs], outputs=outputs)

#from tensorflow website - ensures model does not treat padding as the input
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]

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
        label_text = ""
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

#taken from tensorflow chatbot transformer
def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)

    return tf.matmul(attention_weights, value)

base_filename = "./mit_bih_processed_data_two_leads/"

(training_data, training_labels) = read_data(base_filename + "network_data/training_set/")
([validation_data,unnormalised_validation], validation_labels) = read_data(base_filename + "network_data/validation_set/",save_unnormalised=True)

#training_data = validation_data
#training_labels = validation_labels

training_data = [np.asarray(item) for item in training_data]
training_data = np.array(training_data)

training_data = training_data[:,np.newaxis,:]

training_labels = to_categorical(training_labels, num_classes=len(class_names))

#Turn training labels into element arrays of 1x1 element arrays (each containing a label)
training_labels = [np.asarray(item) for item in training_labels]
training_labels = np.array(training_labels)

validation_data = [np.asarray(item) for item in validation_data]
validation_data = np.array(validation_data)

#Turn training labels into element arrays of 1x1 element arrays (each containing a label)
validation_labels = [np.asarray(item) for item in validation_labels]
validation_labels = np.array(validation_labels)

validation_data = validation_data[:, np.newaxis, :]

validation_labels = to_categorical(validation_labels, num_classes=len(class_names))

print("Training data shape")
print(training_data.shape)

NUM_LAYERS = 6
D_MODEL = training_data.shape[2]
NUM_HEADS = 5
UNITS = 128
DROPOUT = 0.2
OUTPUT_SIZE = len(class_names)
EPOCHS = 100

model = transformer(num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    output_size=OUTPUT_SIZE)

# tf.keras.utils.plot_model(model, to_file='transformer.png', show_shapes=True)
# plt.show()

model.compile(optimizer=tf.keras.optimizers.Adam(0.000001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_data,training_labels, epochs=EPOCHS, validation_data=(validation_data, validation_labels))

if not os.path.exists("./saved_models/"):
    os.makedirs("./saved_models/")
if not os.path.exists("./saved_models/transformer/"):
    os.makedirs("./saved_models/transformer/")

model.save(".\\saved_models\\transformer\\transformer_model")

accuracy = max(history.history['val_accuracy'])

test_loss, test_acc = model.evaluate(validation_data, validation_labels)
predicted_labels = model.predict(validation_data)

print("Accuracy = "+str(accuracy))

analyse_results(history, validation_data, validation_labels, predicted_labels, "transformer", base_filename, unnormalised_validation, test_acc)
