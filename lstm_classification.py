"""
Initial experiment with LSTMs, keeping file for archiving purposes
"""

import random
import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
import pandas as pd

def read_ecg_data(filename):
    f = open(filename,"r")
    ecg_plot = np.fromfile(f, dtype=np.int16)
    return ecg_plot

os.chdir("./afib_nsr_data")
subfolders = [f.name for f in os.scandir('.') if f.is_dir() ] 

data_labels = []
ecg_plot = []
ecg_plot_lengths = []
files = []
list_of_names = []
for folder in subfolders:
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            filename = str(os.path.join(root, name))
            this_ecg = read_ecg_data(filename)

            if len(this_ecg) != 0:
                min_value = min(this_ecg)
                if min_value < 0:
                    new_ecg = [x-(min_value) for x in this_ecg]

                ecg_plot.append(new_ecg)
                list_of_names.append(filename)
                data_labels.append(folder)

for plot in ecg_plot:
    ecg_plot_lengths.append(len(plot))

for i in range(5):
    max_length = max(ecg_plot_lengths)
    print("Max length = "+str(max_length))
    print("Filename = "+str(list_of_names[ecg_plot_lengths.index(max_length)]))

    # indexes = [i for i,val in enumerate(ecg_plot_lengths) if val==max_length]
    # for i in indexes:
    #     print("Filename = "+str(list_of_names[i]))
    ecg_plot_lengths.remove(max_length)

training_data_labels = []
count_0 = 0
count_1 = 0
for item in data_labels:
    if item == 'NSR':
        training_data_labels.append(0)
        count_0 += 1
    else:
        training_data_labels.append(1)
        count_1 += 1

print("Count 0 = "+str(count_0))
print("Count 1 = "+str(count_1))

train_data, test_data, train_labels, test_labels = train_test_split(ecg_plot, training_data_labels, test_size=0.25, random_state=42)

train_data = sequence.pad_sequences(train_data, maxlen=max_length)
test_data = sequence.pad_sequences(test_data, maxlen=max_length)

train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))

df = pd.DataFrame(train_labels)
print("Training split: "+str(df[0].value_counts()))
df = pd.DataFrame(test_labels)
print("Test split: "+str(df[0].value_counts()))

#embedding_vecor_length = 500
model = Sequential()
#model.add(Embedding(200, embedding_vecor_length, input_length=max_length))
model.add(LSTM(60, return_sequences=True, input_shape=(1, max_length)))
model.add(Dropout(0.2))
model.add(LSTM(60))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=100, batch_size=200)
print(model.summary())
# Final evaluation of the model
scores = model.evaluate(test_data, test_labels, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

predictions = model.predict(test_data)

one_hot_predictions = []
for value in predictions:
    one_hot_predictions.append(round(value[0]))

df = pd.DataFrame(one_hot_predictions)
print(df[0].value_counts())

print("F1 Score: "+str(f1_score(test_labels, one_hot_predictions)))