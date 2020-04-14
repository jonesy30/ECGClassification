from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class_names = ['A','E','j','L','N','P','R','V']
labels = ["APB","Vesc","Jesc","LBBB","Normal","Paced","RBBB","VT"]
#labels = ["APB","Vesc","Jesc","LBBB","Normal","Paced","VT"]

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
        if label != "" and label != 4:
            normalized_data = normalize(found_data, file)
            data.append(normalized_data)
            labels.append(label)
    
    return data, labels

#Read the training and validation data and labels and store in arrays
#(training_data, training_labels) = read_data("./split_processed_data/network_data_unfiltered/training_set/")
#(validation_data, validation_labels) = read_data("./split_processed_data/network_data_unfiltered/validation_set/")

(training_data, training_labels) = read_data("./mit_bih_processed_data/network_data/training_set/")
#(validation_data, validation_labels) = read_data("./mit_bih_processed_data/network_data/validation_set/")

# total_data_set = training_data.append(validation_data)
# total_labels = training_labels.append(validation_labels)

X = np.array(training_data)
y = np.array(training_labels)

#y = to_categorical(y)

#X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: labels[i])
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

# plt.gray()
# fig = plt.figure( figsize=(16,7) )
# for i in range(0,15):
#     ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
#     ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
# plt.show()

# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df[feat_cols].values)
# df['pca-one'] = pca_result[:,0]
# df['pca-two'] = pca_result[:,1] 
# df['pca-three'] = pca_result[:,2]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=df.loc[rndperm,:]["pca-one"], 
#     ys=df.loc[rndperm,:]["pca-two"], 
#     zs=df.loc[rndperm,:]["pca-three"], 
#     c=df.loc[rndperm,:]["y"].astype(int),
#     cmap='tab10'
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')
# plt.show()

#t-SNE without pre-PCA
N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
#data_subset = df_subset[feat_cols].values
data_subset = df[feat_cols].values

# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(data_subset)
# df_subset['pca-one'] = pca_result[:,0]
# df_subset['pca-two'] = pca_result[:,1] 
# df_subset['pca-three'] = pca_result[:,2]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(data_subset)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]

# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 8),
#     data=df_subset,
#     legend="full",
#     alpha=0.3
# )

# Compare PCA to t-SNE
# plt.figure(figsize=(16,7))
# ax1 = plt.subplot(1, 2, 1)
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax1
# )
# ax2 = plt.subplot(1, 2, 2)
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax2
# )

# plt.show()

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df['tsne-pca50-one'] = tsne_pca_results[:,0]
df['tsne-pca50-two'] = tsne_pca_results[:,1]
# plt.figure(figsize=(16,4))
# ax1 = plt.subplot(1, 3, 1)
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax1
# )
# ax2 = plt.subplot(1, 3, 2)
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df_subset,
#     legend="full",
#     alpha=0.3,
#     ax=ax2
# )
# ax3 = plt.subplot(1, 3, 3)

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="label",
    palette=sns.color_palette("hls", 7),
    data=df,
    legend="full",
    alpha=0.3
    #ax=ax3
)

# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=df_subset.loc[rndperm,:]["tsne-pca50-one"], 
#     ys=df_subset.loc[rndperm,:]["tsne-pca50-two"], 
#     zs=df_subset.loc[rndperm,:]["tsne-pca50-three"], 
#     c=df_subset.loc[rndperm,:]["y"].astype(int),
#     cmap='tab10',
#     label=labels
# )

plt.xlabel("tsne-one")
plt.ylabel("tsne-two")

#plt.legend(labels)
plt.show()