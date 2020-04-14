from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#mnist = fetch_openml("MNIST original")
#X = mnist.data / 255.0
#y = mnist.target

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
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
data_subset = df_subset[feat_cols].values
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
#     palette=sns.color_palette("hls", 10),
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

df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
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

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
    #ax=ax3
)

plt.show()