import skimage.segmentation
import numpy as np
import skimage
from skimage import io
import keras
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import copy
import sklearn.metrics
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.utils import to_categorical
import sys
import random
from keras.models import Model

import lime

class_names = ['A','E','j','L','N','P','R','V']
two_leads = 0

# Xi = skimage.io.imread("https://arteagac.github.io/blog/lime_image/img/cat-and-dog.jpg")
# Xi = skimage.transform.resize(Xi, (299,299)) 
# Xi = (Xi - 0.5)*2 #Inception pre-processing
# skimage.io.imshow(Xi/2+0.5) # Show image before inception preprocessing

# plt.show()

model_location = 'saved_models\\cnn\\cnn_model'
model = tf.keras.models.load_model(model_location)

print(model.summary())

# model.layers[-1].activation=None

# print(model.summary())

# model = Model(model.input, model.layers[-1].output)
# print(model.summary())

f = open("./mit_bih_processed_data_two_leads/network_data/training_set/ecg_77001.txt", "r")
ecg = []
for i,line in enumerate(f):
    line = line.replace("\n","")
    #ECG signal stored in first line separated by spaces
    if i < 1:
        line_segments = line.split()
        for i,item in enumerate(line_segments):
            line_segments[i] = float(item)

        for item in line_segments:
            ecg.append(item)
f.close()

# plt.plot(ecg)
# plt.show()
ecg_original = ecg.copy()

ecg = [np.asarray(item) for item in ecg]
ecg = np.array(ecg)

ecg = np.array(ecg)

ecg = ecg[:, np.newaxis]
ecg = np.expand_dims(ecg, axis=0)

preds = model.predict_proba(ecg)

print(preds)
print(preds[0].argsort()[-5:])

top_pred_classes = preds[0].argsort()[-5:][::-1] # Save ids of top 5 classes
print(top_pred_classes)

# np.random.seed(222)
# inceptionV3_model = keras.applications.inception_v3.InceptionV3() #Load pretrained model
# preds = inceptionV3_model.predict(Xi[np.newaxis,:,:,:])
# top_pred_classes = preds[0].argsort()[-5:][::-1] # Save ids of top 5 classes
# decode_predictions(preds)[0] #Print top 5 classes

# superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
# num_superpixels = np.unique(superpixels).shape[0]
# skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))

# plt.show()

#Generate random masking blocks
num_blocks = random.randint(50,150)

masking_blocks = []

for i in range(num_blocks):
    #836: length of ecg (860) - min_length of block (25) + 1 fpr random outer bound
    end_index = 1000
    start_index = 0
    mask_length = 0
    
    while end_index > 860:
        start_index = random.randint(0,836)
        mask_length = random.randint(25, 50)
        end_index = start_index+mask_length
    
    masking_blocks.append([start_index, end_index])

#Generate perturbations
# num_perturb = 150
# perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

# #Create function to apply perturbations to images
# def perturb_image(img,perturbation,segments): 
#   active_pixels = np.where(perturbation == 1)[0]
#   mask = np.zeros(segments.shape)
#   for active in active_pixels:
#       mask[segments == active] = 1 
#   perturbed_image = copy.deepcopy(img)
#   perturbed_image = perturbed_image*mask[:,:,np.newaxis]
#   return perturbed_image

# #Show example of perturbations
# print(perturbations[0]) 
# skimage.io.imshow(perturb_image(Xi/2+0.5,perturbations[0],superpixels))

num_perturb = 150

perturbations_indexes = []

for i in range(num_perturb):
    this_block = [1] * num_blocks
    number_off = random.randint(1,num_blocks)
    off_list = random.sample(range(0, num_blocks), number_off)
    for index in off_list:
        this_block[index] = 0
    
    perturbations_indexes.append(this_block)

perturbation_ecgs = []
for perturbation in perturbations_indexes:
    new_ecg = ecg_original.copy()
    for index,value in enumerate(perturbation):
        if value == 0:
            start, end = masking_blocks[index]
            new_ecg[start:end] = [0]*(end-start)
    perturbation_ecgs.append(new_ecg)

#pre-process all perturbations into the correct format
#perturbations = [np.asarray(item) for item in perturbations]

original_perturbations_ecgs = perturbation_ecgs.copy()

perturbation_ecgs = np.array(perturbation_ecgs)

perturbation_ecgs = perturbation_ecgs[:, :, np.newaxis]

# for index,perturbation in enumerate(perturbations):
#     plt.plot(perturbation)
#     plt.title(", ".join(str(v) for v in perturbations_indexes[index]))
#     plt.show()

# predictions = []
# for pert in perturbations:
#   perturbed_img = perturb_image(Xi,pert,superpixels)
#   pred = inceptionV3_model.predict(perturbed_img[np.newaxis,:,:,:])
#   predictions.append(pred)

predictions = model.predict(perturbation_ecgs)

predictions = np.array(predictions)

print("Predictions")

print(predictions.shape)
print(predictions[0])

ecg_reshaped = np.array(ecg_original)
ecg_reshaped = ecg_reshaped.reshape(1,-1)

distances = sklearn.metrics.pairwise_distances(original_perturbations_ecgs, ecg_reshaped, metric='cosine').ravel()
print(distances.shape)

# original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
# distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
# print(distances.shape)

#Transform distances to a value between 0 an 1 (weights) using a kernel function
kernel_width = 0.25
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
print(weights.shape)

class_to_explain = top_pred_classes[0] #Labrador class

print(class_to_explain)

print("Shape")
print(perturbation_ecgs.shape)

simpler_model = LinearRegression()
simpler_model.fit(X=perturbations_indexes, y=predictions[:,class_to_explain], sample_weight=weights)

coeff = simpler_model.coef_

#coeff = simpler_model.coef_[0]

print("Coeff = ")
print(coeff)
print(len(coeff))
print(len(perturbations_indexes))

#Use coefficients from linear model to extract top features
num_top_features = 4
top_features = np.argsort(coeff)[::-1][-num_top_features:] 

print("Top Features")
print(top_features)

print(len(original_perturbations_ecgs))

top_indexes = []
for feature in top_features:
    top_indexes.append(perturbations_indexes[feature])

print(top_indexes)

for perturbation in top_indexes:
    new_ecg = ecg_original.copy()
    for index, value in enumerate(perturbation):
        if value == 0:
            start, end = masking_blocks[index]
            new_ecg[start:end] = [0]*(end-start)
    plt.plot(new_ecg)
    plt.show()

worst_features = np.argsort(coeff)[-num_top_features:] 

print("Worst Features")
print(worst_features)

print(len(original_perturbations_ecgs))

worst_indexes = []
for feature in worst_features:
    worst_indexes.append(perturbations_indexes[feature])

print(worst_indexes)

for perturbation in worst_indexes:
    new_ecg = ecg_original.copy()
    for index, value in enumerate(perturbation):
        if value == 0:
            start, end = masking_blocks[index]
            new_ecg[start:end] = [0]*(end-start)
    plt.plot(new_ecg)
    plt.show()

sys.exit()

# best_perturbations = []
# for perturbation in top_peturbations:
#     new_ecg = ecg_original.copy()
#     for index,value in enumerate(perturbation):
#         if value == 0:
#             start, end = masking_blocks[index]
#             new_ecg[start:end] = [0]*(end-start)
#     best_perturbations.append(new_ecg)

# for ecg in top_peturbations:
#     plt.plot(ecg)
#     plt.show()

# worst_features = np.argsort(coeff)[::-1][-num_top_features:]

# print("Worst Features")
# print(worst_features)

# worst_peturbations = []
# for feature in worst_features:
#     worst_peturbations.append(original_perturbations_ecgs[feature])

# worst_perturbations = []
# for perturbation in worst_peturbations:
#     new_ecg = ecg_original.copy()
#     for index,value in enumerate(perturbation):
#         print(index)
#         if value == 0:
#             start, end = masking_blocks[index]
#             new_ecg[start:end] = [0]*(end-start)
#     worst_peturbations.append(new_ecg)

# for ecg in worst_peturbations:
#     plt.plot(ecg)
#     plt.show()

# sys.exit()



#Show only the superpixels corresponding to the top features
print(num_perturb)
mask = np.zeros(num_perturb) 
mask[top_features]= True #Activate top superpixels

print(mask)
print(len(mask))
print(num_blocks)

print(len(masking_blocks))

print("Enumerating Mask")
masked_ecg = ecg_original.copy()
for mask_index, value in enumerate(mask):
    if value == 0:
        start, end = masking_blocks[mask_index]
        masked_ecg[start:end] = [0]*(end-start)

plt.plot(masked_ecg)
plt.show()
#skimage.io.imshow(perturb_image(Xi/2+0.5,mask,superpixels))

# plt.plot(perturbations)

# plt.show()