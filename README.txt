Disclaimer: this repo is not clean at all - cleaning will happen

OVERVIEW
ECG classification and filter visualisation

RAW DATA TO PROCESSED DATA
pre_processing and pre_processing_split_samples turns annotated ECG data (found at link below) into ECG signals separated into folders by condition
    - pre_processing_split_samples splits the ECGs into exactly 400 sample (2 second - 200Hz signal) blocks, pre_processing leaves samples at the original length
        - The original length is extremely variable since the original data is labelled chunks per condition, chunks are variable sizes

PROCESSED DATA TO NETWORK READY DATA
Using feature extraction, using raw ecg signal
    - format_data_for_network turns processed ECG data (found below) into feature extracted data split into training and validation datasets
        - ecg_feature_extraction is the module used by this script
    - format_ecg_samples_for_network turns processed ECG data into (noise reduced - can be set) feature extracted training and validation datasets
        - noise_reduction is a module used by this script - has two methods
            - highpass filter alone: uses bandpass filter
            - manlio method: uses highpass filter and other techniques given to me by Manlio Tassieri

CLASSIFICATION
network_classification - uses fully-connected deep neural network
cnn_classification - uses a convolutional neural network
lstm_classification - uses an lstm
transformer_classification - uses a transformer

VISUALISATION
compare_correct_incorrect_classes - plots which ECGs were classified incorrectly
ecg_feature_extraction - gets key ECG features (P, Q, R, S)
filter_visualisation - where the majority of my saliency map code lives
saliency_map_visualisation - plots saliency map value by ECG feature (by splitting into 0.1 second segments AND key ECG feature)
lime_interpretation - an attempt to use LIME to visualise these models (unfinished)
t_sne_test - uses t-SNE to visualise classification

HELPER METHODS
plot_ecg plots any ECG
view_feature_extracted_data visualises feature extracted data (as expected)
clear_processed_files deletes everything in a folder (quick!)

DATASET NOTES
14 classes:
    - NSR - normal sinus rhythm (everything is fine)
    - NOISE - "other" or "unknown" (important as I want the network to have an "other" or "I don't know" category, so a doctor can be called in to make the final judgement or so the patient can be called in for more tests)
    - AFIB - atrial fibrillation
    - AFL - atrial flutter
    - AVB_TYPE2 - atrio-ventricular block
    - BIGEMINY - bigeminy
    - EAR - ectopic atrial rhythm
    - IVR - idioventricular block
    - JUNCTIONAL - junctional rhythm
    - SUDDEN_BRADY - bradycardia
    - SVT - supraventricular tachycardia
    - TRIGEMINY - trigeminy
    - VT - ventricular tachycardia
    - WENCKEBACH - wenkebach

Find the ECG data at:

https://irhythm.github.io/cardiol_test_set/

Original paper:

Hannun, A. Y., Rajpurkar, P., Haghpanahi, M. & Tyson, G. H., n.d. Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory Electrocardiograms Using a Deep Neural Network. [Online] 
Available at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6784839/pdf/nihms-1051795.pdf

MIT-BIH Dataset:
https://www.physionet.org/content/mitdb/1.0.0/

Classes:

N		Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
L		Left bundle branch block beat
R		Right bundle branch block beat
B		Bundle branch block beat (unspecified)
A		Atrial premature beat
a		Aberrated atrial premature beat
J		Nodal (junctional) premature beat
S		Supraventricular premature or ectopic beat (atrial or nodal)
V		Premature ventricular contraction
r		R-on-T premature ventricular contraction
F		Fusion of ventricular and normal beat (not included)
e		Atrial escape beat
j		Nodal (junctional) escape beat
n		Supraventricular escape beat (atrial or nodal)
E		Ventricular escape beat
/ (P)   Paced beat - changed to P for network processing (since I can't have / as a directory name)
f		Fusion of paced and normal beat (not included)