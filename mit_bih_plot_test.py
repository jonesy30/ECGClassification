import wfdb
import matplotlib.pyplot as plt

record = wfdb.rdsamp('mit_bih/203', sampto=3000, smoothframes=True)
annotation = wfdb.rdann('mit_bih/203', 'atr', sampto=3000)
#Annotation attributes: record_name, extension, sample (indices), symbol (classes), subtype, chan, num, aux_note, fs, label_store, description, custom_labels, contained_labels

unique_labels_df = annotation.get_contained_labels(inplace=False)
annotation_indices = annotation.sample
annotation_classes = annotation.symbol
sampling_frequency = annotation.fs

sig, _ = wfdb.srdsamp('mit_bih/203')

plt.plot(sig[:720,0])

wfdb.plotrec(record, annotation = annotation, title='Record 203 from MIT-BIH Arrhythmia Database', figsize = (10,4), ecggrids = 'all',plotannsym=True)