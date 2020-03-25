import glob
import os

for path, subdirs, files in os.walk("./split_processed_data/"):
    for f in files:
        os.remove(path+"/"+f)