"""
File which deletes all files in the processed data - yes I could do this manually but it just saves time!
"""

import glob
import os

for path, subdirs, files in os.walk("./split_processed_data/"):
    for f in files:
        os.remove(path+"/"+f)