"""
File which re-categorises the folders created by raw data conversion into classes being used for network
"""

import glob, os
import numpy as np
import shutil

def recategorise_classes(base_filename):
    print("Recategorising classes...")
    
    if os.path.exists(base_filename+"AFIB/") and not os.path.exists(base_filename+"AFIB_AFL/"):
        os.rename(base_filename+"AFIB/", base_filename+"AFIB_AFL/")

    if os.path.exists(base_filename+"AFL/"):
        files = os.listdir(base_filename+"AFL/")

        for file in files:
            shutil.move(base_filename + "AFL/" + file, base_filename + "AFIB_AFL/")
        shutil.rmtree(base_filename+"AFL/")

    if os.path.exists(base_filename+"VT/"):
        shutil.rmtree(base_filename+"VT/")
    if os.path.exists(base_filename+"SUDDEN_BRADY/"):
        shutil.rmtree(base_filename+"SUDDEN_BRADY/")

if __name__ == "__main__":
    recategorise_classes("./testing_classes_reorganise/")