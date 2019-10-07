import os, os.path
import random

os.chdir("./processed_data/NSR")
files = len([name for name in os.listdir('.') if os.path.isfile(name)])
number_of_samples = 400

delete_files = random.sample(range(1, files), (files-number_of_samples))
for i,filename in enumerate(os.listdir('.')):
    if i in delete_files:
        os.remove(filename)