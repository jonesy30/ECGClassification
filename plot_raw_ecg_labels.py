"""
Function used for visualising the raw ECG signals, plots original ECG and annotates plot with information from corresponding episode file
"""

import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ecg_file = "./raw_ecg_data/0b32a14870df10b3429dfa6c360b3a50_0001.ecg"
f = open(ecg_file, "r")
raw_ecg = np.fromfile(f, dtype=np.int16)

plt.plot(raw_ecg, zorder=1)

episode = "./raw_ecg_data/0b32a14870df10b3429dfa6c360b3a50_0001_grp1.episodes.json"

label_file = open(episode,"r")
this_label = label_file.read()
this_label = this_label.strip()

result = re.search("\\[\\{(.*)\\}\\]", this_label)
new_label = result.group(0)

information_array = new_label.split("{")

this_label_onsets = []
this_label_offsets = []
this_label_rhythm_name = []
for item in information_array:
    match = re.search('"onset": (\d+)', item)
    if match:
        this_label_onsets.append(match.group(1))
    
    match = re.search('"offset": (\d+)',item)
    if match:
        this_label_offsets.append(match.group(1))
    
    match = re.search('"rhythm_name": "(.*)",',item)
    if match:
        this_label_rhythm_name.append(match.group(1))

colors = cm.rainbow(np.linspace(0, 1, len(this_label_onsets)))
for index,start in enumerate(this_label_onsets):
    end = this_label_offsets[index]
    rhythm = this_label_rhythm_name[index]

    start = int(start) - 1
    end = int(end) - 1

    x = [start,end]
    y = [raw_ecg[start], raw_ecg[end]]

    plt.plot(x,y,'k|',markersize=50)
    plt.text(x[0]+100, -500, s=rhythm, color='k',size=8)

    plt.annotate(s='', xy=(x[1]-50,-200), xytext=(x[0]+50,-200), arrowprops=dict(arrowstyle='<->'))


    #plt.arrow(x[0]+200,y=-350,dx=x[1]-x[0]-400,dy=0,arrowstyle='<->')

plt.title("30 second ECG annotated by cardiologist (ecg 2)")
plt.xlabel("time")
plt.show()
print(this_label_onsets)
print(this_label_offsets)
print(this_label_rhythm_name)