import numpy as np
import glob,os,shutil

def save_predictions(ecgs,predicted_labels,true_labels,base_filename):
        
    if os.path.exists(base_filename):
        shutil.rmtree(base_filename,ignore_errors=True)
    
    os.makedirs(base_filename)

    for index,true_label in enumerate(true_labels):

        ecg = ecgs[index]
        predicted_label = predicted_labels[index]

        if not os.path.exists(base_filename+str(true_label)+"/"):
            os.makedirs(base_filename+str(true_label)+"/")
        
        f = open(base_filename+str(true_label)+"/pred_"+str(predicted_label)+"_"+str(index)+".txt","w")

        for value in ecg:
            value_int = int(round(value))
            to_write = str(value_int) + " "

            f.write(to_write)
        f.close()

if __name__ == "__main__":
    base_filename = "./mit_bih_processed_data_two_leads_subset/"
    ecgs = [[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]]
    predicted_labels = [1,2,3,4,5]
    true_labels = [2,3,1,5,4]

    save_predictions(ecgs, predicted_labels, true_labels, base_filename)