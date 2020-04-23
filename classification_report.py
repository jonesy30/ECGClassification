import numpy as np
import sklearn.metrics

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter, FixedLocator

def get_classification_report(true_labels, predicted_labels):
    
    recall = sklearn.metrics.recall_score(true_labels,predicted_labels,average=None)
    precision = sklearn.metrics.precision_score(true_labels,predicted_labels,average=None)
    f1_score = sklearn.metrics.f1_score(true_labels,predicted_labels,average=None)

    return recall, precision, f1_score

def plot_classification_report(y_true,y_pred,classes,show_plot=True):
    recall, precision, f1_score = get_classification_report(y_true, y_pred)

    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    # plot with various axes scales
    fig, axs = plt.subplots(1, len(classes), figsize=(12,2), constrained_layout=False)
    fig.set_facecolor('lightgray')

    for i in range(len(classes)):
        # linear
        ax = axs[i]
        #ax.add_patch(p)
        text_to_display = "Precision: "+str(round(precision[i],2))+"\nRecall: "+str(round(recall[i],2))+"\nF1 Score: "+str(round(f1_score[i],2))

        ax.text(0.5*(left+right), 0.5*(bottom+top), text_to_display,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12,
                transform=ax.transAxes)
        ax.set_title(classes[i],fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_axis_off()

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.7,
                    hspace=0.1, wspace=0)
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.65)

    plt.suptitle("Per Class Results",fontweight="bold",size=16)
    
    if show_plot == True:
        plt.show()

if __name__ == "__main__":
    classes = ["one","two","three","four","five","six","seven","eight"]
    y_true = [1,2,3,4,5,6,7,8]
    y_pred = [1,2,4,5,5,6,7,4]

    recall, precision, f1_score = get_classification_report(y_true, y_pred)
    print("Recall = "+str(recall))
    print("Precision = "+str(precision))
    print("F1 score = "+str(f1_score))

    plot_classification_report(y_true,y_pred,classes)