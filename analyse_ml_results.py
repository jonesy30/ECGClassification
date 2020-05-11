import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
import time
from classification_report import plot_classification_report
from visualise_incorrect_predictions import save_predictions
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def analyse_results(history, validation_data, validation_labels, predicted_labels, model_type, base_filename, unnormalised_validation, test_acc):
    class_names = ['A','E','j','L','N','P','R','V']
    labels = ["APB","Vesc","Jesc","LBBB","Normal","Paced","RBBB","VT"]
    label_names = {'N':'Normal','L':'LBBB','R':'RBBB','A':'APB','a':'AAPB','J':'JUNCTIONAL','S':'SP','V':'VT','r':'RonT','e':'Aesc','j':'Jesc','n':'SPesc','E':'Vesc','P':'Paced'}

    

    if history != None:

        if 'accuracy' in history.history.keys():
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
        elif 'acc' in history.history.keys():
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])

        plt.title('Model Accuracy ('+model_type+')')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.figure()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss ('+model_type+')')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.figure()

    tested = 0
    correct = 0
    correct_predictions = [0]*len(class_names)
    incorrect_predictions = [0]*len(class_names)

    predicted_values = []
    incorrectly_identified_ecgs = []
    incorrectly_identified_predicted_labels = []
    incorrectly_identified_true_labels = []

    correctly_identified_ecgs = []
    correctly_identified_predicted_labels = []
    correctly_identified_true_labels = []

    #Format the predictions into incorrect and correct predictions
    for i in range(len(validation_data)):
        predicted = np.where(predicted_labels[i] == np.amax(predicted_labels[i]))
        predicted_value = predicted[0][0]
        predicted_values.append(predicted_value)

        actual = validation_labels[i]
        if ("cnn" in model_type) or ("lstm" in model_type.lower()):
            actual = list(validation_labels[i]).index(1)
        elif "fully connected" in model_type.lower():
            actual = validation_labels[i]

        tested = tested + 1
        if actual == predicted_value:
            correct = correct + 1
            correct_predictions[actual] = correct_predictions[actual] + 1

            correctly_identified_ecgs.append(unnormalised_validation[i])
            correctly_identified_predicted_labels.append(class_names[predicted_value])
            correctly_identified_true_labels.append(class_names[actual])

        else:
            incorrect_predictions[actual] = incorrect_predictions[actual] + 1
            
            incorrectly_identified_ecgs.append(unnormalised_validation[i])
            incorrectly_identified_predicted_labels.append(class_names[predicted_value])
            incorrectly_identified_true_labels.append(class_names[actual])

    file_location = base_filename
    if "cnn" in model_type:
        file_location = base_filename+"/cnn/"
    elif "fully" in model_type.lower():
        file_location = base_filename+"/fully_connected/"
    elif "lstm" in model_type.lower():
        file_location = base_filename+"/lstm/"
    save_incorrect_predictions(incorrectly_identified_ecgs, incorrectly_identified_predicted_labels, incorrectly_identified_true_labels, file_location)

    accuracy = correct/tested

    accuracy_of_predictions = [0]*len(class_names)
    for i,item in enumerate(correct_predictions):
        total_labels = correct_predictions[i] + incorrect_predictions[i]
        if total_labels!=0:
            accuracy_of_predictions[i] = correct_predictions[i]/total_labels*100
        else:
            accuracy_of_predictions[i] = np.nan

    accuracy_of_predictions.append(accuracy*100)

    for i,item in enumerate(class_names):
        total = correct_predictions[i] + incorrect_predictions[i]
        class_names[i] = class_names[i] + " ("+str(total)+")"
    class_names.append("TOTAL")

    print("Test accuracy: "+str(test_acc))

    predicted_encoded = predicted_labels
    actual_encoded = validation_labels

    if ("cnn" in model_type) or ("lstm" in model_type):
        # show the inputs and predicted outputs
        predicted_encoded = np.argmax(predicted_labels, axis=1)
        actual_encoded = np.argmax(validation_labels, axis=1)
    elif "fully_connected" in model_type.lower():
        predicted_encoded = np.argmax(predicted_labels, axis=1)
        actual_encoded = validation_labels


    #Plot the confusion matrix of the expected and predicted classes

    matrix = confusion_matrix(actual_encoded, predicted_encoded, labels=np.arange(0,len(class_names)-1,1), normalize='all')
    plot_confusion_matrix(matrix, classes=labels, normalize=True, title="Confusion Matrix ("+model_type+"), Accuracy = "+str(round(test_acc*100,2))+"%")

    plot_classification_report(actual_encoded, predicted_encoded, labels, show_plot=False)

    plt.figure()

    overall_f1 = f1_score(actual_encoded, predicted_encoded, average='macro')
    overall_precision = precision_score(actual_encoded, predicted_encoded, average='macro')
    overall_recall = recall_score(actual_encoded, predicted_encoded, average='macro')

    print("Overall F1: "+str(overall_f1))
    print("Overall Precision: "+str(overall_precision))
    print("Overall Recall: "+str(overall_recall))

    #Plot prediction accuracy percentages
    plt.bar(class_names, accuracy_of_predictions)
    plt.xticks(class_names, fontsize=7, rotation=30)
    plt.title(model_type.upper()+"\nOverall Accuracy = "+str(round(test_acc*100,2))+"%")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,100))
    plt.ylabel("Accuracy of predictions (%)")
    plt.xlabel("Condition")
    plt.show()