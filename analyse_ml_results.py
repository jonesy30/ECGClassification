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
        if ("cnn" in model_type) or ("lstm" in model_type.lower()) or ("transformer" in model_type.lower()):
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

    file_location = "./saved_ecg_classifications/"

    if not os.path.exists(file_location):
        os.makedirs(file_location)

    if "cnn" in model_type:
        file_location = file_location+"/cnn/"
    elif "fully" in model_type.lower():
        file_location = file_location+"/fully_connected/"
    elif "lstm" in model_type.lower():
        file_location = file_location+"/lstm/"
    elif "transformer" in model_type.lower():
        file_location = file_location+"/transformer/"
    
    if not os.path.exists(file_location):
        os.makedirs(file_location)

    save_predictions(incorrectly_identified_ecgs, incorrectly_identified_predicted_labels, incorrectly_identified_true_labels, file_location+"incorrect_predictions/")
    save_predictions(correctly_identified_ecgs, correctly_identified_predicted_labels, correctly_identified_true_labels, file_location+"correct_predictions/")

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

    if ("cnn" in model_type) or ("lstm" in model_type) or ("transformer" in model_type):
        # show the inputs and predicted outputs
        predicted_encoded = np.argmax(predicted_labels, axis=1)
        actual_encoded = np.argmax(validation_labels, axis=1)
    elif "fully_connected" in model_type.lower():
        predicted_encoded = np.argmax(predicted_labels, axis=1)
        actual_encoded = validation_labels


    #Plot the confusion matrix of the expected and predicted classes

    m_f_1 = f1_score(actual_encoded, predicted_encoded, average='macro')

    matrix = confusion_matrix(actual_encoded, predicted_encoded, labels=np.arange(0,len(class_names)-1,1), normalize='all')
    plot_confusion_matrix(matrix, classes=labels, normalize=True, title="Confusion Matrix ("+model_type+"), Macro F1 = "+str(round(m_f_1*100,2))+"%")

    plot_classification_report(actual_encoded, predicted_encoded, labels, show_plot=False)

    plt.figure()

    overall_f1 = f1_score(actual_encoded, predicted_encoded, average='macro')
    overall_precision = precision_score(actual_encoded, predicted_encoded, average='macro')
    overall_recall = recall_score(actual_encoded, predicted_encoded, average='macro')
    
    fp = matrix.sum(axis=0) - np.diag(matrix)
    fn = matrix.sum(axis=1) - np.diag(matrix)
    tp = np.diag(matrix)
    tn = matrix.sum() - (fp + fn + tp)

    fp = fp.astype(float)
    tn = tn.astype(float)

    specificity = tn / (tn + fp)
    specificity = np.mean(specificity)

    print("Overall F1: "+str(overall_f1))
    print("Overall Precision: "+str(overall_precision))
    print("Overall Recall: "+str(overall_recall))
    print("Overall Specificity: "+str(specificity))

    #Plot prediction accuracy percentages
    plt.bar(class_names, accuracy_of_predictions)
    plt.xticks(class_names, fontsize=7, rotation=30)
    plt.title(model_type.upper()+"\nOverall Accuracy = "+str(round(test_acc*100,2))+"%")
    x1,x2,_,_ = plt.axis()
    plt.axis((x1,x2,0,100))
    plt.ylabel("Accuracy of predictions (%)")
    plt.xlabel("Condition")
    plt.show()