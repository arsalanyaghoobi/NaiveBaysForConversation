import pandas as pd
import openpyxl
import warnings
from collections import defaultdict, Counter
import numpy as np
from math import log

class Charity():
    def __init__(self):
        self.prior = defaultdict(int) #
        self.likelihood = defaultdict(dict)#
        self.class_dict = {'RI': 0, 'WD': 1, 'RP': 2, 'CC':3, 'O':4} #
        self.total_document = 0 #
        self.class_document_count = defaultdict(int) #
        self.class_feature_count = defaultdict(Counter) #
        self.feature_count = defaultdict(int)#
        self.prior_array = np.zeros(len(self.class_dict), )#
        self.likelihood_array = [] #
        self.feature_index = defaultdict(int) #
        self.class_count = defaultdict(int) #
        self.class_count_test = defaultdict(int) #

    def training(self, link):
        xls = pd.ExcelFile(link)
        data = pd.read_excel(xls, 'final_master')
        document_index = data.index[(data['Annotation_3']== "RI")|
                                  (data['Annotation_3']== "WD")|
                                  (data['Annotation_3']== "RP")|
                                  (data['Annotation_3']== "CC")|
                                  (data['Annotation_3'] == "O")]
        # print(document_index)
        self.total_document = len(document_index)
        for index in document_index:
            class_name = data.loc[index, "Annotation_3"]
            self.class_document_count[class_name] += 1
            text = data.loc[index, "Text"]
            text = text.split()
            for word in text:
                self.class_count[class_name] +=1
                self.class_feature_count[class_name][word] += 1
                self.feature_count[word] += 1

        feature_set_list = list(self.feature_count.keys())
        self.likelihood_array = np.zeros((len(self.class_dict), len(self.feature_count)))
        for clas in self.class_dict:
            self.prior[clas] = log(self.class_document_count[clas] / self.total_document)
            if clas == 'RI':
                self.prior_array[0] = self.prior[clas]
            elif clas == 'WD':
                self.prior_array[1] = self.prior[clas]
            elif clas == 'RP':
                self.prior_array[2] = self.prior[clas]
            elif clas == 'CC':
                self.prior_array[3] = self.prior[clas]

            for feature in self.feature_count:
                self.feature_index[feature] = feature_set_list.index(feature)
                self.likelihood[clas][feature] = log((self.class_feature_count[clas][feature] + 1) /
                                                     (self.class_count[clas] + len(self.feature_count)))
                self.likelihood_array[self.class_dict[clas]][self.feature_index[feature]] = self.likelihood[clas][
                    feature]

    def test(self, link):
        results = defaultdict(dict)
        feature_vector = np.zeros((len(self.feature_count)))
        xls = pd.ExcelFile(link)
        data = pd.read_excel(xls, 'final_master')
        document_index = data.index[(data['Annotation_3'] == "RI") |
                                    (data['Annotation_3'] == "WD") |
                                    (data['Annotation_3'] == "RP") |
                                    (data['Annotation_3'] == "CC") |
                                    (data['Annotation_3'] == "O")]
        for index in document_index:
            class_name = data.loc[index, "Annotation_3"]
            self.class_document_count[class_name] += 1
            text = data.loc[index, "Text"]
            text = text.split()
            for word in text:
                self.class_count_test[class_name]+=1
                if word in self.feature_count:
                    word_index = self.feature_index[word]
                    feature_vector[word_index] += 1
            result_matrix = np.dot(self.likelihood_array, feature_vector)
            final_matrix = np.add(result_matrix, self.prior_array)
            max_index = np.argmax(final_matrix)
            if max_index == 0:
                prediction = 'RI'
            elif max_index == 1:
                prediction = 'WD'
            elif max_index == 2:
                prediction = 'RP'
            elif max_index == 3:
                prediction = 'CC'
            elif max_index == 4:
                prediction ='O'
            results[data.loc[index, "INDEX"]]['correct'] = class_name
            results[data.loc[index, "INDEX"]]['predicted'] = prediction
        # print(results)
        return results

    def evaluate(self, results):
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        precision = defaultdict(float)
        recall = defaultdict(float)
        F1 = defaultdict(float)
        for items in results:
            if results[items]['correct'] == results[items]['predicted']:
                if results[items]['predicted'] == 'RI':
                    confusion_matrix[0][0] += 1
                if results[items]['predicted'] == 'WD':
                    confusion_matrix[1][1] += 1
                if results[items]['predicted'] == 'RP':
                    confusion_matrix[2][2] += 1
                if results[items]['predicted'] == 'CC':
                    confusion_matrix[3][3] += 1
                if results[items]['predicted'] == 'O':
                    confusion_matrix[4][4] += 1
            if results[items]['correct'] != results[items]['predicted']:
                if results[items]['correct'] == 'RI':
                    if results[items]['predicted'] == 'WD':
                        confusion_matrix[0][1] += 1
                    if results[items]['predicted'] == 'RP':
                        confusion_matrix[0][2] += 1
                    if results[items]['predicted'] == 'CC':
                        confusion_matrix[0][3] += 1
                    if results[items]['predicted'] == 'O':
                        confusion_matrix[0][4] += 1
                if results[items]['correct'] == 'WD':
                    if results[items]['predicted'] == 'RI':
                        confusion_matrix[1][0] +=1
                    if results[items]['predicted'] == 'RP':
                        confusion_matrix[1][2] +=1
                    if results[items]['predicted'] == 'CC':
                        confusion_matrix[1][3] +=1
                    if results[items]['predicted'] == 'CC':
                        confusion_matrix[1][4] +=1
                if results[items]['correct'] == 'RP':
                    if results[items]['predicted'] == 'RI':
                        confusion_matrix[2][0] +=1
                    if results[items]['predicted'] == 'WD':
                        confusion_matrix[2][1] +=1
                    if results[items]['predicted'] == 'CC':
                        confusion_matrix[2][3] +=1
                    if results[items]['predicted'] == 'O':
                        confusion_matrix[2][4] +=1
                if results[items]['correct'] == 'CC':
                    if results[items]['predicted'] == 'RI':
                        confusion_matrix[3][0] +=1
                    if results[items]['predicted'] == 'WD':
                        confusion_matrix[3][1] +=1
                    if results[items]['predicted'] == 'RP':
                        confusion_matrix[3][2] += 1
                    if results[items]['predicted'] == 'O':
                        confusion_matrix[3][4] += 1
                if results[items]['correct'] == 'O':
                    if results[items]['predicted'] == 'RI':
                        confusion_matrix[4][0] +=1
                    if results[items]['predicted'] == 'WD':
                        confusion_matrix[4][1] +=1
                    if results[items]['predicted'] == 'RP':
                        confusion_matrix[4][2] += 1
                    if results[items]['predicted'] == 'CC':
                        confusion_matrix[4][3] += 1

        print(confusion_matrix)
        print(" ")
        for clas in self.class_dict:
            if clas == 'RI':
                precision[clas] = confusion_matrix[0][0] / np.sum(confusion_matrix[0, 1:])
                recall[clas] = confusion_matrix[0][0] / np.sum(confusion_matrix[:, 0])
                F1[clas] = (2 * precision[clas] * recall[clas]) / (precision[clas] + recall[clas])
                print("Precision for ", clas, " is ", precision[clas])
                print("Recall for ", clas, " is ", recall[clas])
                print("F1 for ", clas, " is ", F1[clas])
            if clas == 'WD':
                precision[clas] = confusion_matrix[1][1] / confusion_matrix[1][0]+ np.sum(confusion_matrix[1, 2:])
                recall[clas] = confusion_matrix[1][1] / np.sum(confusion_matrix[:,1])
                F1[clas] = (2 * precision[clas] * recall[clas]) / (precision[clas] + recall[clas])
                print("Precision for ", clas, " is ", precision[clas])
                print("Recall for ", clas, " is ", recall[clas])
                print("F1 for ", clas, " is ", F1[clas])
            if clas == 'RP':
                precision[clas] = confusion_matrix[2][2] / (np.sum(confusion_matrix[2,:2])+ np.sum(confusion_matrix[2,3:]))
                recall[clas] = confusion_matrix[2][2] / np.sum(confusion_matrix[:,2])
                F1[clas] = (2 * precision[clas] * recall[clas]) / (precision[clas] + recall[clas])
                print("Precision for ", clas, " is ", precision[clas])
                print("Recall for ", clas, " is ", recall[clas])
                print("F1 for ", clas, " is ", F1[clas])
            if clas == 'CC':
                precision[clas] = confusion_matrix[3][3] / (np.sum(confusion_matrix[3,:3] + np.sum(confusion_matrix[3,4])))
                recall[clas] = confusion_matrix[3][3] / np.sum(confusion_matrix[:,3])
                F1[clas] = (2 * precision[clas] * recall[clas]) / (precision[clas] + recall[clas])
                print("Precision for ", clas, " is ", precision[clas])
                print("Recall for ", clas, " is ", recall[clas])
                print("F1 for ", clas, " is ", F1[clas])
            if clas == 'O':
                precision[clas] = confusion_matrix[4][4] / np.sum(confusion_matrix[0, :4])
                recall[clas] = confusion_matrix[4][4] / np.sum(confusion_matrix[:, 4])
                F1[clas] = (2 * precision[clas] * recall[clas]) / (precision[clas] + recall[clas])
                print("Precision for ", clas, " is ", precision[clas])
                print("Recall for ", clas, " is ", recall[clas])
                print("F1 for ", clas, " is ", F1[clas])

        accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1] +
                     confusion_matrix[2][2] + confusion_matrix[3][3] + confusion_matrix[4][4])/ np.sum(confusion_matrix[:,:])
        print('Overal Accuracy: ', accuracy)
        print(np.sum(confusion_matrix))
        print(self.class_count)
        print(self.class_count_test)


if __name__ == '__main__':
    train_link = "D:\\Brandeis University\\Second Semester\\Machine Learning " \
                 "Annotation\\Final Project\\Coding for final assignemnt\\train.xlsx"
    test_link = "D:\\Brandeis University\\Second Semester\\Machine Learning " \
                "Annotation\\Final Project\\Coding for final assignemnt\\test.xlsx"
    naive_runner = Charity()
    naive_runner.training(train_link)
    result = naive_runner.test(test_link)
    naive_runner.evaluate(result)