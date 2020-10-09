#-*- coding: utf-8 -*-

import numpy
import scipy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter

###########################
######### Metrics #########
###########################

def F_score(y_true, y_pred, with_counts=True):
    '''
        Calculates F-score, also outputs accuracy, recall and precision, as well as a counter dictionary.

        The main calculations are as follows:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        For convenience, in the counter dictionary, the following abbreviations are used:
            CP: Correct Predictions
            TP: True Positive
            TN: True Negative
            IP: Incorrect Predictions
            FP: False Positive
            FN: False Negative

        :param (list or 1d array) y_true: list of true target values 
        :param (list or 1d array) y_pred: list of target predictions made by the learning machine
        :param (bool) with_counts: if true, returns the counts dictionary as part of the resulting output.

        :return: results dictionary with shape:
            {
                "precision":_,
                "recall":_,
                "accuracy":_,
                "F1":_,
                "counts": {
                    "CP":_,
                    "TP":_,
                    "TN":_,
                    "IP":_,
                    "FP":_,
                    "FN":_
                }
            }
    '''
    if type(y_true) == type(numpy.array([])):
        y_true = y_true.tolist()
    if type(y_pred) == type(numpy.array([])):
        y_pred = y_pred.tolist()
    if with_counts:
        counts = Counter({"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0})
        testsize = len(y_true)
        for y_set in zip(y_pred, y_true):
            predicted = y_set[0]
            true_value = y_set[1]
            if (predicted == true_value): # test data
                counts["CP"] += 1
                if predicted == 1:
                    counts["TP"] += 1
                else:
                    counts["TN"] += 1
            else:
                counts["IP"] += 1
                if predicted == 1:
                    counts["FP"] += 1
                else:
                    counts["FN"] += 1
        if counts["TP"]+counts["FP"]>0:
            precision = counts["TP"] / (counts["TP"] + counts["FP"])
        else:
            precision = 0
        if counts["TP"]+counts["FN"]>0:
            recall = counts["TP"] / (counts["TP"] + counts["FN"])
        else:
            recall = 0
        accuracy = counts["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
    else:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        F1 = f1_score(y_true, y_pred)
    results = {"precision": precision, "recall": recall, "accuracy": accuracy, "F1":F1}
    if with_counts:
        results["counts"]=counts
    return results

def F_score_Kfolds(true_ys_list, y_pred_list, with_counts=True, with_lists=True):
    '''
        This method is used after having performed a k-folds cross validation separately.
        The input is the nested lists of true values and predictions resulting of that k-folds method.
        The method calculates F-score, accuracy, recall and precision, as well as a counter dictionary in each cycle.
        The average and standard deviation values for each of those are saved in a dictionary, as well as a list of the results if specified.

        The main calculations are as follows:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        For convenience, in the counter dictionary, the following abbreviations are used:
            CP: Correct Predictions
            TP: True Positive
            TN: True Negative
            IP: Incorrect Predictions
            FP: False Positive
            FN: False Negative

        :param (nested list or 2d array) true_ys_list: nested list of true target values. The length of the list is the number k of folds used in the cross validation.
        :param (nested list or 2d array) y_pred_list: nested list of target predictions made by the learning machine. The length of the list is the number k of folds used in the cross validation.
        :param (bool) with_counts: if true, returns a list of the counts dictionaries as part of the resulting output for each cycle in the k-folds operation.
        :param (bool) with_lists: if true, returns the list of values used to calculate the average and standard deviation of each result.

        :return: results dictionary with shape:
            {
                "precision":{
                    "average":_,
                    "std": _,
                    "list": [...],
                },
                "recall": {
                    "average":_,
                    "std": _,
                    "list": [...],
                },
                "accuracy": {
                    "average":_,
                    "std": _,
                    "list": [...],
                },
                "F1": {
                    "average":_,
                    "std": _,
                    "list": [...],
                },
                "counts": [
                    {
                        "CP":_,
                        "TP":_,
                        "TN":_,
                        "IP":_,
                        "FP":_,
                        "FN":_
                    }, {...} ...
                ]
            }
    '''
    precisions = []
    accuracies = []
    recalls = []
    f1s = []
    k = len(y_pred_list)
    counts = []
    for t in range(k):
        y_true = true_ys_list[t]
        y_pred = y_pred_list[t]
        t_results = F_score(y_true, y_pred, with_counts=with_counts)
        accuracies.append(t_results["accuracy"])
        recalls.append(t_results["recall"])
        precisions.append(t_results["precision"])
        f1s.append(t_results["F1"])
        if with_counts:
            counts.append(t_results["counts"])
    avpr = sum(precisions)/len(precisions)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/len(recalls)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/len(accuracies)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/len(f1s)
    stf1 = scipy.std(f1s)
    results = {
        "precision": {
            "average": avpr,
            "std": stpr
        },
        "recall": {
            "average": avre,
            "std": stre
        },
        "accuracy": {
            "average": avac,
            "std": stac
        },
        "F1": {
            "average": avf1,
            "std": stf1
        }
    }
    if with_counts:
        results["counts"] = counts
    if with_lists:
        results["precision"]["list"]= precisions
        results["recall"]["list"]= recalls
        results["accuracy"]["list"]= accuracies
        results["F1"]["list"]= f1s
    return results

def F_score_multiclass(y_true, y_pred, with_counts=True, with_confusion_matrix=True):
    '''
        For a multi-class problem, calculates F-score, also outputs accuracy, recall and precision, as well as a counter dictionary.
        Because the F-score is inherently a binary measure in nature, this method employs a workaround.
        For each class, we compare the results of predictions of that class contrasting with all the rest of the classes combined into, what basically is "not this class".
        This way we can perform a binary classification method on each class separately, and then compare the results between them.

        For example, in a 3 class scenario: a,b,c.
            a results: a vs. b & c
            b results: b vs. a & c
            c results: c vs. a & b

        So the output is still a binary measure for each class in a multi-class problem.
        Because this is a multi-class problem, the method also outputs a confusion matrix.

        The main calculations are as follows:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        For convenience, in the counter dictionary, the following abbreviations are used:
            CP: Correct Predictions
            TP: True Positive
            TN: True Negative
            IP: Incorrect Predictions
            FP: False Positive
            FN: False Negative

        :param (list or 1d array) y_true: list of true target values 
        :param (list or 1d array) y_pred: list of target predictions made by the learning machine
        :param (bool) with_counts: if true, returns the counts dictionary as part of the resulting output.
        :param (bool) with_confusion_matrix: if true, returns the confusion matrix used in the multi-class analysis.

        :return: results dictionary with shape:
            {
                0:  {
                        "precision":_,
                        "recall":_,
                        "accuracy":_,
                        "F1":_,
                        "counts": {
                            "CP":_,
                            "TP":_,
                            "TN":_,
                            "IP":_,
                            "FP":_,
                            "FN":_
                        },
                        "confusion_matrix":_
                    },
                1: {...},
                2: {...},
                ...
                class_index_n: {...}
            } 
    '''
    cm = confusion_matrix(y_true, y_pred)
    results = {}
    counts = Counter({})
    testsize = len(y_true)
    for index, row in enumerate(cm):
        results[index] = {}
        counts["TP"] = cm[index][index]
        counts["TN"] = sum(numpy.delete(cm.diagonal(), index))
        counts["CP"] = counts["TN"]+counts["TP"]
        counts["FP"] = sum(numpy.delete(cm[:,index], index))
        counts["FN"] = sum(numpy.delete(cm[index,:], index))
        counts["IP"] = counts["FP"] + counts["FN"]
        if counts["TP"]+counts["FP"]>0:
            precision = counts["TP"] / (counts["TP"] + counts["FP"])
        else:
            precision = 0
        if counts["TP"]+counts["FN"]>0:
            recall = counts["TP"] / (counts["TP"] + counts["FN"])
        else:
            recall = 0
        accuracy = counts["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        results[index]["precision"] = precision
        results[index]["recall"] = recall
        results[index]["accuracy"] = accuracy
        results[index]["F1"] = F1
        if with_counts:
            results[index]["counts"] = counts
        if with_confusion_matrix:
            results[index]["confusion_matrix"] = cm
    return results

def F_score_multiclass_Kfolds(true_ys_list, y_pred_list, with_counts=True, with_lists=True, with_confusion_matrix=True):
    '''
        This method is used after having performed a k-folds cross validation separately.
        The input is the nested lists of true values and predictions resulting of that k-folds method.
        For a multi-class problem, calculates F-score, also outputs accuracy, recall and precision, as well as a counter dictionary in each cycle.
        The average and standard deviation values for each of those are saved in a dictionary, as well as a list of the results if specified.
        
        Because the F-score is inherently a binary measure in nature, this method employs a workaround.
        For each class, we compare the results of predictions of that class contrasting with all the rest of the classes combined into, what basically is "not this class".
        This way we can perform a binary classification method on each class separately, and then compare the results between them.

        For example, in a 3 class scenario: a,b,c.
            a results: a vs. b & c
            b results: b vs. a & c
            c results: c vs. a & b

        So the output is still a binary measure for each class in a multi-class problem.
        Because this is a multi-class problem, the method also outputs a confusion matrix.
        However, since it is applied several times in the cycles of k-folds, the method outputs, the sum, average and standard deviations of the confusion matrices, as well as a list of each occurrence.

        The main calculations are as follows:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        For convenience, in the counter dictionary, the following abbreviations are used:
            CP: Correct Predictions
            TP: True Positive
            TN: True Negative
            IP: Incorrect Predictions
            FP: False Positive
            FN: False Negative

        :param (nested list or 2d array) true_ys_list: nested list of true target values. The length of the list is the number k of folds used in the cross validation.
        :param (nested list or 2d array) y_pred_list: nested list of target predictions made by the learning machine. The length of the list is the number k of folds used in the cross validation.
        :param (bool) with_counts: if true, returns a list of the counts dictionaries as part of the resulting output for each cycle in the k-folds operation.
        :param (bool) with_lists: if true, returns the list of values used to calculate the average and standard deviation of each result.
        :param (bool) with_confusion_matrix: if true, returns the confusion matrix used in the multi-class analysis.

        :return: results dictionary with shape:
            {
                0:  {
                        "precision":{
                            "average":_,
                            "std": _,
                            "list": [...],
                        },
                        "recall": {
                            "average":_,
                            "std": _,
                            "list": [...],
                        },
                        "accuracy": {
                            "average":_,
                            "std": _,
                            "list": [...],
                        },
                        "F1": {
                            "average":_,
                            "std": _,
                            "list": [...],
                        },
                        "counts": [
                            {
                                "CP":_,
                                "TP":_,
                                "TN":_,
                                "IP":_,
                                "FP":_,
                                "FN":_
                            }, {...} ...
                        ]
                        "confusion_matrix": {
                            "sum":_,
                            "average":_,
                            "std":_,
                            "list": [...]
                        }
                    },
                1: {...},
                2: {...},
                ...
                class_index_n: {...}
            } 
    '''
    indexes = numpy.unique(y_pred_list[0])
    results = {}
    for index in indexes:
        precisions = []
        accuracies = []
        recalls = []
        f1s = []
        k = len(y_pred_list)
        counts = []
        cms = []
        for t in range(k):
            y_true = true_ys_list[t]
            y_pred = y_pred_list[t]
            t_results = F_score_multiclass(y_true, y_pred, with_counts=with_counts, with_confusion_matrix=with_confusion_matrix)
            accuracies.append(t_results[index]["accuracy"])
            recalls.append(t_results[index]["recall"])
            precisions.append(t_results[index]["precision"])
            f1s.append(t_results[index]["F1"])
            if with_counts:
                counts.append(t_results[index]["counts"])
            if with_confusion_matrix:
                cms.append(t_results[index]["confusion_matrix"])
        avpr = sum(precisions)/len(precisions)
        stpr = scipy.std(precisions)
        avre = sum(recalls)/len(recalls)
        stre = scipy.std(recalls)
        avac = sum(accuracies)/len(accuracies)
        stac = scipy.std(accuracies)
        avf1 = sum(f1s)/len(f1s)
        stf1 = scipy.std(f1s)
        results[index] = {
            "precision": {
                "average": avpr,
                "std": stpr
            },
            "recall": {
                "average": avre,
                "std": stre
            },
            "accuracy": {
                "average": avac,
                "std": stac
            },
            "F1": {
                "average": avf1,
                "std": stf1
            }
        }
        if with_counts:
            results[index]["counts"] = counts
        if with_lists:
            results[index]["precision"]["list"]= precisions
            results[index]["recall"]["list"]= recalls
            results[index]["accuracy"]["list"]= accuracies
            results[index]["F1"]["list"]= f1s
    if with_confusion_matrix:
            cm_sum = sum(cms)
            cmav = cm_sum/len(cms)
            cmstd = numpy.ndarray.std(numpy.array(cms), axis=0)
            results["confusion_matrix"] = {
                "sum": cm_sum,
                "average": cmav,
                "std": cmstd
            }
            if with_lists:
                results["confusion_matrix"]["list"] = cms
    return results

if __name__ == '__main__':
    pass
