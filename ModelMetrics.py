#-*- coding: utf-8 -*-

import numpy
import scipy 

###########################
######### Metrics #########
###########################

# precision = true_positives / (true_positives + false_positives)
# recall = true_positives / (true_positives + false_negatives)
# accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
# Correct Prediction, True Positive, True Negative, Incorrect Prediction, False Positive, False Negative
def F_score(test_y, y_preds):
    if type(test_y) == type(numpy.array([])):
        test_y = test_y.tolist()
    counts = {"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0}
    testsize = len(test_y)
    for y_set in zip(y_preds, test_y):
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
    results = {"precision": precision, "recall": recall, "accuracy": accuracy, "F1":F1, "counts":counts}
    return results

def F_score_Kfolds(true_ys_list, y_pred_list):
    precisions = []
    accuracies = []
    recalls = []
    f1s = []
    k = len(y_pred_list)
    counts = []
    for t in range(k):
        test_y = true_ys_list[t]
        y_preds = y_pred_list[t]
        results = F_score(test_y, y_preds)
        accuracies.append(results["accuracy"])
        recalls.append(results["recall"])
        precisions.append(results["precision"])
        f1s.append(results["F1"])
        counts.append(results["counts"])
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
            "std": stpr,
            "list":precisions
        },
        "recall": {
            "average": avre,
            "std": stre,
            "list": recalls
        },
        "accuracy": {
            "average": avac,
            "std": stac,
            "list": accuracies
        },
        "F1": {
            "average": avf1,
            "std": stf1,
            "list": f1s
        },
        "counts": counts
    }
    # results = [
    #     [avpr, stpr, precisions],
    #     [avre, stre, recalls],
    #     [avac, stac, accuracies],
    #     [avf1, stf1, f1s],
    #     counts
    #     ]
    return results

if __name__ == '__main__':
    pass
