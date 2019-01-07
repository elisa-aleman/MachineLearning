#-*- coding: utf-8 -*-

import scipy
import numpy
import gensim
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from sklearn.linear_model import LogisticRegression
import random

############################
########## Models ##########
############################

# corpus = vector >> using each title and its kinds of answers as different dimensions or "words"
# num_topics is the number of topics
# id2word = titles >> the column titles_answer are the "words" in our data
def LDA(vectorized, num_topics, vec_titles):
    vec_titles = [[i] for i in vec_titles]
    titles = gensim.corpora.Dictionary(vec_titles)
    vector = [[(key,int(val)) for key,val in enumerate(row) if int(val)!=0] for row in vectorized]
    lda = gensim.models.ldamodel.LdaModel(corpus=vector, num_topics=num_topics, id2word=titles)
    return lda

def HDP(vectorized, vec_titles):
    vec_titles = [[i] for i in vec_titles]
    titles = gensim.corpora.Dictionary(vec_titles)
    # vector = [[(key,int(val)) if val!=' ' else (key,0) for key,val in enumerate(row)] for row in vector]
    vector = [[(key,int(val)) for key,val in enumerate(row) if int(val)!=0] for row in vectorized]
    hdp = gensim.models.hdpmodel.HdpModel(corpus=vector, id2word=titles)
    return hdp

def tSNE(input_filename, output_filename, header=True, n_dim=2):
    if header:
        raw_data = numpy.genfromtxt(input_filename, delimiter=",", headerfilling_values=(0, 0, 0), skiprows=1)
    else:
        raw_data = numpy.genfromtxt(input_filename, delimiter=",", headerfilling_values=(0, 0, 0))
    compressed_data = sklearn.manifold.TSNE(n_dim).fit_transform(raw_data)
    numpy.savetxt(output_filename, compressed_data, delimiter=",")

#########################################
############## Vectorize ################
#########################################

def Vectorize(sentences, dictionary):
    # sentences: ["text",1 or 0] 1: positive, 0: negative
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    IM = vectorizer.fit_transform(dictionary)
    # Method 1
    X_list = []
    y_list = []
    for i in sentences:
        vector = vectorizer.transform([i[0]]).toarray().tolist()
        X_list.append(vector[0])
        y_list.append(i[1])
    X = numpy.array(X_list)
    y = numpy.array(y_list)
    return X, y

####################
### SVM Learning ###
####################

def SVM_Kfolds(x, y, k, kernel = 'linear', C = 1.0, gamma = 0.001):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(y)//k
    # Correct Prediction, True Positive, True Negative, Incorrect Prediction, False Positive, False Negative
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in xrange(k)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in xrange(k):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
        clf.fit(X[:-testsize],y[:-testsize])
        #Test data
        for i in range(1, testsize+1):
            predicted = clf.predict(X[-i].reshape(1,-1))[0]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FP"])
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FN"])
        else:
            recall = 0
        accuracy = counts[t]["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/len(precisions)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/len(recalls)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/len(accuracies)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/len(f1s)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results

def SVM_weights(x, y, feature_names, kernel = 'linear', C = 1.0, gamma = 0.001):
    if type(x) == type([]):
        x = numpy.array(x)
    if type(y) == type([]):
        y = numpy.array(y)
    clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf.fit(x,y)
    weights = clf.coef_.tolist()[0]
    influences = zip(feature_names, weights)
    return influences

def GBM_Kfolds(x, y, k, n_estimators=100, subsample=0.8, max_depth=3):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(y)//k
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in xrange(k)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in xrange(k):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        clf = GradientBoostingClassifier(n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        clf.fit(X[:-testsize],y[:-testsize])
        #Test data
        for i in range(1, testsize+1):
            predicted = clf.predict(X[-i].reshape(1,-1))[0]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FP"])
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FN"])
        else:
            recall = 0
        accuracy = counts[t]["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/len(precisions)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/len(recalls)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/len(accuracies)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/len(f1s)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results


# http://xgboost.readthedocs.io/en/latest/parameter.html
def XGBoost_Kfolds(x, y, k, probability_cutoff=0.5, max_depth=3, learning_rate=0.1, n_estimators=100, eta=1, silent=1, objective='binary:logistic', min_child_weight=1, num_round=2):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(y)//k
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in range(k)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in range(k):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        # specify parameters via map
        param = {'max_depth':max_depth, 'learning_rate':learning_rate, 'eta':eta, 'silent':silent, 'objective':objective, 'n_estimators':n_estimators}
        dtrain = xgboost.DMatrix(X[:-testsize], label=y[:-testsize])
        clf = xgboost.train(param, dtrain, num_round)
        #Test data
        dtest = xgboost.DMatrix(X[-testsize:])
        predicted_probs = clf.predict(dtest)
        ypreds = []
        for ypred in predicted_probs:
            if ypred>probability_cutoff:
                ypreds.append(1)
            elif ypred==probability_cutoff:
                ypreds.append(random.randint(0,1))
            else:
                ypreds.append(0)
        for i in range(1, testsize+1):
            predicted = ypreds[i-1]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FP"])
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FN"])
        else:
            recall = 0
        accuracy = counts[t]["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/len(precisions)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/len(recalls)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/len(accuracies)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/len(f1s)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results


def LogisticRegression_Kfolds(x,y,k):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(y)//k
    # Correct Prediction, True Positive, True Negative, Incorrect Prediction, False Positive, False Negative
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in xrange(k)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in xrange(k):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        clf = LogisticRegression()
        clf.fit(X[:-testsize],y[:-testsize])
        #Test data
        for i in range(1, testsize+1):
            predicted = clf.predict(X[-i].reshape(1,-1))[0]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FP"])
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FN"])
        else:
            recall = 0
        accuracy = counts[t]["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/len(precisions)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/len(recalls)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/len(accuracies)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/len(f1s)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results


if __name__ == '__main__':
    pass

