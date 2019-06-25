#-*- coding: utf-8 -*-

import scipy
import numpy
import gensim
import sklearn
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from sklearn.linear_model import LogisticRegression
import lightgbm
import random
from libraries.ModelMetrics import F_score_Kfolds
from sklearn.model_selection import train_test_split
from scipy.special import softmax

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

######################################
############## Useful ################
######################################

def parse_predictions_binary_Probability_Cutoff(predicted_probs, probability_cutoff=0.5):
    y_preds = []
    for ypred in predicted_probs:
        if ypred>probability_cutoff:
            y_preds.append(1)
        elif ypred==probability_cutoff:
            y_preds.append(random.randint(0,1))
        else:
            y_preds.append(0)
    return y_preds

####################
### SVM Learning ###
####################

def SVM_Train(x, y, test_size, shuffle=True, kerne ='linear', C=1.0, gamma=0.001):
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    testsize = len(test_y)
    #Define classifier
    clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf.fit(train_x,train_y)
    #Test data
    y_preds = []
    for i in range(testsize):
        predicted = clf.predict(test_x[i].reshape(1,-1))[0]
        y_preds.append(predicted)
    return clf, test_y, y_preds

def SVM_Kfolds(x, y, k, kernel='linear', C=1.0, gamma=0.001):
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = SVM_Train(x, y, test_size, shuffle=True, kernel=kernel, C=C, gamma=gamma)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    results = F_score_Kfolds(true_ys_list, y_pred_list)
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

#########################
### Boosting Learning ###
#########################

def GBM_Train(x, y, test_size, shuffle=True, n_estimators=100, subsample=0.8, max_depth=3):
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    testsize = len(test_y)
    #Define classifier
    clf = GradientBoostingClassifier(n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    clf.fit(train_x,train_y)
    #Test data
    y_preds = []
    for i in range(testsize):
        predicted = clf.predict(test_x[i].reshape(1,-1))[0]
        y_preds.append(predicted)
    return clf, test_y, y_preds

def GBM_Kfolds(x, y, k, n_estimators=100, subsample=0.8, max_depth=3):
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = GBM_Train(x, y, test_size, shuffle=True, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    results = F_score_Kfolds(true_ys_list, y_pred_list)
    return results

########################
### XGBoost Learning ###
########################

# http://xgboost.readthedocs.io/en/latest/parameter.html
def XGBoost_Train(x, y, test_size, shuffle=True, probability_cutoff=0.5, max_depth=3, learning_rate=0.1, n_estimators=100, eta=1, silent=1, objective='binary:logistic', min_child_weight=1, num_round=2):
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    #Define classifier
    # specify parameters via map
    param = {'max_depth':max_depth, 'learning_rate':learning_rate, 'eta':eta, 'silent':silent, 'objective':objective, 'n_estimators':n_estimators}
    dtrain = xgboost.DMatrix(train_x, label=train_y)
    clf = xgboost.train(param, dtrain, num_round)
    #Test data
    dtest = xgboost.DMatrix(test_x)
    predicted_probs = clf.predict(dtest)
    if param["objective"].startswith("binary"):
        y_preds = parse_predictions_binary_Probability_Cutoff(predicted_probs, probability_cutoff=probability_cutoff)
    elif param["objective"].startswith("multi"):
        y_preds = predicted_probs.argmax(axis=1)
    return clf, test_y, y_preds

def XGBoost_Kfolds(x, y, k, probability_cutoff=0.5, max_depth=3, learning_rate=0.1, n_estimators=100, eta=1, silent=1, objective='binary:logistic', min_child_weight=1, num_round=2):
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = XGBoost_Train(x, y, test_size, shuffle=True, probability_cutoff=probability_cutoff, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, eta=eta, silent=silent, objective=objective, min_child_weight=min_child_weight, num_round=num_round)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    results = F_score_Kfolds(true_ys_list, y_pred_list)
    return results

#########################
### LightGBM Learning ###
#########################

def LightGBM_train(x, y, test_size = 0.1, shuffle=True, binary=True, probability_cutoff=0.5, params = None):
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    train_data = lightgbm.Dataset(train_x, label=train_y)
    validation_data =  train_data.create_valid(test_x, label=test_y)
    #Define classifier
    if not params:
        params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.05,
                "min_data": 10,
                "num_leaves": 31,
                "verbose": -1,
                "num_threads": 1,
                "max_bin": 255
            }
        if not binary:
            params["objective"]="multiclass"
            params["num_class"]=3
            params["metric"]="multi_logloss"
    gbm = lightgbm.train(params, train_data, valid_sets=validation_data)
    # Test data
    predicted_probs = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    if binary:
        y_preds = parse_predictions_binary_Probability_Cutoff(predicted_probs, probability_cutoff=probability_cutoff)
    else:
        y_preds = predicted_probs.argmax(axis=1)
    return gbm, test_y, y_preds

def LightGBM_Kfolds(x, y, k, binary=True, test_size = 0.1, probability_cutoff=0.5, params = None):
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        gbm, test_y, y_preds = LightGBM_train(x, y, test_size, shuffle=True, binary=binary, probability_cutoff=probability_cutoff, params=params)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    results = F_score_Kfolds(true_ys_list, y_pred_list)
    return results

###########################
### Logistic Regression ###
###########################

def LogisticRegression(x,y, test_size, shuffle=True):
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    testsize = len(test_y)
    #Define classifier
    clf = LogisticRegression()
    clf.fit(train_x,train_y)
    #Test data
    y_preds = []
    for i in range(1, testsize+1):
        predicted = clf.predict(test_x[i].reshape(1,-1))[0]
        y_preds.append(predicted)
    return clf, test_y, y_preds

def LogisticRegression_Kfolds(x,y,k):
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = LogisticRegression(x,y, test_size, shuffle=True)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    results = F_score_Kfolds(true_ys_list, y_pred_list)
    return results

if __name__ == '__main__':
    pass