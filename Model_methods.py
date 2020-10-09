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
import lightgbm
import random
from Model_metrics import F_score_multiclass_Kfolds
from sklearn.model_selection import train_test_split
from scipy.special import softmax

##############################
### Support Vector Machine ###
##############################

def SVM_Train(x, y, test_size=None,  shuffle=True,  C=1.0, kernel ='linear', gamma=0.001):
    '''
        Trains a Support Vector Classifier using the data and test_size given to split it into training data and testing data.
        Returns the classifier, and the predictions and true values for performance testing.

        Parameters for train_test_split:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        
        :param (float or int) test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                                         If int, represents the absolute number of test samples. 
                                         If None, the value is set to the complement of the train size. 
                                         If train_size is also None, it will be set to 0.25.
        :param (bool) shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

        Parameters for SVC:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        :param (float) C: Regularization parameter. The strength of the regularization is inversely proportional to C. 
                          Must be strictly positive. The penalty is a squared l2 penalty.
        :param (str) kernel: Specifies the kernel type to be used in the algorithm.
                                It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
                                If none is given, 'linear' will be used. If a callable is given it is
                                used to pre-compute the kernel matrix from data matrices; that matrix
                                should be an array of shape ``(n_samples, n_samples)``.
        :param (str or float) gamma: {'scale', 'auto'} or float, default=0.001. Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        
        
        :return:
            (SVC)  clf:     The SVC classifier object
            (list) test_y:  true values of y, used for model performance testing purposes
            (list) y_preds: predicted values of y, used for model performance testing purposes
    '''
    if test_size>0:
        train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    else:
        train_x = x
        train_y = y
        test_x = []
        test_y = []
    testsize = len(test_y)
    #Define classifier
    clf = svm.SVC(
        kernel = kernel,
        C = C,
        gamma = gamma
        )
    clf.fit(train_x,train_y)
    #Test data
    y_preds = []
    if test_size>0:
        for i in range(testsize):
            predicted = clf.predict(test_x[i].reshape(1,-1))[0]
            y_preds.append(predicted)
    return clf, test_y, y_preds

def SVM_Kfolds(x, y, k, kernel='linear', C=1.0, gamma=0.001, multiclass=False, with_counts=True, with_lists=True, with_confusion_matrix=True):
    '''
        Trains a Support Vector Classifier using the shuffled and split data for each cycle of a K-folds cross validation process.
        Then it calculates the performance of the SVC for each cycle and outputs the average performance results.

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        :param (int) k: Number of cycles for the k-folds cross validation. Test size is len(y)//k, and the data is shuffled each cycle.

        Parameters for SVC:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        :param (float) C: Regularization parameter. The strength of the regularization is inversely proportional to C. 
                          Must be strictly positive. The penalty is a squared l2 penalty.
        :param (str) kernel: Specifies the kernel type to be used in the algorithm.
                                It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
                                If none is given, 'linear' will be used. If a callable is given it is
                                used to pre-compute the kernel matrix from data matrices; that matrix
                                should be an array of shape ``(n_samples, n_samples)``.
        :param (str or float) gamma: {'scale', 'auto'} or float, default=0.001. Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        
        Parameters for the F-score method:
        
        :param (bool) multiclass: if true, uses the F_score_multiclass_Kfolds() method. If false, uses F_score_Kfolds() for output.
        :param (bool) with_counts: if true, returns a list of the counts dictionaries as part of the resulting output for each cycle in the k-folds operation.
        :param (bool) with_lists: if true, returns the list of values used to calculate the average and standard deviation of each result.
        :param (bool) with_confusion_matrix: if true, returns the confusion matrix used in the multi-class analysis.

        :return:
            if multiclass:
                results dictionary with shape:
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
        if not multiclass:
            results dictionary with shape:
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
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = SVM_Train(x, y, test_size, shuffle=True, kernel=kernel, C=C, gamma=gamma)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    if multiclass:
        results = F_score_multiclass_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists, with_confusion_matrix=with_confusion_matrix)
    else:
        results = F_score_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists)
    return results

def SVM_weights_trained(clf,keyword_list, min_df=1, token_pattern='(?u)\\b\\w+\\b'):
    '''
        For knowing the weight vector in an SVM used in text classification with the Bag Of Words method.
        Input a trained SVC classifier, the keyword list used to classify text and the settings used in the BOW process.
        Words with stronger weight will be closer to the dividing hyperplane, and will have a stronger impact on the decision for either class.
        High weighted keywords can be interpreted as vital for classification.

        :param (SVC) clf: SVC classifier object.
        :param (list of strings) keyword_list: list of keywords used in the Bag Of Words as features in the training process.
        
        Parameters of the CountVectorizer:

        :param (float [0.0, 1.0] or int) min_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
        :param (string) token_pattern: Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'. The default regexp select tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).

        :return:
            influences: zipped list of feature names and weight values
    '''
    weights = clf.coef_.tolist()[0]
    vectorizer = CountVectorizer(min_df=min_df, token_pattern=token_pattern)
    IM = vectorizer.fit_transform(keyword_list)
    feature_names = vectorizer.get_feature_names()
    influences = list(zip(feature_names, weights))
    return influences

#################################
### Gradient Boosting Machine ###
#################################

def GBM_Train(x, y, test_size, shuffle=True, n_estimators=100, subsample=0.8, max_depth=3):
    '''
        Trains a Gradient Boosting Machine using the data and test_size given to split it into training data and testing data.
        Returns the classifier, and the predictions and true values for performance testing.

        Parameters for train_test_split:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        
        :param (float or int) test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                                         If int, represents the absolute number of test samples. 
                                         If None, the value is set to the complement of the train size. 
                                         If train_size is also None, it will be set to 0.25.
        :param (bool) shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

        Parameters for GradientBoostingClassifier:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        :param (int) n_estimators: The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        :param (float) subsample: The fraction of samples to be used for fitting the individual base learners. 
                                  If smaller than 1.0 this results in Stochastic Gradient Boosting. 
                                  subsample interacts with the parameter n_estimators. 
                                  Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
        :param (int) max_depth: maximum depth of the individual regression estimators. 
                                The maximum depth limits the number of nodes in the tree. 
                                Tune this parameter for best performance; the best value depends on the interaction of the input variables.
        
        :return:
            (GBC)  clf:     The GradientBoostingClassifier object
            (list) test_y:  true values of y, used for model performance testing purposes
            (list) y_preds: predicted values of y, used for model performance testing purposes
    '''
    if test_size>0:
        train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    else:
        train_x = x
        train_y = y
        test_x = []
        test_y = []
    testsize = len(test_y)
    #Define classifier
    clf = GradientBoostingClassifier(n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    clf.fit(train_x,train_y)
    #Test data
    y_preds = []
    if test_size>0:
        for i in range(testsize):
            predicted = clf.predict(test_x[i].reshape(1,-1))[0]
            y_preds.append(predicted)
    return clf, test_y, y_preds

def GBM_Kfolds(x, y, k, n_estimators=100, subsample=0.8, max_depth=3, multiclass=False, with_counts= True, with_lists= True, with_confusion_matrix=True):
    '''
        Trains a Gradient Boosting Classifier using the shuffled and split data for each cycle of a K-folds cross validation process.
        Then it calculates the performance of the GBC for each cycle and outputs the average performance results.

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        :param (int) k: Number of cycles for the k-folds cross validation. Test size is len(y)//k, and the data is shuffled each cycle.

        Parameters for GradientBoostingClassifier:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        :param (int) n_estimators: The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        :param (float) subsample: The fraction of samples to be used for fitting the individual base learners. 
                                  If smaller than 1.0 this results in Stochastic Gradient Boosting. 
                                  subsample interacts with the parameter n_estimators. 
                                  Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
        :param (int) max_depth: maximum depth of the individual regression estimators. 
                                The maximum depth limits the number of nodes in the tree. 
                                Tune this parameter for best performance; the best value depends on the interaction of the input variables.
        
        Parameters for the F-score method:
        
        :param (bool) multiclass: if true, uses the F_score_multiclass_Kfolds() method. If false, uses F_score_Kfolds() for output.
        :param (bool) with_counts: if true, returns a list of the counts dictionaries as part of the resulting output for each cycle in the k-folds operation.
        :param (bool) with_lists: if true, returns the list of values used to calculate the average and standard deviation of each result.
        :param (bool) with_confusion_matrix: if true, returns the confusion matrix used in the multi-class analysis.

        :return:
            if multiclass:
                results dictionary with shape:
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
        if not multiclass:
            results dictionary with shape:
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
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = GBM_Train(x, y, test_size, shuffle=True, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    if multiclass:
        results = F_score_multiclass_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists, with_confusion_matrix=with_confusion_matrix)
    else:
        results = F_score_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists)
    return results

########################
### XGBoost Learning ###
########################

# http://xgboost.readthedocs.io/en/latest/parameter.html
def XGBoost_Train(x, y, test_size, shuffle=True, probability_cutoff=0.5, max_depth=3, learning_rate=0.1, eta=0.1, n_estimators=100, verbosity=1, objective='binary:logistic', min_child_weight=1, num_round=2):
    '''
        Trains an XGBoost using the data and test_size given to split it into training data and testing data.
        Returns the classifier, and the predictions and true values for performance testing.

        Parameters for train_test_split:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        
        :param (float or int) test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                                         If int, represents the absolute number of test samples. 
                                         If None, the value is set to the complement of the train size. 
                                         If train_size is also None, it will be set to 0.25.
        :param (bool) shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
        
        Parameters for parse_predictions_binary_Probability_Cutoff:
        
        :param (float) probability_cutoff: Probability cutoff point for binary class decisions.
                                           XGBoost returns probabilities of belonging to either class. In the case of binary predictions, it just returns one probability. 
                                           To be able to run performance tests, the cutoff decides it is class 1 when above it, or class 0 when below it.

        Parameters for XGBoost:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        :param (int) max_depth: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
        :param (float) learning_rate (alias eta): Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
        :param (int) n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.
        :param (int) verbosity: Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message. If there’s unexpected behaviour, please try to increase value of verbosity.
        :param (str) objective: There's more options in XGBoost, but since I only know binary or multiclass uses, my method only accepts these:
                binary:logistic: logistic regression for binary classification, output probability

                binary:logitraw: logistic regression for binary classification, output score before logistic transformation

                binary:hinge: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.

                multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)

                multi:softprob: same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata * nclass matrix. The result contains predicted probability of each data point belonging to each class.
        :param (int) min_child_weight: Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
        :param (int) num_round: The number of rounds for boosting


        :return:
            (XGBoost)   clf:     The XGBoost object
            (list)      test_y:  true values of y, used for model performance testing purposes
            (list)      y_preds: predicted values of y, used for model performance testing purposes
    '''
    if test_size>0:
        train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    else:
        train_x = x
        train_y = y
        test_x = []
        test_y = []
    #Define classifier
    # specify parameters via map
    param = {'max_depth':max_depth, 'learning_rate':learning_rate, 'eta':eta, 'verbosity':verbosity, 'objective':objective, 'n_estimators':n_estimators}
    dtrain = xgboost.DMatrix(train_x, label=train_y)
    clf = xgboost.train(param, dtrain, num_round)
    #Test data
    if test_size>0:
        dtest = xgboost.DMatrix(test_x)
        predicted_probs = clf.predict(dtest)
        if param["objective"].startswith("binary"):
            y_preds = parse_predictions_binary_Probability_Cutoff(predicted_probs, probability_cutoff=probability_cutoff)
        elif param["objective"].startswith("multi"):
            y_preds = predicted_probs.argmax(axis=1)
        else:
            y_preds = []
    else:
        y_preds = []
    return clf, test_y, y_preds

def XGBoost_Kfolds(x, y, k, probability_cutoff=0.5, max_depth=3, learning_rate=0.1, n_estimators=100, eta=1, silent=1, objective='binary:logistic', min_child_weight=1, num_round=2, with_counts=True, with_lists=True, with_confusion_matrix=True):
    '''
        Trains an XGBoost using the shuffled and split data for each cycle of a K-folds cross validation process.
        Then it calculates the performance of the GBC for each cycle and outputs the average performance results.

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        :param (int) k: Number of cycles for the k-folds cross validation. Test size is len(y)//k, and the data is shuffled each cycle.

        Parameters for parse_predictions_binary_Probability_Cutoff:
        
        :param (float) probability_cutoff: Probability cutoff point for binary class decisions.
                                           XGBoost returns probabilities of belonging to either class. In the case of binary predictions, it just returns one probability. 
                                           To be able to run performance tests, the cutoff decides it is class 1 when above it, or class 0 when below it.

        Parameters for XGBoost:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        :param (int) max_depth: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
        :param (float) learning_rate (alias eta): Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
        :param (int) n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.
        :param (int) verbosity: Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message. If there’s unexpected behaviour, please try to increase value of verbosity.
        :param (str) objective: There's more options in XGBoost, but since I only know binary or multiclass uses, my method only accepts these:
                binary:logistic: logistic regression for binary classification, output probability

                binary:logitraw: logistic regression for binary classification, output score before logistic transformation

                binary:hinge: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.

                multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)

                multi:softprob: same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata * nclass matrix. The result contains predicted probability of each data point belonging to each class.
        :param (int) min_child_weight: Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
        :param (int) num_round: The number of rounds for boosting

        Parameters for the F-score method:
        
        :param (bool) multiclass: if true, uses the F_score_multiclass_Kfolds() method. If false, uses F_score_Kfolds() for output.
        :param (bool) with_counts: if true, returns a list of the counts dictionaries as part of the resulting output for each cycle in the k-folds operation.
        :param (bool) with_lists: if true, returns the list of values used to calculate the average and standard deviation of each result.
        :param (bool) with_confusion_matrix: if true, returns the confusion matrix used in the multi-class analysis.

        :return:
            if objective starts with multi (multiclass):
                results dictionary with shape:
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
        if objective starts with binary:
            results dictionary with shape:
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
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = XGBoost_Train(x, y, test_size, shuffle=True, probability_cutoff=probability_cutoff, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, eta=eta, silent=silent, objective=objective, min_child_weight=min_child_weight, num_round=num_round)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    if objective.startswith("multi"):
        results = F_score_multiclass_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists, with_confusion_matrix=with_confusion_matrix)
    elif objective.startswith("binary"):
        results = F_score_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists)
    else:
        results = []
    return results

#########################
### LightGBM Learning ###
#########################

def LightGBM_train(x, y, test_size = 0.1, shuffle=True, binary=True, multiclass=False, n_class=2, params = None):
    '''
        Trains a LightGBM using the data and test_size given to split it into training data and testing data.
        Returns the classifier, and the predictions and true values for performance testing.
        The method has some default parameters, but they can be overwritten by the dictionary params.

        Parameters for train_test_split:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        
        :param (float or int) test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                                         If int, represents the absolute number of test samples. 
                                         If None, the value is set to the complement of the train size. 
                                         If train_size is also None, it will be set to 0.25.
        :param (bool) shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
        
        Parameter for binary or multiclass decision:
        :param (bool) binary: Vestigial parameter. if false, it sets parameters for LightGBM to use several classes. 
                            It is the opposite of the newer param multiclass, but some old projects are using this older param.
        :param (bool) multiclass: if true, it sets parameters for LightGBM to use several classes.
        :param (int) n_class: if multiclass, will pass to LightGBM the number of classes to use.

        Parameters for LightGBM:
        Originally, the method has more parameters available, but for simplicity I only use the following:
            
        :param (dict) params: Parameters for LightGBM. Consult https://lightgbm.readthedocs.io/en/latest/Parameters.html

        :return:
            (LightGBM)  clf:     The LightGBM object
            (list)      test_y:  true values of y, used for model performance testing purposes
            (list)      y_preds: predicted values of y, used for model performance testing purposes
    '''

    if binary and multiclass:
        binary=False
    if not binary and not multiclass:
        multiclass=True
    x = numpy.array(x)
    y = numpy.array(y)
    if test_size>0:
        train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    else:
        train_x = numpy.array(x)
        train_y = numpy.array(y)
        test_x = numpy.array([])
        test_y = numpy.array([])
    train_data = lightgbm.Dataset(train_x, label=train_y)
    validation_data =  train_data.create_valid(test_x, label=test_y)
    #Define classifier
    default_params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": 2,
            "learning_rate": 0.05,
            "min_data": 10,
            "num_leaves": 31,
            "verbose": -1,
            "num_threads": 1,
            "max_bin": 255
        }
    if multiclass:
        default_params["objective"]="multiclass"
        default_params["num_class"]= n_class
        default_params["metric"]="multi_logloss"
    params = default_params.update(params)
    if test_size>0:
        clf = lightgbm.train(params, train_data, valid_sets=validation_data)
    else:
        clf = lightgbm.train(params, train_data)
    # Test data
    if test_size>0:
        predicted_probs = clf.predict(test_x, num_iteration=clf.best_iteration)
        # LightGBM's prediction output is always multi-class shaped even in binary, so:
        y_preds = predicted_probs.argmax(axis=1)
    else:
        y_preds=[]
    return clf, test_y, y_preds

def LightGBM_Kfolds(x, y, k, binary=True, multiclass=False, n_class=2, params = None, with_counts=True, with_lists=True, with_confusion_matrix=True):
    '''
        Trains an LightGBM using the shuffled and split data for each cycle of a K-folds cross validation process.
        The method has some default parameters, but they can be overwritten by the dictionary params.
        Then it calculates the performance of the GBC for each cycle and outputs the average performance results.

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        :param (int) k: Number of cycles for the k-folds cross validation. Test size is len(y)//k, and the data is shuffled each cycle.

        Parameter for binary or multiclass decision:
        :param (bool) binary: Vestigial parameter. if false, it sets parameters for LightGBM to use several classes. 
                            It is the opposite of the newer param multiclass, but some old projects are using this older param.
        :param (bool) multiclass: if true, it sets parameters for LightGBM to use several classes.
        :param (int) n_class: if multiclass, will pass to LightGBM the number of classes to use.

        Parameters for LightGBM:
        Originally, the method has more parameters available, but for simplicity I only use the following:
            
        :param (dict) params: Parameters for LightGBM. Consult https://lightgbm.readthedocs.io/en/latest/Parameters.html
        
        Parameters for the F-score method:
        
        :param (bool) multiclass: if true, uses the F_score_multiclass_Kfolds() method. If false, uses F_score_Kfolds() for output.
        :param (bool) with_counts: if true, returns a list of the counts dictionaries as part of the resulting output for each cycle in the k-folds operation.
        :param (bool) with_lists: if true, returns the list of values used to calculate the average and standard deviation of each result.
        :param (bool) with_confusion_matrix: if true, returns the confusion matrix used in the multi-class analysis.

        :return:
            if objective starts with multi (multiclass):
                results dictionary with shape:
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
        if objective starts with binary:
            results dictionary with shape:
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
    if binary and multiclass:
        binary=False
    if not binary and not multiclass:
        multiclass=True
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = LightGBM_train(x, y, test_size, shuffle=True, binary=binary, multiclass=multiclass, n_class=n_class, params=params)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    if multiclass:
        results = F_score_multiclass_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists, with_confusion_matrix=with_confusion_matrix)
    else:
        results = F_score_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists)
    return results

def LightGBM_importance(clf, feature_names):
    '''
        For knowing the importance vector in a LightGBM with their feature names.

        :param (LightGBM object) clf: The LightGBM object
        :param (list of strings) feature_names: List of features used in the LightGBM training

        :return:
            importance_list: zipped list of feature names and importance values
    '''
    importance = clf.feature_importance()
    importance_list = list(zip(feature_names, importance))
    return importance_list

###########################
### Logistic Regression ###
###########################

def LogisticRegression(x,y, test_size, shuffle=True):
    '''
        Performs a logistic regression using the data and test_size given to split it into training data and testing data.
        Returns the classifier, and the predictions and true values for performance testing.

        Parameters for train_test_split:
        Originally, the method has more parameters available, but for simplicity I only use the following:

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        
        :param (float or int) test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                                         If int, represents the absolute number of test samples. 
                                         If None, the value is set to the complement of the train size. 
                                         If train_size is also None, it will be set to 0.25.
        :param (bool) shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
        
        :return:
            (LogisticRegression)  clf:     The LogisticRegression classifier object
            (list) test_y:  true values of y, used for model performance testing purposes
            (list) y_preds: predicted values of y, used for model performance testing purposes
    '''
    if test_size>0:
        train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    else:
        train_x = x
        train_y = y
        test_x = []
        test_y = []
    testsize = len(test_y)
    #Define classifier
    clf = LogisticRegression()
    clf.fit(train_x,train_y)
    #Test data
    y_preds = []
    if test_size>0:
        for i in range(1, testsize+1):
            predicted = clf.predict(test_x[i].reshape(1,-1))[0]
            y_preds.append(predicted)
    return clf, test_y, y_preds

def LogisticRegression_Kfolds(x,y,k, multiclass=False, with_counts=True, with_lists=True, with_confusion_matrix=True):
    '''
        Performs a logistic regression using the shuffled and split data for each cycle of a K-folds cross validation process.
        Then it calculates the performance of the logistic regression for each cycle and outputs the average performance results.

        *arrays: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        :param (array or indexable) x: input data
        :param (array or indexable) y: target data
        :param (int) k: Number of cycles for the k-folds cross validation. Test size is len(y)//k, and the data is shuffled each cycle.

        Parameters for the F-score method:
        
        :param (bool) multiclass: if true, uses the F_score_multiclass_Kfolds() method. If false, uses F_score_Kfolds() for output.
        :param (bool) with_counts: if true, returns a list of the counts dictionaries as part of the resulting output for each cycle in the k-folds operation.
        :param (bool) with_lists: if true, returns the list of values used to calculate the average and standard deviation of each result.
        :param (bool) with_confusion_matrix: if true, returns the confusion matrix used in the multi-class analysis.

        :return:
            if multiclass:
                results dictionary with shape:
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
        if not multiclass:
            results dictionary with shape:
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
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = LogisticRegression(x,y, test_size, shuffle=True)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    if multiclass:
        results = F_score_multiclass_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists, with_confusion_matrix=with_confusion_matrix)
    else:
        results = F_score_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists)
    return results

######################################
############## Useful ################
######################################

def OneHot(Y):
    '''
        Change a list of integers like [1,2,0] into a One-Hot encoding, [[0,1,0],[0,0,1],[1,0,0]]

        :param (list or 1d array) Y: list denoting multiclass targets by their index number, like [1,2,0]

        :return:
            (2d numpy array) oneHotY: One-Hot encoding of the input, like [[0,1,0],[0,0,1],[1,0,0]]
    '''
    uniqueY = numpy.unique(Y)
    oneHotY = numpy.zeros([Y.shape[0], uniqueY.shape[0]])
    for num, i in enumerate(Y):
        oneHotY[num][i] = 1
    return oneHotY

# get X, Y, test_x, test_y
def ReadyData(data, test_size = 1000, do_shuffle=True):
    '''
        get X, Y, test_x, test_y from a numpy data file

        :param (numpy array) data: array with pairs of x and y data points:
            [[[x1],[y1]],
             [[x2],[y2]],
             ...
             [[xn],[yn]]
            ]
        :param (float or int) test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                                         If int, represents the absolute number of test samples. 
                                         If None, the value is set to the complement of the train size. 
                                         If train_size is also None, it will be set to 0.25.
        :param (bool) shuffle: Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

        :return:
            X,Y,test_x,test_y: split numpy arrays
    '''
    x, y = zip(*data)
    X, test_x, Y, test_y = train_test_split(x,y, test_size=test_size, shuffle=do_shuffle)
    return X,Y,test_x,test_y

def getCurrentAverageError(model,test_x,test_y):
    '''
        Test training average loss squared

        :param (model) model: Any classifier that has a .predict() method.
        :param (numpy array) test_x, test_y: test data

        :return:
            (float) av_error: average loss squared
    '''
    pred_y = [p[0] for p in model.predict(test_x)]
    losses = [(i[0]-i[1])**2 for i in zip(pred_y,test_y)]
    mean_square_man = numpy.average(losses)
    av_error = mean_square_man**0.5
    return av_error

def parse_predictions_binary_Probability_Cutoff(predicted_probs, probability_cutoff=0.5):
    '''
        Probability results converted to binary 0 or 1 using the cutoff value.
        (e.g. 0.85 confidence being class 1, 0.15 being class 0 with threshold 0.5)

        :param (list or 1d-array) predicted_probs: Results from a classifier in the shape of [0.0 to 1.0] probability of belonging to class 1.
        :param (float) probability_cutoff: Probability cutoff point for binary class decisions.
                                           To be able to run performance tests, the cutoff decides it is class 1 when above it, or class 0 when below it.
        
        :return:
            (list) y_preds: Predictions in binary form.
    '''
    y_preds = []
    for ypred in predicted_probs:
        if ypred>probability_cutoff:
            y_preds.append(1)
        elif ypred==probability_cutoff:
            y_preds.append(random.randint(0,1))
        else:
            y_preds.append(0)
    return y_preds


##############################################
####### Tensorboard helpful methods ##########
##############################################

def print_log_instructions():
    '''
    Instructions to check the tensorboard log from a local machine after training on a server.
    '''
    print("To be able to see Tensorboard on your local machine after training on a server")
    print("    1. exit current server session")
    print("    2. connect again with the following command:")
    print("        ssh -L 16006:127.0.0.1:6006 -p [port] [user]@[server]")
    print("    3. execute in terminal")
    print("        tensorboard --logdir='{}'".format(MakeLogFile('', server=True)))
    print("    4. on local machine, open browser on:")
    print("        http://127.0.0.1:16006")


##################################
########## Other Models ##########
##################################

def LDA(vectorized, num_topics, vec_titles):
    '''
        Returns Latent Dirichlet allocation model for text analysis and topic detection. Still not sure how it works fully. No warranties.
        corpus = vector >> using each title and its types of answers as different dimensions or "words"
        id2word = titles >> the column titles_answer are the "words" in our data

        :param (array) vectorized: Vectorized Bag of Words for the corpus.
        :param (int) num_topics: Number of topics to split the data into.
        :param (list) vec_titles: list of words used as features in the Bag of Words vector.

        :return:
            (LdaModel) lda: LdaModel object after applying to corpus.
    '''
    vec_titles = [[i] for i in vec_titles]
    titles = gensim.corpora.Dictionary(vec_titles)
    vector = [[(key,int(val)) for key,val in enumerate(row) if int(val)!=0] for row in vectorized]
    lda = gensim.models.ldamodel.LdaModel(corpus=vector, num_topics=num_topics, id2word=titles)
    return lda

def HDP(vectorized, vec_titles):
    '''
        Returns Hierarchical Dirichlet process model for clustering data. Still not sure how it works fully. No warranties.
        corpus = vector >> using each title and its types of answers as different dimensions or "words"
        id2word = titles >> the column titles_answer are the "words" in our data

        :param (array) vectorized: Vectorized Bag of Words for the corpus.
        :param (list) vec_titles: list of words used as features in the Bag of Words vector.

        :return:
            (HdpModel) hdp: HdpModel object after applying to corpus.
    '''
    vec_titles = [[i] for i in vec_titles]
    titles = gensim.corpora.Dictionary(vec_titles)
    vector = [[(key,int(val)) for key,val in enumerate(row) if int(val)!=0] for row in vectorized]
    hdp = gensim.models.hdpmodel.HdpModel(corpus=vector, id2word=titles)
    return hdp

def tSNE(input_filename, output_filename, header=True, n_dim=2):
    '''
        t-distributed stochastic neighbor embedding
        This method is for visualizing multidimensional data in a lower dimension by using compression via the embedding method.

        :param (path) input_filename: path to the data input file
        :param (path) output_filename: path to the output file
        :param (bool) header: if true, skips the first row of the input file
        :param (int) n_dim: Number of dimensions to compress the data into.

        :output:
            Not returned but saved to output file, the tSNE compressed data form
    '''
    if header:
        raw_data = numpy.genfromtxt(input_filename, delimiter=",", headerfilling_values=(0, 0, 0), skiprows=1)
    else:
        raw_data = numpy.genfromtxt(input_filename, delimiter=",", headerfilling_values=(0, 0, 0))
    compressed_data = sklearn.manifold.TSNE(n_dim).fit_transform(raw_data)
    numpy.savetxt(output_filename, compressed_data, delimiter=",")


if __name__ == '__main__':
    pass
