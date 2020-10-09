# Machine Learning

I mainly use machine learning in my research, so I have collected a few methods that are useful for training a model and reviewing their performances.

Besides *tensorflow* and *tflearn*, for which I don't really have any helper functions since they're pretty well equipped on their own, I use these methods the most:

- __Support Vector Machine__
- __Gradient Boost Machine__
- __XGBoost__
- __LightGBM__
- __Logistic Regression__

Some of them need special installation, which I'll write here. In the future I'll write installation guides for Windows, but I work in MacOSX and Linux, so it's what I'll write for now.

## Model metrics: Binary and Multi-class K-folds Cross Validation

Here I wrote methods for Binary and Multi-class K-folds cross validation with returns for F1, Accuracy, Precision, Recall, counts for each prediction, and a confusion matrix if necessary. The pre-equipped methods usually only give the Accuracy, which is not the best measure for machine learning performance, or only the F1 value, which is the most useful but hard to analyze without the other components. 

- __F_score__ : uses a list of predictions and a list of correct predictions. Outputs dictionary with F1, Accuracy, Precision, Recall, and counts.
- __F_score_Kfolds__ : uses a nested list of predictions and a nested list of correct predictions, the nested list length is the k-folds *k*. Outputs dictionary with the average, standard deviation and list of results for F1, Accuracy, Precision, Recall, and a counts dictionary for each cycle.
- __F_score_multiclass__ : makes a binary comparison for each class vs. not that class in a multiclass problem. uses a list of predictions and a list of correct predictions. Outputs dictionary with F1, Accuracy, Precision, Recall, and counts, as well as a confusion matrix. 
- __F_score_multiclass_Kfolds__ : makes a binary comparison for each class vs. not that class in a multiclass problem. uses a nested list of predictions and a nested list of correct predictions, the nested list length is the k-folds *k*. Outputs dictionary with the average, standard deviation and list of results for F1, Accuracy, Precision, Recall, and a counts dictionary, as well as confusion matrices for each cycle.

## Model methods: Single train and K-folds train methods

Here I wrote methods for a single training process with training and testing data split for each of the following methods, as well as a weight vector or importance vector extraction for SVM and LightGBM.

- __Support Vector Machine__ : Train once, train with k-folds, get weight vector
- __Gradient Boost Machine__ : Train once, train with k-folds
- __XGBoost__ : Train once, train with k-folds
- __LightGBM__ : Train once, train with k-folds, get importance vector.
- __Logistic Regression__ : Train once, train with k-folds

### Useful Methods in Model_methods

I also have a few other useful methods in this library, such as:

- __OneHot(Y)__ : Change a list of integers like [1,2,0] into a One-Hot encoding, [[0,1,0],[0,0,1],[1,0,0]]
- __ReadyData()__ : get X, Y, test_x, test_y from a numpy data file
- __getCurrentAverageError(model,test_x,test_y)__ : Test training average loss squared
- __parse_predictions_binary_Probability_Cutoff(predicted_probs, probability_cutoff=0.5)__ : Probability results (e.g. 0.85 confidence to being class 0, 0.15 to being class 1) converted to binary 0 or 1 using the cutoff value.

### Tensorflow reminder method

I usually run tensorflow in a server machine, so I like to remind myself how to see the Tensorboard results on my local machine.

```
To be able to see Tensorboard on your local machine after training on a server
    1. exit current server session
    2. connect again with the following command:
        ssh -L 16006:127.0.0.1:6006 -p [port] [user]@[server]
    3. execute in terminal")
        tensorboard --logdir= [your log directory]
    4. on local machine, open browser on:
        http://127.0.0.1:16006
```

### Other models I use less often

I use these less often. So much so, I don't guarantee they work correctly here. I left them here in case I come back to using them at some point.

* __LDA()__ : Simple gensim implementation of LDA model
* __HDP()__ : Simple gensim implementation of HDP model
* __tSNE()__ : Scikit-learn implementation of tSNE model

