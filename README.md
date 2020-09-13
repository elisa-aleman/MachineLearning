# MachineLearning Kfolds
K-folds with returns for F1, Accuracy, Precision, Recall and counts for each prediction

* __Support Vector Machine__
* __Gradient Boost Machine__
* __XGBoost__
* __LightGBM__
* __Logistic Regression__

## Useful Methods in Model_methods

* __OneHot(Y)__ : Change a list of integers like [1,2,0] into a One-Hot encoding, [[0,1,0],[0,0,1],[1,0,0]]
* __ReadyData()__ : get X, Y, test_x, test_y from a numpy data file
* __getCurrentAverageError(model,test_x,test_y)__ : Test training output
* __parse_predictions_binary_Probability_Cutoff(predicted_probs, probability_cutoff=0.5)__ : Probability results to binary 0 or 1 with cutoff.


## Tensorflow reminder methods 

```
To be able to see Tensorboard in your local machine after training on a server
    1. exit current server session
    2. connect again with the following command:
        ssh -L 16006:127.0.0.1:6006 -p [port] [user]@[server]
    3. execute in terminal")
        tensorboard --logdir= [your log directory]
    4. on local machine, open browser on:
        http://127.0.0.1:16006
```

## Models 

* __LDA()__ : Simple gensim implementation of LDA model
* __HDP()__ : Simple gensim implementation of HDP model
* __tSNE()__ : Scikit-learn implementation of tSNE model

