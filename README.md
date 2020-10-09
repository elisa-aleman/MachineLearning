# Machine Learning

I mainly use machine learning in my research, so I have collected a few methods that are useful for training a model and reviewing their performances.

Besides *tensorflow* and *tflearn*, for which I don't really have any helper functions since they're pretty well equipped on their own, I use these methods the most:

- **Support Vector Machine**
- **Gradient Boost Machine**
- **XGBoost**
- **LightGBM**
- **Logistic Regression**

Some of them need special installation, which I'll write here. In the future I'll write installation guides for Windows, but I work in MacOSX and Linux, so it's what I'll write for now.

## Model metrics: Binary and Multi-class K-folds Cross Validation

Here I wrote methods for Binary and Multi-class K-folds cross validation with returns for F1, Accuracy, Precision, Recall, counts for each prediction, and a confusion matrix if necessary. The pre-equipped methods usually only give the Accuracy, which is not the best measure for machine learning performance, or only the F1 value, which is the most useful but hard to analyze without the other components. 

- **F_score** : uses a list of predictions and a list of correct predictions. Outputs dictionary with F1, Accuracy, Precision, Recall, and counts.
- **F_score_Kfolds** : uses a nested list of predictions and a nested list of correct predictions, the nested list length is the k-folds *k*. Outputs dictionary with the average, standard deviation and list of results for F1, Accuracy, Precision, Recall, and a counts dictionary for each cycle.
- **F_score_multiclass** : makes a binary comparison for each class vs. not that class in a multiclass problem. uses a list of predictions and a list of correct predictions. Outputs dictionary with F1, Accuracy, Precision, Recall, and counts, as well as a confusion matrix. 
- **F_score_multiclass_Kfolds** : makes a binary comparison for each class vs. not that class in a multiclass problem. uses a nested list of predictions and a nested list of correct predictions, the nested list length is the k-folds *k*. Outputs dictionary with the average, standard deviation and list of results for F1, Accuracy, Precision, Recall, and a counts dictionary, as well as confusion matrices for each cycle.

## Model methods: Single train and K-folds train methods

Here I wrote methods for a single training process with training and testing data split for each of the following methods, as well as a weight vector or importance vector extraction for SVM and LightGBM.

- **Support Vector Machine** : Train once, train with k-folds, get weight vector
- **Gradient Boost Machine** : Train once, train with k-folds
- **XGBoost** : Train once, train with k-folds
- **LightGBM** : Train once, train with k-folds, get importance vector.
- **Logistic Regression** : Train once, train with k-folds

### Useful Methods in Model_methods

I also have a few other useful methods in this library, such as:

- **OneHot(Y)** : Change a list of integers like [1,2,0] into a One-Hot encoding, [[0,1,0],[0,0,1],[1,0,0]]
- **ReadyData()** : get X, Y, test_x, test_y from a numpy data file
- **getCurrentAverageError(model,test_x,test_y)** : Test training average loss squared
- **parse_predictions_binary_Probability_Cutoff(predicted_probs, probability_cutoff=0.5)** : Probability results (e.g. 0.85 confidence to being class 0, 0.15 to being class 1) converted to binary 0 or 1 using the cutoff value.

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

* **LDA()** : Simple gensim implementation of LDA model
* **HDP()** : Simple gensim implementation of HDP model
* **tSNE()** : Scikit-learn implementation of tSNE model

## Installing libraries

Some libraries I use in these methods have more complicated installation processes. For example XGBoost and LightGBM with GPU integration. Others are just as easy as a `pip install`. 

### Support Vector Machine, Gradient Boosting Machine and Logistic Regression

```
pip install sklearn
```

### Tensorflow and TFlearn

```
pip3 install tensorflow tflearn
```

### XGBoost

It's best to follow the [official guide](http://xgboost.readthedocs.io/en/latest/build.html ), but I think I had some trouble with GCC on a mac, so here is the steps I took:

#### MacOSX

First of all, you need Homebrew.
**If you don't have Homebrew**, click [here](https://brew.sh), or just do the following:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

**If you're behind a proxy and can't install Homebrew**, first configure your git proxy settings:
```
git config --global http.proxy http://{PROXY_HOST}:{PORT}
```
Replace your {PROXY_HOST} and your {PORT}.

Then install homebrew using proxy settings as well:
```
/bin/bash -c "$(curl -x {PROXY_HOST}:{PORT} -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```
And finally alias `brew` so it always uses your proxy settings:

```
alias brew="https_proxy={PROXY_HOST}:{PORT} brew"
```

**Now that you have homebrew:**

```
brew install gcc
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; cp make/config.mk ./config.mk
```

Open config.mk and uncomment these two lines
```
export CC = gcc
export CXX = g++
```

and replace these two lines into the num of your version(depending on your gcc-version)
```
export CC = gcc-9
export CXX = g++-9
```

To check the version
```
ls /usr/local/bin | grep gcc
ls /usr/local/bin | grep g++
```

and build using the following commands
```
make -j4
cd python-package; sudo python3 setup.py install
```

#### Linux

```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
cmake ..
make -j8
cd ../python-package
/usr/bin/python3 setup.py install
```

#### Linux with GPU integration:

```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
make -j8
cd ../python-package
/usr/bin/python3 setup.py install
```

### LightGBM

Best to follow the [official installation guide](https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version), but these are the steps I took. I installed in Linux with GPU integration, and on MacOSX with only CPU integration.

#### Linux with GPU integration:

For LightGBM, Boost and OpenCL need to be installed
```
apt-get install libboost-all-dev
apt install ocl-icd-libopencl1
apt install opencl-headers
apt install clinfo
apt install ocl-icd-opencl-dev
```

```
pip3 install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
```

#### MacOSX without GPU:

```
pip3 install lightgbm
```