### Avazu click through rate prediction

#### Task:

Given 11 days of clicks/not clicks data from Avazu, together with 
various info about the ad content, placement, etc, can we predict whether
an ad will get clicked on.

From [this](https://www.kaggle.com/c/avazu-ctr-prediction) kaggle competition.

#### Challenges:
- very big data size: the whole training dataset is over 6GB. In memory
sklearn transforms and models pretty much cause the python process
to run out of memory (even on a AWS machine with 32GB of RAM)
- feature engineering: should we use only the more 'content' based features
e.g. type of ad, banner size etc, or should we add in the more detailed
'user' based features, e.g. ip address, browser etc. Also, if we are using
the 'user' features, how to reduce dimension size after one hot encoding?

#### Initial attempts:

1. Data exploration on a small subset of the data (500000 entries)
    - notebook [here](https://github.com/wwymak/avazu-kaggle/blob/master/avazu-ctr-explore.ipynb)
    - some of the more generic features such as 'banner_pos', 'device_type', 'device_conn_type',
 as well as some of the 'mystery' features e.g. C1 seems to have an influence
 on CTR when looking at them in isolation
    - next trial-- can a model be trained based on this subset and only these
 features to predict clicks with reasonable accuracy (and precision and recall 
 balance)
2. Initial modelling
    - notebook [here](https://github.com/wwymak/avazu-kaggle/blob/master/avazu-ctr-initial-modelling.ipynb)
    - tried out a few different modelling algorithms from different families, but
    they all have pretty bad results...
    - could be because of the high class imbalance (only around 17% of the
    data is clicks vs 83% non clicks), or because the more important features 
    have been excluded, or just that the hyperparameters are way off.

3. Trying some more 
    - tested out xlearn's factorisation machine algorithm (since from 
    kaggle/ literature this is one of the good algorithms for CTR prediction)(code 
    in https://github.com/wwymak/avazu-kaggle/blob/master/avazu-modelling-ffm.ipynb and https://github.com/wwymak/avazu-kaggle/blob/master/ffm-modelling-xlearn.py)
    - unfortunately still not any better than the other algorithms
    - tried using the subset, more features, and after one hot encoding, 
    do feature selection with PCA-- didn't work as the PCA process ran out
    of memory
    

#### ToDo
- test out balancing the classes either with changing the threshold with predict_proba or using
the class_weights option in sklearn (different models may or may not have
this param and might be called slightly different names...)
- out of core ML with dask-ml/xlearn/vowpal wabbit (and possible models in sklearn that have
the option to do partial_fit, e.g. logistic regression, and also gradient
boosting that runs on gpu)
- entity embeddings as a feature selector
- model ensembles 
- lots of hyperparameter tuning (hyperopt or randomCV probably)
- after getting the models to have a reasonable precision/recall evaluate
them on multiple metrics to choose the best one. (or use ensemble/voting prediction of the
 good ones)

