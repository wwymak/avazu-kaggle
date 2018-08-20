import numpy as np
import pandas as pd
import xlearn as xl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

if __name__ == '__main__':
    # ffm_model = xl.FFMModel(task='binary',

    #                         lr=0.02,
    #                         epoch=10,
    #                         reg_lambda=0.002,
    #                         metric='f1')
    ffm_model = xl.create_ffm()
    param= {'task':'binary',
                            'lr':0.02,
                            'epoch':20,
                            'lambda':0.02,
                            'metric':'auc'}
    ffm_model.setTrain('data/train_ffm.txt')
    ffm_model.setValidate('data/train_ffm.txt')
    ffm_model.setSigmoid()
    ffm_model.fit(param, "./ffm_model.out")

    # ffm_model.fit('data/train_ffm.txt', "./model.out", eval_set='data/val_ffm.txt', )
    # y_pred = ffm_model.predict('data/val_ffm.txt')

