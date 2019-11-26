import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, scorer, mean_squared_error
import logging
from sklearn.preprocessing import LabelEncoder
# import RealtyTime.config as cf
from joblib import dump, load
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA
PATH_TO_TIME_MODEL = SETTINGS.MODEL


def Model_0(data: pd.DataFrame):
    data = data
    print("Data: ", data.shape)
    ds0 = data[(data.price < data.price.quantile(0.25))]
    print('Data #0 length: ', ds0.shape)
    X0 = ds0.drop(['term'], axis=1)
    sc = StandardScaler()
    X0 = sc.fit_transform(X0)
    ds0["term"] = np.log1p(ds0["term"])
    y0 = ds0[['term']]
    X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, y0, test_size=0.01, random_state=42)

    clf = GradientBoostingRegressor(n_estimators=50, max_depth=4, verbose=10)
    clf.fit(X0, y0)

    print('Saving ModelMain0')
    #if not os.path.exists(cf.base_dir + '/models'):
    #    os.makedirs(cf.base_dir + '/models')
    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM0.joblib')


def Model_1(data: pd.DataFrame):
    ds1 = data[((data.price >= data.price.quantile(0.25)) & (data.price <= data.price.quantile(0.8)))]
    print('Data #1 length: ', ds1.shape)
    X1 = ds1.drop(['term'], axis=1)
    sc = StandardScaler()
    X1 = sc.fit_transform(X1)

    ds1["term"] = np.log1p(ds1["term"])
    y1 = ds1[['term']]


    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.01, random_state=42)

    clf = GradientBoostingRegressor(n_estimators=150, max_depth=4, verbose=10)
    clf.fit(X1, y1)

    print('Saving ModelMain1')
    #if not os.path.exists(cf.base_dir + '/models'):
    #    os.makedirs(cf.base_dir + '/models')
    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM1.joblib')


def Model_2(data: pd.DataFrame):
    ds2 = data[(data.price > data.price.quantile(0.8))]
    print('Data #2 length: ', ds2.shape)
    X2 = ds2.drop(['term'], axis=1)
    sc = StandardScaler()
    X2 = sc.fit_transform(X2)

    ds2["term"] = np.log1p(ds2["term"])
    y2 = ds2[['term']]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.01, random_state=42)

    clf = GradientBoostingRegressor(n_estimators=50, max_depth=2, verbose=10)
    clf.fit(X2, y2)

    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM2.joblib')


def model():
    # Load and split dataset.
    ds = pd.read_csv(prepared_data + '/COORDINATES_Pred_Term.csv')
    ds = ds.iloc[:-100]
    Model_0(ds)

    Model_1(ds)

    Model_2(ds)

if __name__ == '__main__':
    model()
