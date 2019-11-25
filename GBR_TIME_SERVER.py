import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, scorer, mean_squared_error
import logging
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA
PATH_TO_TIME_MODEL = SETTINGS.MODEL


def model():
    # Load and split dataset.
    ds = pd.read_csv(prepared_data + 'COORDINATES_Pred_Term.csv')

    # 0
    # Drop all flats with full_sq > 75
    ds0 = ds[((ds.term <= ds.term.quantile(0.2)))]
    print("ds0: ", ds0.shape)

    X0 = ds0.drop(['term'], axis=1)
    y0 = ds0[['term']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.1, random_state=42)

    clf = GradientBoostingRegressor(n_estimators=1000, max_depth=10, verbose=10)
    # clf = GradientBoostingRegressor(n_estimators=1300, subsample=1, max_features=11,
    #                                max_depth=12, learning_rate=0.1, verbose=10,
    #                                min_samples_split=4)

    clf.fit(X_train, y_train)
    print('Saving ModelMain')

    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM0.joblib')

    # 1
    # Drop all flats with full_sq > 75
    ds1 = ds[((ds.term > ds.term.quantile(0.2))&(ds.term < ds.term.quantile(0.85)))]
    print('ds1: ', ds1.shape)

    X1 = ds1.drop(['term'], axis=1)
    y1 = ds1[['term']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=42)



    clf = GradientBoostingRegressor(n_estimators=1000, max_depth=10, verbose=10)
    #clf = GradientBoostingRegressor(n_estimators=1300, subsample=1, max_features=11,
    #                                max_depth=12, learning_rate=0.1, verbose=10,
    #                                min_samples_split=4)

    clf.fit(X_train, y_train)
    print('Saving ModelMain')

    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM1.joblib')

    # 2
    # Drop all flats with full_sq > 75
    ds2 = ds[((ds.term >= ds.term.quantile(0.85)))]
    print("ds2: ", ds2.shape)

    X2 = ds2.drop(['term'], axis=1)
    y2 = ds2[['term']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.1, random_state=42)

    clf = GradientBoostingRegressor(n_estimators=1000, max_depth=10, verbose=10)
    # clf = GradientBoostingRegressor(n_estimators=1300, subsample=1, max_features=11,
    #                                max_depth=12, learning_rate=0.1, verbose=10,
    #                                min_samples_split=4)

    clf.fit(X_train, y_train)
    print('Saving ModelMain')

    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM2.joblib')


    #mse = mean_squared_error(y_test, clf.predict(X_test))
    #r2_score_val = r2_score(y_test, pd.Series(pred))

    #print("\nR2_score: ", r2_score_val)
    #print("MSE: %.4f" % mse)


if __name__ == '__main__':
    model()

