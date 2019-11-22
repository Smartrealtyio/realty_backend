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
    ds = pd.read_csv(prepared_data + '/COORDINATES_Pred_Term.csv')

    X = ds.drop(['term'], axis=1)
    y = ds[['term']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    clf = GradientBoostingRegressor(n_estimators=200, max_depth=12, verbose=10)
    # clf = GradientBoostingRegressor(n_estimators=1300, subsample=1, max_features=12,
    #                                 max_depth=12, learning_rate=0.1, verbose=10,
    #                                 min_samples_split=4)

    clf.fit(X_train, y_train)

    # Save model
    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM.joblib')


if __name__ == '__main__':
    model()

