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
    # X0 = sc.fit_transform(X0)
    # ds0["term"] = np.log1p(ds0["term"])
    y0 = ds0[['term']].values.ravel()
    '''
    model = CatBoostRegressor(iterations=50, max_depth=4, learning_rate=0.1,
                              task_type="GPU",
                              devices='0:1')
    model.fit(X0,
              y0,
              verbose=10)
    '''
    clf = GradientBoostingRegressor(n_estimators=50, max_depth=4, verbose=10)
    clf.fit(X0, y0)
    """
    clf = GradientBoostingRegressor() # {'n_estimators': 150, 'max_depth': 4}
    param_grid = {
        #'min_samples_split': [10, 30, 70, 100],
        'n_estimators': [50, 150, 250, 350, 500],
        'max_depth': [2, 4, 6, 8, 10]
    }
    n_iter_search = 25
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                       n_iter=n_iter_search, cv=3, verbose=5)


    random_search.fit(X0, y0)
    print("RandomizedSearchCV" )
    print("Best0 : ", random_search.best_params_)
    """
    print('Saving ModelMain0')
    # if not os.path.exists(cf.base_dir + '/models'):
    #    os.makedirs(cf.base_dir + '/models')
    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM0.joblib')


def Model_1(data: pd.DataFrame):
    ds1 = data[((data.price >= data.price.quantile(0.25)) & (data.price <= data.price.quantile(0.8)))]
    print('Data #1 length: ', ds1.shape)
    X1 = ds1.drop(['term'], axis=1)
    sc = StandardScaler()
    # X1 = sc.fit_transform(X1)

    # ds1["term"] = np.log1p(ds1["term"])
    y1 = ds1[['term']].values.ravel()

    '''
    model = CatBoostRegressor(iterations=50, max_depth=4, learning_rate=0.1,
                              task_type="GPU",
                              devices='0:1')
    model.fit(X1,
              y1,
              verbose=10)
    '''
    clf = GradientBoostingRegressor(n_estimators=50, max_depth=6, verbose=10)
    clf.fit(X1, y1)
    '''
    clf = GradientBoostingRegressor() # {'n_estimators': 50, 'max_depth': 4}
    param_grid = {
        #'min_samples_split': [10, 30, 70, 100],
        'n_estimators': [50, 150, 250, 350, 500],
        'max_depth': [2, 4, 6, 8, 10]
    }
    n_iter_search = 30
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                       n_iter=n_iter_search, cv=3, verbose=5)

    random_search.fit(X1, y1)
    print("RandomizedSearchCV")
    print("Best1 : ", random_search.best_params_)
    '''
    print('Saving ModelMain1')
    # if not os.path.exists(cf.base_dir + '/models'):
    #    os.makedirs(cf.base_dir + '/models')
    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM1.joblib')


def Model_2(data: pd.DataFrame):
    ds2 = data[(data.price > data.price.quantile(0.8))]
    print('Data #2 length: ', ds2.shape)
    X2 = ds2.drop(['term', 'offer_id'], axis=1)
    sc = StandardScaler()
    # X2 = sc.fit_transform(X2)

    # ds2["term"] = np.log1p(ds2["term"])
    y2 = ds2[['term']].values.ravel()

    '''
    model = CatBoostRegressor(iterations=50, max_depth=2, learning_rate=0.1,
                              task_type="GPU",
                              devices='0:1')
    model.fit(X2,
              y2,
              verbose=10)
        '''

    clf = GradientBoostingRegressor(n_estimators=350, max_depth=2, verbose=10)
    clf.fit(X2, y2)
    '''
    clf = GradientBoostingRegressor() #{'n_estimators': 50, 'max_depth': 6}
    param_grid = {
        #'min_samples_split': [10, 30, 70, 100],
        'n_estimators': [50, 150, 250, 350, 500],
        'max_depth': [2, 4, 6, 8, 10]
    }
    n_iter_search = 30
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                       n_iter=n_iter_search, cv=3, verbose=5)

    random_search.fit(X2, y2)
    print("RandomizedSearchCV")
    print("Best2 : ", random_search.best_params_)
    '''
    print('Saving ModelMain2')
    # if not os.path.exists(cf.base_dir + '/models'):
    #    os.makedirs(cf.base_dir + '/models')
    dump(clf, PATH_TO_TIME_MODEL + '/GBR_COORDINATES_TERM2.joblib')


def model():
    # Load and split dataset.
    ds = pd.read_csv(prepared_data + '/COORDINATES_Pred_Term.csv')
    ds = ds.iloc[:-100]
    # Model_0(ds)

    # Model_1(ds)

    # Model_2(ds)

if __name__ == '__main__':
    model()
