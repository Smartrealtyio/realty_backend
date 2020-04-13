import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
# import Realty.config as cf
from scipy import stats
from joblib import dump, load

from scipy import stats
import os
import settings_local as SETTINGS

prepared_data_VTOR = SETTINGS.DATA_SPB + '/SPB_VTOR.csv'
prepared_data_NEW = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'

PATH_TO_PRICE_MODEL_GBR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_GBR.joblib'
PATH_TO_PRICE_MODEL_RF = SETTINGS.MODEL_SPB + '/PriceModel_SPB_RF.joblib'
PATH_TO_PRICE_MODEL_LGBM = SETTINGS.MODEL_SPB + '/PriceModel_SPB_LGBM.joblib'





def Price_Main(data: pd.DataFrame):

    # Remove price and term outliers (out of 3 sigmas)
    data1 = data[(np.abs(stats.zscore(data.price)) < 3)]
    data2 = data[(np.abs(stats.zscore(data.term)) < 3)]


    data = pd.merge(data1, data2, on=list(data.columns), how='left')

    # Fill NaN if it appears after merging
    data[['term']] = data[['term']].fillna(data[['term']].mean())

    # Log Transformation
    data["longitude"] = np.log1p(data["longitude"])
    data["latitude"] = np.log1p(data["latitude"])
    data["full_sq"] = np.log1p(data["full_sq"])
    data["life_sq"] = np.log1p(data["life_sq"])
    data["kitchen_sq"] = np.log1p(data["kitchen_sq"])
    data["price"] = np.log1p(data["price"])
    X = data[['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
              'time_to_metro', 'floor_last', 'floor_first', 'clusters']]

    y = data[['price']].values.ravel()
    print(X.shape, y.shape, flush=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # GBR model
    gbr_model = GradientBoostingRegressor(n_estimators=350, max_depth=9, verbose=1, random_state=42,
                                          learning_rate=0.07)

    gbr_model.fit(X_train, y_train)
    gbr_preds = gbr_model.predict(X_test)
    print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds), flush=True)
    print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)

    print('Train on full dataset GBR Spb: ', flush=True)
    gbr_model.fit(X, y)

    print('Save model GBR Spb: ', flush=True)
    dump(gbr_model, PATH_TO_PRICE_MODEL_GBR)

    # RANDOM FOREST REGRESSOR
    RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1)

    RF.fit(X_train, y_train)

    rf_predicts = RF.predict(X_test)

    print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
    print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)

    print('Train on full dataset RF secondary Spb: ', flush=True)
    RF.fit(X, y)

    print('Save model RF secondary Spb: ', flush=True)
    dump(RF, PATH_TO_PRICE_MODEL_RF)


    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.1,
                               n_estimators=1250, max_depth=5, min_child_samples=1, verbose=False)

    lgbm_model.fit(X_train, y_train)
    lgbm_preds = lgbm_model.predict(X_test)
    print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
    print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)

    print('Train on full dataset LGBM secondary Spb: ', flush=True)
    lgbm_model.fit(X, y)

    print('Save model LGBM secondary Spb: ', flush=True)
    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM)


def learner():

    # Load Data Flats New and Data Secondary flats
    df1 = pd.read_csv(prepared_data_VTOR)
    df2 = pd.read_csv(prepared_data_NEW)

    # Concatenate two types of flats
    all_data = pd.concat([df1, df2], ignore_index=True)

    # Train models
    Price_Main(all_data)


if __name__ == '__main__':
    learner()
