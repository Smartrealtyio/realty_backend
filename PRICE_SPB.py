import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
# import Realty.config as cf
from scipy import stats
from joblib import dump, load
import xgboost
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA_SPB

PATH_TO_PRICE_MODEL_GBR = SETTINGS.MODEL_SPB + '/PriceGBR_SPB.joblib'
PATH_TO_PRICE_MODEL_CAT = SETTINGS.MODEL_MOSCOW + '/PriceCat_SPB.joblib'



def Model(data: pd.DataFrame):


    data = data[(np.abs(stats.zscore(data.price)) < 3)]
    data = data[(np.abs(stats.zscore(data.term)) < 3)]
    data["longitude"] = np.log1p(data["longitude"])
    data["latitude"] = np.log1p(data["latitude"])
    data["full_sq"] = np.log1p(data["full_sq"])
    data["kitchen_sq"] = np.log1p(data["kitchen_sq"])
    data["X"] = np.log1p(data["X"])
    data["Y"] = np.log1p(data["Y"])
    data["price"] = np.log1p(data["price"])
    X1 = data[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
               'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y', 'clusters']]
    y1 = data[['price']].values.ravel()
    print(X1.shape, y1.shape, flush=True)

    gbr = GradientBoostingRegressor(n_estimators=350, max_depth=8, verbose=5, learning_rate=0.05)
    gbr.fit(X1, y1)
    dump(gbr, PATH_TO_PRICE_MODEL_GBR)

    # XGBoost
    '''
    X1_xgb = X1.values
    y1_xgb = data[['price']].values

    best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                                          gamma=0.5,
                                          learning_rate=0.1,
                                          max_depth=5,
                                          min_child_weight=3,
                                          n_estimators=300,
                                          reg_alpha=0,
                                          reg_lambda=0.6,
                                          subsample=0.8,
                                          seed=42)
    print("XGB start fitting: ")
    best_xgb_model.fit(X1_xgb, y1_xgb)
    dump(best_xgb_model, PATH_TO_PRICE_MODEL_X)
    '''
    # Cat Gradient
    cat = CatBoostRegressor(random_state=42, learning_rate=0.1, iterations=1000)
    train = Pool(X1, y1)
    cat.fit(train, verbose=1)
    dump(cat, PATH_TO_PRICE_MODEL_CAT)

def model():
    data = pd.read_csv(prepared_data + '/SPB.csv')
    Model(data)


if __name__ == '__main__':
    model()