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
from joblib import dump, load
import xgboost
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA_MOSCOW

PATH_TO_PRICE_MODEL_GBR = SETTINGS.MODEL_MOSCOW + '/PriceModel_GBR.joblib'
PATH_TO_PRICE_MODEL_RF = SETTINGS.MODEL_MOSCOW + '/PriceModel_RF.joblib'
PATH_TO_PRICE_MODEL_LGBM = SETTINGS.MODEL_MOSCOW + '/PriceModel_LGBM.joblib'



def Model(data: pd.DataFrame):
    from scipy import stats

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

    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
    gbr_model = GradientBoostingRegressor(n_estimators=350, max_depth=8, verbose=5, max_features=4, random_state=42,
                                          learning_rate=0.07).fit(X_train, y_train)
    gbr_preds = gbr_model.predict(X_test)
    print('The R2_score of the Gradient boosting is', r2_score(y_test, gbr_preds), flush=True)
    print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)

    print('Train on full dataset: ', flush=True)
    gbr_model.fit(X1, y1)
    print('Save model: ')
    dump(gbr_model, PATH_TO_PRICE_MODEL_GBR)

    RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=3, n_jobs=-1).fit(X_train, y_train)

    rf_predicts = RF.predict(X_test)

    print('The accuracy of the Random Forest is', r2_score(y_test, rf_predicts), flush=True)
    print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)

    print('Train on full dataset: ', flush=True)
    RF.fit(X1, y1)
    dump(RF, PATH_TO_PRICE_MODEL_RF)

    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.1,
                               n_estimators=1250, max_depth=6, min_child_samples=1, verbose=3).fit(X_train, y_train)
    lgbm_preds = lgbm_model.predict(X_test)
    print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
    print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)

    print('Train on full dataset: ', flush=True)
    lgbm_model.fit(X1, y1)

    print('Save model: ')
    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM)


def model():
    data = pd.read_csv(prepared_data + '/MOSCOW.csv')
    Model(data)


if __name__ == '__main__':
    model()


