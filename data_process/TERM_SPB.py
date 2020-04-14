import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
# import Realty.config as cf
from joblib import dump, load
import xgboost
import os
import settings_local as SETTINGS

prepared_data_secondary = SETTINGS.DATA_SPB+ '/SPB_VTOR.csv'
prepared_data_new = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'

PATH_TO_TERM_MODEL_GBR_NEW_D = SETTINGS.MODEL_SPB + '/TermModel_SPB_NEW_GBR_D.joblib'
PATH_TO_TERM_MODEL_GBR_Secondary_D = SETTINGS.MODEL_SPB + '/TermModel_SPB_Secondary_GBR_D.joblib'





# Function to calculate term
def term_gbr(data: pd.DataFrame, type: str):
    # Remove price and term outliers (out of 3 sigmas)
    data1 = data[(np.abs(stats.zscore(data.price)) < 3)]
    data2 = data[(np.abs(stats.zscore(data.term)) < 3)]

    data = pd.merge(data1, data2, on=list(data.columns), how='left')

    # Fill NaN if it appears after merging
    data[['term']] = data[['term']].fillna(data[['term']].mean())

    # data.full_sq = np.log1p(data.full_sq)
    data.price_meter_sq = np.log1p(data.price_meter_sq)
    # data.mm_announce = np.log1p(data.mm_announce)
    data.was_opened = np.log1p(data.was_opened)
    data.term = np.log1p(data.term)
    data.profit = np.log1p(data.profit)

    data = data[['term', 'price_meter_sq', 'profit', 'price', 'full_sq', 'kitchen_sq', 'life_sq', 'is_apartment',
      'renovation', 'has_elevator',
      'time_to_metro', 'floor_first', 'floor_last',
      'is_rented', 'rent_quarter',
      'rent_year', 'to_center', 'mm_announce__1',
      'mm_announce__2', 'mm_announce__3', 'mm_announce__4',
      'mm_announce__5', 'mm_announce__6', 'mm_announce__7', 'mm_announce__8', 'mm_announce__9',
      'mm_announce__10', 'mm_announce__11', 'mm_announce__12', 'rooms__0',
      'rooms__1', 'rooms__2', 'rooms__3', 'rooms__4', 'rooms__5', 'rooms__6', 'yyyy_announce__18',
      'yyyy_announce__19', 'yyyy_announce__20',
      'cluster__0', 'cluster__1',
      'cluster__2', 'cluster__3', 'cluster__4', 'cluster__5', 'cluster__6', 'cluster__7', 'cluster__8',
      'cluster__9', 'cluster__10', 'cluster__11', 'cluster__12', 'cluster__13', 'cluster__14', 'cluster__15',
      'cluster__16',
      'cluster__17', 'cluster__18', 'cluster__19',
      'cluster__20', 'cluster__21', 'cluster__22', 'cluster__23', 'cluster__24',
      'cluster__25', 'cluster__26', 'cluster__27', 'cluster__28', 'cluster__29', 'cluster__30',
      'cluster__31', 'cluster__32',
      'cluster__33', 'cluster__34', 'cluster__35', 'cluster__36', 'cluster__37', 'cluster__38',
      'cluster__39', 'cluster__40',
      'cluster__41', 'cluster__42', 'cluster__43', 'cluster__44', 'cluster__45', 'cluster__46',
      'cluster__47', 'cluster__48', 'cluster__49', 'cluster__50', 'cluster__51', 'cluster__52',
      'cluster__53', 'cluster__54', 'cluster__55',
      'cluster__56', 'cluster__57', 'cluster__58', 'cluster__59']]

    X = data.drop(['term'], axis=1)
    y = data[['term']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    reg = GradientBoostingRegressor(n_estimators=350, max_depth=3, verbose=1, random_state=42,
                                    learning_rate=0.07)
    reg.fit(X_train, y_train)
    gbr_preds = reg.predict(X_test)
    print('The R2_score GBR Spb term ', r2_score(y_test, gbr_preds), flush=True)
    print('RMSE GBR Spb term ', mean_squared_error(y_test, gbr_preds), flush=True)


    print('Train on full dataset GBR Spb term ', flush=True)
    reg.fit(X, y)

    if type == "New_flats":
        dump(reg, PATH_TO_TERM_MODEL_GBR_NEW_D)
    elif type == "Secondary":
        dump(reg, PATH_TO_TERM_MODEL_GBR_Secondary_D)

    cdf = pd.DataFrame(np.transpose(reg.feature_importances_), X.columns, columns=['Coefficients']).sort_values(
        by=['Coefficients'], ascending=False)
    # print(cdf)

    return reg



def LEARNER_D():
    data_secondary = pd.read_csv(prepared_data_secondary)
    data_new_flats = pd.read_csv(prepared_data_new)

    term_gbr(data_secondary, type='New_flats')
    term_gbr(data_new_flats, type='Secondary')

if __name__ == '__main__':
    LEARNER_D()


