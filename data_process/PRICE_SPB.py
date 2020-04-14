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
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import settings_local as SETTINGS

prepared_data_VTOR = SETTINGS.DATA_SPB + '/SPB_VTOR.csv'
prepared_data_NEW = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'


# Path to prices models with dummies features
PATH_TO_PRICE_MODEL_GBR_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_GBR_D.joblib'
PATH_TO_PRICE_MODEL_RF_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_RF_D.joblib'
PATH_TO_PRICE_MODEL_LGBM_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_LGBM_D.joblib'


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
    dump(gbr_model, PATH_TO_PRICE_MODEL_GBR_D)

    # RANDOM FOREST REGRESSOR
    RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1)

    RF.fit(X_train, y_train)

    rf_predicts = RF.predict(X_test)

    print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
    print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)

    print('Train on full dataset RF secondary Spb: ', flush=True)
    RF.fit(X, y)

    print('Save model RF secondary Spb: ', flush=True)
    dump(RF, PATH_TO_PRICE_MODEL_RF_D)


    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.1,
                               n_estimators=1250, max_depth=5, min_child_samples=1, verbose=0)

    lgbm_model.fit(X_train, y_train)
    lgbm_preds = lgbm_model.predict(X_test)
    print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
    print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)

    print('Train on full dataset LGBM secondary Spb: ', flush=True)
    lgbm_model.fit(X, y)

    print('Save model LGBM secondary Spb: ', flush=True)
    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM_D)

# def Price_Secondary_D(data: pd.DataFrame):
#
#     # Remove price and term outliers (out of 3 sigmas)
#     data1 = data[(np.abs(stats.zscore(data.price)) < 3)]
#     data2 = data[(np.abs(stats.zscore(data.term)) < 3)]
#
#
#     data = pd.merge(data1, data2, on=list(data.columns), how='left')
#
#     # Fill NaN if it appears after merging
#     data[['term']] = data[['term']].fillna(data[['term']].mean())
#
#     df = data[['price', 'full_sq', 'kitchen_sq', 'life_sq', 'is_apartment',
#              'renovation', 'has_elevator',
#              'time_to_metro', 'floor_first', 'floor_last',
#              'is_rented', 'rent_quarter',
#              'rent_year', 'to_center', 'was_opened', 'mm_announce__1',
#              'mm_announce__2', 'mm_announce__3', 'mm_announce__4',
#              'mm_announce__5', 'mm_announce__6', 'mm_announce__7', 'mm_announce__8', 'mm_announce__9',
#              'mm_announce__10', 'mm_announce__11', 'mm_announce__12', 'rooms__0',
#              'rooms__1', 'rooms__2', 'rooms__3', 'rooms__4', 'rooms__5', 'rooms__6', 'yyyy_announce__18',
#              'yyyy_announce__19', 'yyyy_announce__20',
#              'cluster__0', 'cluster__1',
#              'cluster__2', 'cluster__3', 'cluster__4', 'cluster__5', 'cluster__6', 'cluster__7', 'cluster__8',
#              'cluster__9', 'cluster__10',
#              'cluster__11', 'cluster__12', 'cluster__13', 'cluster__14', 'cluster__15', 'cluster__16',
#              'cluster__17', 'cluster__18', 'cluster__19',
#              'cluster__20', 'cluster__21', 'cluster__22', 'cluster__23', 'cluster__24',
#              'cluster__25', 'cluster__26', 'cluster__27', 'cluster__28', 'cluster__29', 'cluster__30',
#              'cluster__31', 'cluster__32',
#              'cluster__33', 'cluster__34', 'cluster__35', 'cluster__36', 'cluster__37', 'cluster__38',
#              'cluster__39', 'cluster__40',
#              'cluster__41', 'cluster__42', 'cluster__43', 'cluster__44', 'cluster__45', 'cluster__46',
#              'cluster__47', 'cluster__48', 'cluster__49', 'cluster__50', 'cluster__51', 'cluster__52',
#              'cluster__53', 'cluster__54', 'cluster__55',
#              'cluster__56', 'cluster__57', 'cluster__58', 'cluster__59']]
#
#     # Save leaved columns to variable
#     columns = list(df.columns)
#
#     # Log transformation
#     df["full_sq"] = np.log1p(df["full_sq"])
#     df["life_sq"] = np.log1p(df["life_sq"])
#     df["kitchen_sq"] = np.log1p(df["kitchen_sq"])
#     df["price"] = np.log1p(df["price"])
#
#     # Create features - predictors
#     X = df.drop(['price'], axis=1)
#
#     # Target feature
#     y = df[['price']].values.ravel()
#
#     # Split for train and test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
#     # Define Gradient Boosting Machine model
#     gbr_model = GradientBoostingRegressor(n_estimators=350, max_depth=4, verbose=1, max_features='sqrt',
#                                           random_state=42,
#                                           learning_rate=0.07)
#     # Train GBR on train dataset
#     gbr_model.fit(X_train, y_train)
#     gbr_preds = gbr_model.predict(X_test)
#     print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds), flush=True)
#     print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)
#
#     # Train GBR on full dataset
#     print('Train on full dataset GBR secondary Spb: ', flush=True)
#     gbr_model.fit(X, y)
#
#     print('Save model GBR secondary Spb: ', flush=True)
#     dump(gbr_model, PATH_TO_PRICE_MODEL_GBR_VTOR_D)
#
#     RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1)
#
#     RF.fit(X_train, y_train)
#
#     rf_predicts = RF.predict(X_test)
#
#     print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
#     print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)
#
#     print('Train on full dataset RF secondary Spb: ', flush=True)
#     RF.fit(X, y)
#
#     print('Save model RF secondary Spb: ', flush=True)
#     dump(RF, PATH_TO_PRICE_MODEL_RF_VTOR_D)
#
#     # LGBM model
#     lgbm_model = LGBMRegressor(objective='regression',
#                                learning_rate=0.1,
#                                n_estimators=1250, max_depth=4, min_child_samples=1, verbose=0)
#
#     # lgbm_model.fit(X_train, y_train)
#     # lgbm_preds = lgbm_model.predict(X_test)
#     # print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
#     # print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)
#
#     print('Train on full dataset LGBM secondary Spb: ', flush=True)
#     lgbm_model.fit(X, y)
#
#     print('Save model LGBM secondary Spb: ', flush=True)
#     dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM_VTOR_D)

# def Pirce_NewFlats(data: pd.DataFrame):
#
#     # Remove price and term outliers (out of 3 sigmas)
#     data1 = data[(np.abs(stats.zscore(data.price)) < 3)]
#     data2 = data[(np.abs(stats.zscore(data.term)) < 3)]
#
#
#     data = pd.merge(data1, data2, on=list(data.columns), how='left')
#
#     # Fill NaN if it appears after merging
#     data[['term']] = data[['term']].fillna(data[['term']].mean())
#
#     # Log Transformation
#     data["longitude"] = np.log1p(data["longitude"])
#     data["latitude"] = np.log1p(data["latitude"])
#     data["full_sq"] = np.log1p(data["full_sq"])
#     data["life_sq"] = np.log1p(data["life_sq"])
#     data["kitchen_sq"] = np.log1p(data["kitchen_sq"])
#     data["price"] = np.log1p(data["price"])
#     X = data[['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq', 'time_to_metro', 'floor_last', 'floor_first', 'clusters', 'is_rented', 'rent_quarter', 'rent_year']]
#
#     y = data[['price']].values.ravel()
#     print(X.shape, y.shape, flush=True)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#     gbr_model = GradientBoostingRegressor(n_estimators=350, max_depth=8, verbose=1, max_features=4, random_state=42,
#                                           learning_rate=0.07)
#     # gbr_model.fit(X_train, y_train)
#     # gbr_preds = gbr_model.predict(X_test)
#     # print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds), flush=True)
#     # print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)
#
#     print('Train on full dataset GBR new Spb: ', flush=True)
#     gbr_model.fit(X, y)
#
#     print('Save model GBR new Spb: ', flush=True)
#     dump(gbr_model, PATH_TO_PRICE_MODEL_GBR_NEW)
#
#     RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1).fit(X_train, y_train)
#
#     # rf_predicts = RF.predict(X_test)
#     #
#     # print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
#     # print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)
#
#     print('Train on full dataset RF new Spb: ', flush=True)
#     RF.fit(X, y)
#
#     print('Save model RF new Spb: ', flush=True)
#     dump(RF, PATH_TO_PRICE_MODEL_RF_NEW)
#
#     # LGBM model
#     lgbm_model = LGBMRegressor(objective='regression',
#                                learning_rate=0.1,
#                                n_estimators=1250, max_depth=6, min_child_samples=1, verbose=0)
#     # lgbm_model.fit(X_train, y_train)
#     # lgbm_preds = lgbm_model.predict(X_test)
#     # print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
#     # print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)
#
#     print('Train on full dataset LGBM new Spb: ', flush=True)
#     lgbm_model.fit(X, y)
#
#     print('Save model LGBM new Spb: ', flush=True)
#     dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM_NEW)
#
#
# def Pirce_NewFlats_D(data: pd.DataFrame):
#     # Remove price and term outliers (out of 3 sigmas)
#     data1 = data[(np.abs(stats.zscore(data.price)) < 3)]
#     data2 = data[(np.abs(stats.zscore(data.term)) < 3)]
#
#     data = pd.merge(data1, data2, on=list(data.columns), how='left')
#
#     # Fill NaN if it appears after merging
#     data[['term']] = data[['term']].fillna(data[['term']].mean())
#
#     df = data[['price', 'full_sq', 'kitchen_sq', 'life_sq', 'is_apartment',
#              'renovation', 'has_elevator',
#              'time_to_metro', 'floor_first', 'floor_last',
#              'is_rented', 'rent_quarter',
#              'rent_year', 'to_center', 'was_opened', 'mm_announce__1',
#              'mm_announce__2', 'mm_announce__3', 'mm_announce__4',
#              'mm_announce__5', 'mm_announce__6', 'mm_announce__7', 'mm_announce__8', 'mm_announce__9',
#              'mm_announce__10', 'mm_announce__11', 'mm_announce__12', 'rooms__0',
#              'rooms__1', 'rooms__2', 'rooms__3', 'rooms__4', 'rooms__5', 'rooms__6', 'yyyy_announce__18',
#              'yyyy_announce__19', 'yyyy_announce__20',
#              'cluster__0', 'cluster__1',
#              'cluster__2', 'cluster__3', 'cluster__4', 'cluster__5', 'cluster__6', 'cluster__7', 'cluster__8',
#              'cluster__9', 'cluster__10',
#              'cluster__11', 'cluster__12', 'cluster__13', 'cluster__14', 'cluster__15', 'cluster__16',
#              'cluster__17', 'cluster__18', 'cluster__19',
#              'cluster__20', 'cluster__21', 'cluster__22', 'cluster__23', 'cluster__24',
#              'cluster__25', 'cluster__26', 'cluster__27', 'cluster__28', 'cluster__29', 'cluster__30',
#              'cluster__31', 'cluster__32',
#              'cluster__33', 'cluster__34', 'cluster__35', 'cluster__36', 'cluster__37', 'cluster__38',
#              'cluster__39', 'cluster__40',
#              'cluster__41', 'cluster__42', 'cluster__43', 'cluster__44', 'cluster__45', 'cluster__46',
#              'cluster__47', 'cluster__48', 'cluster__49', 'cluster__50', 'cluster__51', 'cluster__52',
#              'cluster__53', 'cluster__54', 'cluster__55',
#              'cluster__56', 'cluster__57', 'cluster__58', 'cluster__59']]
#
#     # Save leaved columns to variable
#     columns = list(df.columns)
#
#     # Log transformation
#     df["full_sq"] = np.log1p(df["full_sq"])
#     df["life_sq"] = np.log1p(df["life_sq"])
#     df["kitchen_sq"] = np.log1p(df["kitchen_sq"])
#     df["price"] = np.log1p(df["price"])
#
#     # Create features - predictors
#     X = df.drop(['price'], axis=1)
#
#     # Target feature
#     y = df[['price']].values.ravel()
#
#     # Split for train and test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
#     # Define Gradient Boosting Machine model
#     gbr_model = GradientBoostingRegressor(n_estimators=450, max_depth=4, verbose=1, max_features='sqrt',
#                                           random_state=42,
#                                           learning_rate=0.07)
#     # Train GBR on train dataset
#     gbr_model.fit(X_train, y_train)
#     gbr_preds = gbr_model.predict(X_test)
#     print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds), flush=True)
#     print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)
#
#     # Train GBR on full dataset
#     print('Train on full dataset GBR new Spb: ', flush=True)
#     gbr_model.fit(X, y)
#
#     print('Save model GBR new Spb: ', flush=True)
#     dump(gbr_model, PATH_TO_PRICE_MODEL_GBR_NEW_D)
#
#     RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1).fit(X_train, y_train)
#
#     rf_predicts = RF.predict(X_test)
#
#     print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
#     print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)
#
#     print('Train on full dataset RF new Spb: ', flush=True)
#     RF.fit(X, y)
#
#     print('Save model RF new Spb: ', flush=True)
#     dump(RF, PATH_TO_PRICE_MODEL_RF_NEW_D)
#
#     # LGBM model
#     lgbm_model = LGBMRegressor(objective='regression',
#                                learning_rate=0.1,
#                                n_estimators=1250, max_depth=4, min_child_samples=1, verbose=0)
#     lgbm_model.fit(X_train, y_train)
#     lgbm_preds = lgbm_model.predict(X_test)
#     print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
#     print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)
#
#     print('Train on full dataset LGBM new Spb: ', flush=True)
#     lgbm_model.fit(X, y)
#
#     print('Save model LGBM new Spb: ', flush=True)
#     dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM_NEW_D)


# def learner_OLD():
#     data = pd.read_csv(prepared_data_VTOR)
#     Price_Secondary(data)
#     new_data = pd.read_csv(prepared_data_NEW)
#     Pirce_NewFlats(new_data)

def learner_D():

    # Load Data Flats New and Data Secondary flats
    df1 = pd.read_csv(prepared_data_VTOR)
    df2 = pd.read_csv(prepared_data_NEW)

    # Concatenate two types of flats
    all_data = pd.concat([df1, df2], ignore_index=True)

    # Train models
    Price_Main(all_data)


if __name__ == '__main__':
    learner_D()
