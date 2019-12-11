import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, scorer, mean_squared_error
import logging
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostRegressor, Pool
# import RealtyTime.config as cf
from joblib import dump, load
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA
PATH_TO_TIME_MODEL = SETTINGS.MODEL



def model():
    data = pd.read_csv(prepared_data+'/COORDINATES_MAIN.csv')
    X_term = data[
        ['renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'full_sq', 'kitchen_sq',
         'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']]
    y_term = data[['term']]

    # Cat Gradient
    cat = CatBoostRegressor(random_state=42, iterations=2000, max_depth=5, learning_rate=0.1)
    train = Pool(X_term,y_term)
    cat.fit(train, verbose=5)
    dump(cat, PATH_TO_TIME_MODEL+'/CAT_TIME_MODEL.joblib')

if __name__ == '__main__':
    model()
