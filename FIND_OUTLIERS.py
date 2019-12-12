from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from joblib import dump, load
import settings_local as SETTINGS
from sklearn.ensemble import GradientBoostingRegressor

DATA_OUTLIERS = SETTINGS.DATA + '/COORDINATES_OUTLIERS.csv'
PATH_TO_PRICE_MODEL = SETTINGS.MODEL + '/PriceModel.joblib'
MODEL_OUTLIERS = SETTINGS.MODEL + '/models.joblib'



def OutliersSearch():

    data = pd.read_csv(DATA_OUTLIERS)
    # data = data[['price_meter_sq', 'full_sq']]
    # data = data[data.price_meter_sq < data.price_meter_sq.quantile(0.2)]
    print("Data price_meter_sq < price_meter_sq.quantile(0.2): ", data.shape, flush=True)

    X1 = data[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                 'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']]
    data["price"] = np.log1p(data["price"])
    y1 = data[['price']].values.ravel()
    print(X1.shape, y1.shape)

    clf = GradientBoostingRegressor(n_estimators=350, max_depth=12, verbose=10, max_features=5)
    clf.fit(X1, y1)
    dump(clf, PATH_TO_PRICE_MODEL)


# if __name__ == '__main__':
#     OutliersSearch()


