from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from joblib import dump, load
import settings_local as SETTINGS

DATA_OUTLIERS = SETTINGS.DATA + '/COORDINATES_OUTLIERS.csv'
MODEL_OUTLIERS = SETTINGS.MODEL + '/models.joblib'


def OutliersSearch():

    data = pd.read_csv(DATA_OUTLIERS)
    # data = data[['price_meter_sq', 'full_sq']]
    # data = data[data.price_meter_sq < data.price_meter_sq.quantile(0.2)]
    print("Data price_meter_sq < price_meter_sq.quantile(0.2): ", data.shape, flush=True)



    clf = IsolationForest(contimation=0.2)
    clf.fit(data)
    # save model
    dump(clf, MODEL_OUTLIERS)


if __name__ == '__main__':
    OutliersSearch()


