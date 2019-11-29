from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

DATA_OUTLIERS = 'C:/Storage/DDG/DATA/PREPARED/COORDINATES_OUTLIERS.csv'
MODEL_OUTLIERS = 'C:/Storage/DDG/MedianPrices/Server/outliers_models/models.joblib'


def OutliersSearch(full_sq: float, rooms: int, latitude_from: float, latitude_to: float,
               longitude_from: float, longitude_to: float):

    data = pd.read_csv(DATA_OUTLIERS)
    data = data[['price_meter_sq', 'full_sq']]
    data = data[data.price_meter_sq < data.price_meter_sq.quantile(0.2)]
    print("Data price_meter_sq < price_meter_sq.quantile(0.2): ", data)



    clf = IsolationForest(max_samples=10, random_state=42, contamination=.01, max_features=2)
    clf.fit(data)
    # save model
    dump(clf, MODEL_OUTLIERS)
    model = load(MODEL_OUTLIERS)
    # outliers = model.predict(data)
    outliers_it = data[model.predict(data) == -1]
    print('Outliers: ', outliers_it.shape[0])
    outliers_it['flat_id'] = outliers_it.index
    new_data = pd.read_csv(DATA_OUTLIERS)
    print(new_data.shape)
    new_data['flat_id'] = new_data.index
    ds: pd.DataFrame = pd.merge(new_data, outliers_it, left_on="flat_id", right_on="flat_id", suffixes=['', 'double'])
    ds = ds.drop(['flat_id', 'full_sqdouble', 'price_meter_sqdouble'], axis=1)

    filter = (((ds.full_sq >= full_sq-5) & (ds.full_sq <= full_sq+5)) & (ds.rooms == rooms) &
              ((ds.latitude >= latitude_from)&(ds.latitude <= latitude_to)
               &(ds.longitude >= longitude_from)&(ds.longitude <= longitude_to)))
    ds = ds[filter]


    return "Outliers: \n\n {0}".format(ds.to_dict('index'))









OutliersSearch()
