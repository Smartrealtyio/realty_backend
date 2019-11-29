from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import settings_local as SETTINGS

DATA_OUTLIERS = SETTINGS.DATA + '/COORDINATES_OUTLIERS.csv'
MODEL_OUTLIERS = SETTINGS.MODEL + '/models.joblib'


def OutliersSearch(full_sq_from, full_sq_to, rooms, latitude_from, latitude_to,
               longitude_from, longitude_to, price_from=None, price_to=None, building_type_str=None, kitchen_sq=None,
               life_sq=None, renovation=None, has_elevator=None, floor_first=None, floor_last=None, time_to_metro=None):

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

    filter = (((ds.full_sq >= full_sq_from) & (ds.full_sq <= full_sq_to)) & (ds.rooms == rooms) &
              ((ds.latitude >= latitude_from)&(ds.latitude <= latitude_to)
               &(ds.longitude >= longitude_from)&(ds.longitude <= longitude_to)))
    ds = ds[filter]

    if time_to_metro != None:
        ds = ds[(ds.time_to_metro <= time_to_metro)]
    if rooms != None:
        ds = ds[ds.rooms == rooms]
    if building_type_str != None:
        ds = ds[ds.building_type_str == building_type_str]
    if kitchen_sq != None:
        ds = ds[(ds.kitchen_sq >= kitchen_sq - 5) & (ds.kitchen_sq <= kitchen_sq + 5)]
    if life_sq != None:
        ds = ds[(ds.life_sq >= life_sq - 5) & (ds.life_sq <= life_sq + 5)]
    if renovation != None:
        ds = ds[ds.renovation == renovation]
    if has_elevator != None:
        ds = ds[ds.has_elevator == has_elevator]
    if floor_first != None:
        ds = ds[ds.floor_first == 0]
    if floor_last != None:
        ds = ds[ds.floor_last == 0]
    if price_from != None:
        ds = ds[ds.price >= price_from]
    if price_to != None:
        ds = ds[ds.price <= price_to]

    return ds.to_dict('record')









OutliersSearch()
