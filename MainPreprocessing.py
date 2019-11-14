import pandas as pd
import numpy as np
from sklearn import preprocessing
import settings_local as SETTINGS
import backports.datetime_fromisoformat as bck



# FINAL PARAMETERS ORDER:
# ['building_type_str', 'renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'term', 'full_sq', 'kitchen_sq',
# 'life_sq', 'is_apartment', 'time_to_metro', 'floor_last', 'floor_first']

def main_preprocessing():

    # Data preprocessing #
    prices: pd.DataFrame = pd.read_csv(SETTINGS.PATH_TO_SINGLE_CSV_FILES + "prices.csv", names=[
        'id', 'price', 'changed_date', 'flat_id', 'created_at', 'updated_at'
    ], usecols=["price", "flat_id", 'changed_date', 'updated_at'])
    print(prices.shape)

    # Calculating selling term. TIME UNIT: DAYS
    prices['term'] = prices[['updated_at', 'changed_date']].apply(
        lambda row: (bck.date_fromisoformat(row['updated_at'][:-9])
                     - bck.date_fromisoformat(row['changed_date'][:-9])).days, axis=1)
    flats: pd.DataFrame = pd.read_csv(SETTINGS.PATH_TO_SINGLE_CSV_FILES + "flats.csv",
                                      names=['id', 'full_sq', 'kitchen_sq', 'life_sq', 'floor', 'is_apartment',
                                             'building_id', 'created_at',
                                             'updated_at', 'offer_id', 'closed', 'rooms'],
                                      usecols=["id", "full_sq",
                                               "kitchen_sq",
                                               "life_sq",
                                               "floor", "is_apartment",
                                               "building_id",
                                               "closed", 'rooms'
                                               ],
                                      true_values="t", false_values="f", header=0)
    buildings: pd.DataFrame = pd.read_csv(SETTINGS.PATH_TO_SINGLE_CSV_FILES + "buildings.csv",
                                          names=["id", "max_floor", 'building_type_str', "built_year", "flats_count",
                                                                                              "address", "renovation",
                                                                                              "has_elevator",
                                                                                              'longitude', 'latitude',
                                                                                              "district_id",
                                                                                              'created_at',
                                                                                              'updated_at'],
                                          usecols=["id", "max_floor", 'building_type_str', "built_year", "flats_count",
                                                   "renovation",
                                                   "has_elevator",
                                                   "district_id", 'longitude', 'latitude',  # nominative scale
                                                   ],
                                          true_values="t", false_values="f", header=0)
    districts: pd.DataFrame = pd.read_csv(SETTINGS.PATH_TO_SINGLE_CSV_FILES + "districts.csv",
                                          names=['id', 'name', 'population', 'city_id',
                                                 'created_at', 'updated_at', 'prefix'],
                                          usecols=["name", 'id'],
                                          true_values="t", false_values="f", header=0)
    time_to_metro: pd.DataFrame = pd.read_csv(SETTINGS.PATH_TO_SINGLE_CSV_FILES + "time_metro_buildings.csv",
                                              names=['id', 'building_id', 'metro_id', 'time_to_metro',
                                                     'transport_type', 'created_at', 'updated_at'],
                                              usecols=["building_id", "time_to_metro", "transport_type"], header=0)

    time_to_metro = time_to_metro.drop_duplicates(subset='building_id', keep="first")
    time_to_metro = time_to_metro[time_to_metro['transport_type'] == "ON_FOOT"]

    # choose the shortest path on foot
    ds: pd.DataFrame = pd.merge(prices, flats, left_on="flat_id", right_on="id")

    print('merge#1: ', ds.shape)
    new_ds: pd.DataFrame = pd.merge(districts, buildings, left_on='id', right_on='district_id',
                                    suffixes=['_district', '_building'])
    print('HEADERS NAME: ', list(new_ds.columns))

    ds = pd.merge(new_ds, ds, left_on='id_building', right_on='building_id')
    print('HEADERS NAME: ', list(ds.columns))
    # ds = pd.merge(ds, buildings, left_on="building_id", right_on="id_")
    ds = pd.merge(ds, time_to_metro, left_on="id_building", right_on="building_id")
    # ds = pd.get_dummies(ds, columns=["transport_type"])

    # ds = pd.get_dummies(ds, columns=["max_floor"])
    # ONLY CLOSED DEAL
    ds = ds.loc[ds['closed'] == True]
    print(ds.closed.value_counts())
    ds = ds.drop(['closed'], axis=1)
    print('HEADERS NAME: ', list(ds.columns))
    ds = ds.drop(['id', 'built_year', 'flats_count', 'id_district', 'name', 'building_id_x', 'building_id_y'], axis=1)
    ds.has_elevator = ds.has_elevator.astype(int)
    ds.renovation = ds.renovation.astype(int)
    ds.is_apartment = ds.is_apartment.astype(int)
    print('HEADERS NAME: ', list(ds.columns))
    max_floor_list = ds['max_floor'].tolist()

    ds['floor_last'] = np.where(ds['max_floor']==ds['floor'], 1, 0)
    ds['floor_first'] = np.where(ds['floor']==1, 1, 0)
    ds = ds.drop_duplicates(subset='flat_id', keep="last")
    print(ds.building_type_str.value_counts())
    # le = preprocessing.LabelEncoder()
    # PANEL = 4; BLOCK = 0; BRICK = 1; MONOLIT = 2; UNKNOWN = 5; MONOLIT_BRICK = 3; WOOD = 6
    buildings_types = dict(PANEL=4, BLOCK=0, BRICK=1, MONOLIT=2,
                           UNKNOWN=5, MONOLIT_BRICK=3, WOOD=6)
    ds.building_type_str.replace(buildings_types, inplace=True)
    #uniques = ds.building_type_str.unique()
    #print(uniques)
    #keys= buildings_types.keys()
    #if ds[ds['building_type_str']].isin([keys]):
    #    ds[ds['building_type_str']] = buildings_types.get(ds['building_type_str'])
    #y = ds.sentiment
    #ds['building_type_str'] = le.fit_transform(ds['building_type_str'])

    ds = ds.drop(['max_floor', "flat_id", 'floor', 'updated_at', 'changed_date', 'id_building', 'district_id', 'transport_type'], axis=1)
    print(ds.shape)
    print('HEADERS NAME FINALY: ', list(ds.columns))

    print('Saving to new csv')
    print(ds.building_type_str[:3])
    ds.to_csv(SETTINGS.DATA+'/COORDINATES_MEAN_PRICE.csv', index=None, header=True)


if __name__ == '__main__':
    main_preprocessing()
