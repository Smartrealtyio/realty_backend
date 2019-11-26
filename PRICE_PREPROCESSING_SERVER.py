import pandas as pd
import numpy as np
import settings_local as SETTINGS

raw_data = SETTINGS.PATH_TO_SINGLE_CSV_FILES
prepared_data = SETTINGS.DATA


# FINAL PARAMETERS ORDER:
# ['building_type_str', 'renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'full_sq', 'kitchen_sq',
#   'life_sq', 'is_apartment', 'time_to_metro', 'floor_last', 'floor_first']

def main_processing():
    ## DATA proccessing ##

    prices = pd.read_csv(raw_data + "prices.csv", names=[
        'id', 'price', 'changed_date', 'flat_id', 'created_at', 'updated_at'
    ],
                                       usecols=["price", "flat_id"],
                                       )
    flats = pd.read_csv(raw_data + "flats.csv",
                                      names=['id', 'full_sq', 'kitchen_sq', 'life_sq', 'floor', 'is_apartment',
                                             'building_id', 'created_at',
                                             'updated_at', 'offer_id', 'closed', 'rooms', 'image', 'resource_id'],
                                      usecols=["id", "full_sq",
                                               "kitchen_sq",
                                               "life_sq",
                                               "floor", "is_apartment",
                                               "building_id",
                                               "closed", 'rooms'
                                               ],
                                      true_values="t", false_values="f", header=0)
    buildings = pd.read_csv(raw_data + "buildings.csv", names=["id",
                                                                             "max_floor", 'building_type_str',
                                                                             "built_year", "flats_count",
                                                                             "address", "renovation", "has_elevator",
                                                                             'longitude', 'latitude', "district_id",
                                                                             'created_at', 'updated_at'],
                                          usecols=["id", "max_floor", 'building_type_str', "built_year", "flats_count",
                                                   # "address", #this comes from building_id
                                                   "renovation",
                                                   "has_elevator",
                                                   "district_id", 'longitude', 'latitude',  # nominative scale
                                                   ],
                                          true_values="t", false_values="f", header=0)
    districts = pd.read_csv(raw_data + "districts.csv",
                                          names=['id', 'name', 'population', 'city_id',
                                                 'created_at', 'updated_at', 'prefix'],
                                          usecols=["name", 'id'],
                                          true_values="t", false_values="f", header=0)
    time_to_metro = pd.read_csv(raw_data + "time_metro_buildings.csv",
                                              names=['id', 'building_id', 'metro_id', 'time_to_metro',
                                                     'transport_type', 'created_at', 'updated_at'],
                                              usecols=["building_id", "time_to_metro", "transport_type"], header=0)

    time_to_metro.sort_values('time_to_metro', ascending=True).drop_duplicates(subset='building_id',
                                                                               keep="first").sort_index()
    time_to_metro = time_to_metro[time_to_metro['transport_type'] == "ON_FOOT"]

    # choose the shortest path on foot
    ds = pd.merge(prices, flats, left_on="flat_id", right_on="id")

    new_ds = pd.merge(districts, buildings, left_on='id', right_on='district_id',
                                    suffixes=['_district', '_building'])

    ds = pd.merge(new_ds, ds, left_on='id_building', right_on='building_id')

    # ds = pd.merge(ds, buildings, left_on="building_id", right_on="id_")
    ds = pd.merge(ds, time_to_metro, left_on="id_building", right_on="building_id")

    # Building_type Labels encoding
    buildings_types = dict(PANEL=2, BLOCK=3, BRICK=4, MONOLIT=6,
                           UNKNOWN=0, MONOLIT_BRICK=5, WOOD=1)
    ds.building_type_str.replace(buildings_types, inplace=True)

    # ONLY CLOSED DEAL
    ds = ds.loc[ds['closed'] == True]

    ds = ds.drop(['closed'], axis=1)

    ds = ds.drop(['id', 'flats_count', 'id_district', 'name', 'building_id_x', 'building_id_y'], axis=1)
    ds.has_elevator = ds.has_elevator.astype(int)
    ds.renovation = ds.renovation.astype(int)
    ds.is_apartment = ds.is_apartment.astype(int)

    max_floor_list = ds['max_floor'].tolist()

    # НЕ первый\НЕ последний этаж
    ds['floor_last'] = np.where(ds['max_floor'] == ds['floor'], 1, 0)
    ds['floor_first'] = np.where(ds['floor'] == 1, 1, 0)
    ds = ds.drop_duplicates(subset='flat_id', keep="last")

    # REPLACE -1 WITH 0
    num = ds._get_numeric_data()

    num[num < 0] = 0
    ds['price_meter_sq'] = ds[['price', 'full_sq']].apply(
        lambda row: (row['price'] /
                     row['full_sq']), axis=1)

    ds = ds.drop(['max_floor', 'life_sq', 'rooms', 'built_year', "flat_id", 'floor', 'id_building', 'district_id',
                  'transport_type',
                  "building_type_str"], axis=1)

    ds.to_csv(prepared_data + 'COORDINATES_Pred_Price.csv', index=None, header=True)


if __name__ == '__main__':
    main_processing()
