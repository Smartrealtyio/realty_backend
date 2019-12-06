import pandas as pd
import numpy as np
from sklearn import preprocessing
import backports.datetime_fromisoformat as bck
from joblib import dump
import settings_local as SETTINGS
from sklearn.cluster import KMeans

# FINAL PARAMETERS ORDER:
# ['building_type_str', 'renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'term', 'full_sq', 'kitchen_sq',
# 'life_sq', 'is_apartment', 'time_to_metro', 'floor_last', 'floor_first']
raw_data = SETTINGS.PATH_TO_SINGLE_CSV_FILES
prepared_data = SETTINGS.DATA
PATH_TO_TIME_MODEL = SETTINGS.MODEL

def main_preprocessing():

        # DATA preprocessing #
        prices = pd.read_csv(raw_data+ "prices.csv", names=[
            'id', 'price', 'changed_date', 'flat_id', 'created_at', 'updated_at'
        ], usecols=["price", "flat_id", 'changed_date', 'updated_at'])
        print(prices.shape)
        prices = prices.iloc[:-3]

        # Calculating selling term. TIME UNIT: DAYS
        prices['term'] = prices[['updated_at', 'changed_date']].apply(
            lambda row: (bck.date_fromisoformat(row['updated_at'][:-9])
                         - bck.date_fromisoformat(row['changed_date'][:-9])).days, axis=1)
        flats = pd.read_csv(raw_data+ "flats.csv",
                                          names=['id', 'full_sq', 'kitchen_sq', 'life_sq', 'floor', 'is_apartment',
                                                 'building_id', 'created_at',
                                                 'updated_at', 'offer_id', 'closed', 'rooms', 'image', 'resource_id'],
                                          usecols=["id", "full_sq",
                                                   "kitchen_sq",
                                                   "life_sq",
                                                   "floor", "is_apartment",
                                                   "building_id",
                                                   "closed", 'rooms', 'offer_id', 'resource_id'
                                                   ],
                                          true_values="t", false_values="f", header=0)
        buildings = pd.read_csv(raw_data+ "buildings.csv",
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
        districts = pd.read_csv(raw_data+ "districts.csv",
                                              names=['id', 'name', 'population', 'city_id',
                                                     'created_at', 'updated_at', 'prefix'],
                                              usecols=["name", 'id'],
                                              true_values="t", false_values="f", header=0)
        time_to_metro = pd.read_csv(raw_data+ "time_metro_buildings.csv",
                                                  names=['id', 'building_id', 'metro_id', 'time_to_metro',
                                                         'transport_type', 'created_at', 'updated_at'],
                                                  usecols=["building_id", "time_to_metro", "transport_type"], header=0)

        time_to_metro.sort_values('time_to_metro', ascending=True).drop_duplicates(subset='building_id', keep="first").sort_index()
        time_to_metro = time_to_metro[time_to_metro['transport_type'] == "ON_FOOT"]

        # choose the shortest path on foot
        ds = pd.merge(prices, flats, left_on="flat_id", right_on="id")

        print('HEADERS NAME: ', list(ds.columns))
        print('merge#1: ', ds.shape)
        new_ds = pd.merge(districts, buildings, left_on='id', right_on='district_id',
                                        suffixes=['_district', '_building'])
        print('HEADERS NAME: ', list(new_ds.columns))

        ds = pd.merge(new_ds, ds, left_on='id_building', right_on='building_id')
        print('HEADERS NAME: ', list(ds.columns))
        # ds = pd.merge(ds, buildings, left_on="building_id", right_on="id_")
        ds = pd.merge(ds, time_to_metro, left_on="id_building", right_on="building_id")
        # ds = pd.get_dummies(ds, columns=["transport_type"])

        # ds = pd.get_dummies(ds, columns=["max_floor"])

        ds = ds.drop(['id', 'built_year', 'flats_count', 'id_district', 'name', 'building_id_x', 'building_id_y'], axis=1)
        ds.has_elevator = ds.has_elevator.astype(int)
        ds.renovation = ds.renovation.astype(int)
        ds.is_apartment = ds.is_apartment.astype(int)
        print('HEADERS NAME: ', list(ds.columns))
        max_floor_list = ds['max_floor'].tolist()

        ds['floor_last'] = np.where(ds['max_floor']==ds['floor'], 1, 0)
        ds['floor_first'] = np.where(ds['floor']==1, 1, 0)
        ds = ds.drop_duplicates(subset='flat_id', keep="last")

        # Building_type Labels encoding
        buildings_types = dict(PANEL=2, BLOCK=3, BRICK=4, MONOLIT=6,
                               UNKNOWN=0, MONOLIT_BRICK=5, WOOD=1)
        ds.building_type_str.replace(buildings_types, inplace=True)

        # ONLY CLOSED DEAL
        ds = ds.loc[ds['closed'] == True]
        print(ds.closed.value_counts())
        ds = ds.drop(['closed'], axis=1)
        print('HEADERS NAME: ', list(ds.columns))

        # REPLACE -1 WITH 0
        num = ds._get_numeric_data()

        num[num < 0] = 0
        import math as m
        ds['X'] = ds[['latitude', 'longitude']].apply(
            lambda row: (m.cos(row['latitude']) *
                         m.cos(row['longitude'])), axis=1)
        ds['Y'] = ds[['latitude', 'longitude']].apply(
            lambda row: (m.cos(row['latitude']) *
                         m.sin(row['longitude'])), axis=1)
        ds['price_meter_sq'] = ds[['price', 'full_sq']].apply(
            lambda row: (row['price'] /
                         row['full_sq']), axis=1)
        ds = ds.drop(['max_floor', "flat_id", 'floor','building_type_str', 'rooms', 'life_sq', 'updated_at', 'changed_date',
                      'id_building', 'district_id', 'transport_type'], axis=1)

        print('HEADERS NAME FINALY: ', list(ds.columns))

        print("All data: ", ds.shape)
        ds = ds[ds.resource_id == 0]
        print('Just Yandex: ', ds.shape)


        ''''''
        def remove_outlier(df_in, col_name):
            q1 = df_in[col_name].quantile(0.10)
            q3 = df_in[col_name].quantile(0.90)
            #iqr = q3 - q1  # Interquartile range
            #fence_low = q1 - 1.5 * iqr
            #fence_high = q3 + 1.5 * iqr
            df_out = df_in.loc[(df_in[col_name] > q1) & (df_in[col_name] < q3)]
            return df_out

        from scipy import stats
        df = ds[(np.abs(stats.zscore(ds.price)) < 3)]
        #df = remove_outlier(ds, 'price')
        print("After removing price_outliers: ", df.shape)


        df1 = ds[(np.abs(stats.zscore(ds.term)) < 3)]
        #df1 = remove_outlier(ds, 'term')
        print("After removing term_outliers: ", df1.shape)

        clean_data = pd.merge(df, df1, on=list(ds.columns))
        kmeans = KMeans(n_clusters=100, random_state=42).fit(clean_data[['longitude', 'latitude']])

        dump(kmeans, PATH_TO_TIME_MODEL + '/KMEAN_CLUSTERIZATION.joblib')
        labels = kmeans.labels_
        clean_data['clusters'] = labels

        print('Saving to new csv')
        clean_data.to_csv(prepared_data+'/COORDINATES_Pred_Term.csv', index=None, header=True)


if __name__ == '__main__':
    main_preprocessing()
