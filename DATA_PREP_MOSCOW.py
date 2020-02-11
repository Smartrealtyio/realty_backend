import pandas as pd
import numpy as np
from sklearn import preprocessing
import backports.datetime_fromisoformat as bck
from joblib import dump
import settings_local as SETTINGS
from scipy import stats
from sklearn.cluster import KMeans

# FINAL PARAMETERS ORDER:
# ['building_type_str', 'renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'term', 'full_sq', 'kitchen_sq',
# 'life_sq', 'is_apartment', 'time_to_metro', 'floor_last', 'floor_first']
raw_data = SETTINGS.PATH_TO_SINGLE_CSV_FILES_MOSCOW
prepared_data = SETTINGS.DATA_MOSCOW
PATH_TO_TIME_MODEL = SETTINGS.MODEL_MOSCOW


def main_preprocessing():

        # DATA preprocessing #
        prices = pd.read_csv(raw_data + "prices.csv", names=[
            'id', 'price', 'changed_date', 'flat_id', 'created_at', 'updated_at'
        ], usecols=["price", "flat_id", 'created_at', 'changed_date', 'updated_at'])
        print("With duplicates: ", len(prices.flat_id.unique()))
        prices = prices.drop_duplicates(subset='flat_id', keep="last")
        print(prices.shape)
        print("Prices all years: ", prices.shape)
        prices = prices[((prices['changed_date'].str.contains('2018'))|(prices['changed_date'].str.contains('2020')) | (prices['changed_date'].str.contains('2019')))]
        print("Prices 2018/2019/2020 yearS: ", prices.shape)

        # Calculating selling term. TIME UNIT: DAYS
        prices['term'] = prices[['updated_at', 'changed_date']].apply(
            lambda row: (bck.date_fromisoformat(row['updated_at'][:-9])
                         - bck.date_fromisoformat(row['changed_date'][:-9])).days, axis=1)
        flats = pd.read_csv(raw_data + "flats.csv",
                            names=['id', 'full_sq', 'kitchen_sq', 'life_sq', 'floor', 'is_apartment',
                                   'building_id', 'created_at',
                                   'updated_at', 'offer_id', 'closed', 'rooms', 'image', 'resource_id', 'flat_type'],
                            usecols=["id", "full_sq",
                                     "kitchen_sq",
                                     "life_sq",
                                     "floor", "is_apartment",
                                     "building_id",
                                     "closed", 'rooms', 'resource_id', 'flat_type'
                                     ],
                            true_values="t", false_values="f", header=0)

        # Leave only VTORICHKA
        flats = flats[flats.flat_type == 'SECONDARY']
        flats = flats.rename(columns={"id": "flat_id"})
        print("flats: ", flats.shape)
        flats = flats.drop_duplicates(subset='flat_id', keep="last")
        print("flats_drop_duplicates: ", flats.shape)
        buildings = pd.read_csv(raw_data + "buildings.csv",
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
        districts = pd.read_csv(raw_data + "districts.csv", names=['id', 'name', 'population', 'city_id',
                                                                   'created_at', 'updated_at', 'prefix'],
                                usecols=["name", 'id'],
                                true_values="t", false_values="f", header=0)
        districts = districts.rename(columns={"id": "district_id"})
        buildings = buildings.rename(columns={"id": "building_id"})
        time_to_metro = pd.read_csv(raw_data + "time_metro_buildings.csv",
                                    names=['id', 'building_id', 'metro_id', 'time_to_metro',
                                           'transport_type', 'created_at', 'updated_at'],
                                    usecols=["building_id", "time_to_metro", "transport_type"], header=0)

        # Sort time_to_metro values
        time_to_metro = time_to_metro[time_to_metro['transport_type'] == "ON_FOOT"].sort_values('time_to_metro',
                                                                                                ascending=True)

        # Keep just shortest time to metro
        time_to_metro = time_to_metro.drop_duplicates(subset='building_id', keep="first")
        print('time_to_metro: \n', time_to_metro.iloc[:5])

        # Merage prices and flats on flat_id
        prices_and_flats = pd.merge(prices, flats, on="flat_id")
        print("Prices + Flats: ", prices_and_flats.shape)

        # Merge districts and buildings on district_id
        districts_and_buildings = pd.merge(districts, buildings, on='district_id')
        print("districts + buildings: ", districts_and_buildings.shape)

        # Merge to one main DF on building_id
        df = pd.merge(prices_and_flats, districts_and_buildings, on='building_id')
        print("Without metro: ", df.shape)

        # Merge main DF and time_to_metro on building_id, fill the zero value with the mean value
        df = pd.merge(df, time_to_metro, on="building_id", how='left')
        df[['time_to_metro']] = df[['time_to_metro']].apply(lambda x: x.fillna(x.mean()), axis=0)
        print('plus metro: ', df.shape)

        # Drop categorical column
        df = df.drop(['transport_type'], axis=1)

        # Check if main DF constains null values
        print(df.isnull().sum())

        df = df.drop(['built_year', 'flats_count', 'district_id', 'name'], axis=1)

        # Transform bool values to int
        df.has_elevator = df.has_elevator.astype(int)
        df.renovation = df.renovation.astype(int)
        df.is_apartment = df.is_apartment.astype(int)

        # Set values for floor_last/floor_first column: if floor_last/floor_first set 1, otherwise 0
        max_floor_list = df['max_floor'].tolist()
        df['floor_last'] = np.where(df['max_floor'] == df['floor'], 1, 0)
        df['floor_first'] = np.where(df['floor'] == 1, 1, 0)

        print("Check if there are duplicates in dataframe: ", df.shape)
        df = df.drop_duplicates(subset='flat_id', keep="last")
        print("Check if there are duplicates in dataframe: ", df.shape)
        print(df.shape)

        num = df._get_numeric_data()

        num[num < 0] = 0

        df1 = df[(np.abs(stats.zscore(df.price)) < 3)]

        df2 = df[(np.abs(stats.zscore(df.term)) < 3)]

        print("After removing term_outliers: ", df2.shape)
        print("After removing price_outliers: ", df1.shape)
        clean_data = pd.merge(df1, df2, on=list(df.columns))

        clean_data_VTOR = clean_data[(clean_data.flat_type == 'SECONDARY')]
        kmeans_VTOR = KMeans(n_clusters=150, random_state=42).fit(clean_data_VTOR[['longitude', 'latitude']])
        dump(kmeans_VTOR, PATH_TO_TIME_MODEL + '/KMEAN_CLUSTERING_MOSCOW_VTOR.joblib')
        labels = kmeans_VTOR.labels_
        clean_data_VTOR['clusters'] = labels

        print("MOSCOW headers finally: ", list(clean_data_VTOR.columns), flush=True)
        print('Saving to new csv', clean_data_VTOR.shape[0], flush=True)
        clean_data_VTOR.to_csv(prepared_data + '/MOSCOW_VTOR.csv', index=None, header=True)


        clean_data_new_flats = clean_data[((clean_data.flat_type == 'NEW_FLAT')|(clean_data.flat_type == 'NEW_SECONDARY'))]
        kmeans_NEW_FLAT = KMeans(n_clusters=30, random_state=42).fit(clean_data_new_flats[['longitude', 'latitude']])
        dump(kmeans_NEW_FLAT, PATH_TO_TIME_MODEL + '/KMEAN_CLUSTERING_MOSCOW_NEW_FLAT.joblib')
        labels = kmeans_NEW_FLAT.labels_
        clean_data_new_flats['clusters'] = labels


        print("MOSCOW headers finally: ", list(clean_data.columns), flush=True)
        print('Saving to new csv', clean_data_new_flats.shape[0], flush=True)
        clean_data_new_flats.to_csv(prepared_data + '/MOSCOW_NEW_FLATS.csv', index=None, header=True)


if __name__ == '__main__':
    main_preprocessing()
