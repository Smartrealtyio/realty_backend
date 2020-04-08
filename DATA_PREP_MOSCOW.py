import pandas as pd
import numpy as np
from sklearn import preprocessing
import backports.datetime_fromisoformat as bck
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump
import settings_local as SETTINGS
from numpy.random import randint
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy import stats
import time, ciso8601
from sklearn.cluster import KMeans
import math as m
np.random.seed(42)

# Define paths
raw_data = SETTINGS.PATH_TO_SINGLE_CSV_FILES_MOSCOW
PREPARED_DATA = SETTINGS.DATA_MOSCOW
PATH_TO_CLUSTERING_MODELS = SETTINGS.MODEL_MOSCOW



class MainPreprocessing():
    """Create class for data preprocessing"""
    def __init__(self):
        """Initialize class"""
        pass

    def load_and_merge(self, raw_data: str):
        prices = pd.read_csv(raw_data + "prices.csv", names=[
            'id', 'price', 'changed_date', 'flat_id', 'created_at', 'updated_at'
        ], usecols=["price", "flat_id", 'created_at', 'changed_date', 'updated_at'])


        # Count number of price changing for each unique flat and SORT changed_date for each subgroup (group consist of one flat)
        prices['nums_of_changing'] = prices.sort_values(['changed_date'][-9:], ascending=True).groupby(['flat_id'])[
            "flat_id"].transform("count")
        # Group by falt_id and sort in ascending order for term counting
        # prices = prices.sort_values(['changed_date'][-9:],ascending=True).groupby('flat_id')

        # Keep just first date
        prices = prices.drop_duplicates(subset='flat_id', keep="first")
        prices = prices[((prices['changed_date'].str.contains('2020')) | (prices['changed_date'].str.contains('2019')) | (
            prices['changed_date'].str.contains('2018')))]


        # Calculating selling term. TIME UNIT: DAYS
        prices['term'] = prices[['updated_at', 'changed_date']].apply(
            lambda row: (bck.date_fromisoformat(row['updated_at'][:-9])
                         - bck.date_fromisoformat(row['changed_date'][:-9])).days, axis=1)

        flats = pd.read_csv(raw_data + "flats.csv",
                            names=['id', 'full_sq', 'kitchen_sq', 'life_sq', 'floor', 'is_apartment',
                                                     'building_id', 'created_at',
                                                     'updated_at', 'offer_id', 'closed', 'rooms', 'image', 'resource_id',
                                                            'flat_type', 'is_rented', 'rent_quarter', 'rent_year', 'agency'],
                            usecols=["id", "full_sq",
                                                       "kitchen_sq",
                                                       "life_sq",
                                                       "floor", "is_apartment",
                                                       "building_id",
                                                       "closed", 'rooms', 'resource_id', 'flat_type', 'is_rented', 'rent_quarter',
                                                       'rent_year'
                                                       ],
                            true_values="t", false_values="f", header=0)

        # Replace all missed values in FLAT_TYPE with 'SECONDARY'
        flats.flat_type = flats['flat_type'].fillna('SECONDARY')

        # Replace all missed values in CLOSED with 'False'
        flats.closed = flats.closed.fillna(False)

        # Leave only VTORICHKA
        # flats = flats[flats.flat_type == 'SECONDARY']
        flats = flats.rename(columns={"id": "flat_id"})

        buildings = pd.read_csv(raw_data + "buildings.csv",
                                names=["id", "max_floor", 'building_type_str', "built_year", "flats_count",
                                       "address", "renovation",
                                       "has_elevator",
                                       'longitude', 'latitude',
                                       "district_id",
                                       'created_at',
                                       'updated_at', 'schools_500m', 'schools_1000m', 'kindergartens_500m',
                                       'kindergartens_1000m', 'clinics_500m', 'clinics_1000m', 'shops_500m',
                                       'shops_1000m'],
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

        # Merage prices and flats on flat_id
        prices_and_flats = pd.merge(prices, flats, on='flat_id', how="left")


        # Merge districts and buildings on district_id
        districts_and_buildings = pd.merge(districts, buildings, on='district_id', how='right')


        # Merge to one main DF on building_id
        df = pd.merge(prices_and_flats, districts_and_buildings, on='building_id', how='left')

        # Merge main DF and time_to_metro on building_id, fill the zero value with the mean value
        df = pd.merge(df, time_to_metro, on="building_id", how='left')
        # df[['time_to_metro']] = df[['time_to_metro']].apply(lambda x: x.fillna(x.mean()), axis=0)
        df.time_to_metro = df.time_to_metro.fillna(df.time_to_metro.mean())


        # Check if main DF constains null values
        # print(df.isnull().sum())

        # Drop all offers without important data
        df = df.dropna(subset=['full_sq'])

        # Replace missed "IS_RENTED" with 1 and convert bool -> int
        df.is_rented = df.is_rented.fillna(True)
        df.is_rented = df.is_rented.astype(int)

        df.rooms = df.rooms.fillna(0)
        df = df.fillna(0)


        # Replace missed value 'RENT_YEAR' with posted year
        # now = datetime.datetime.now()
        df.rent_year = df.rent_year.fillna(df.changed_date.apply(lambda x: x[:4]))

        # Replace missed value "RENT_QUARTER" with current quarter, when value was posted
        df.rent_quarter = df.rent_quarter.fillna(df.changed_date.apply(lambda x: x[5:7]))
        df.rent_quarter = df.rent_quarter.astype(int)
        df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 12, 4, 4)
        df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 9, 3, df.rent_quarter)
        df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 6, 2, df.rent_quarter)
        df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 3, 1, df.rent_quarter)

        # Transform bool values to int
        df.has_elevator = df.has_elevator.astype(int)
        df.renovation = df.renovation.astype(int)
        df.is_apartment = df.is_apartment.astype(int)
        df.has_elevator = df.has_elevator.astype(int)
        df.renovation = df.renovation.astype(int)
        df.is_apartment = df.is_apartment.astype(int)
        df.rent_year = df.rent_year.astype(int)

        df = df.drop(['built_year', 'flats_count', 'district_id', 'name', 'transport_type'], axis=1)

        # Set values for floor_last/floor_first column: if floor_last/floor_first set 1, otherwise 0
        # max_floor_list = df['max_floor'].tolist()
        df['floor_last'] = np.where(df['max_floor'] == df['floor'], 1, 0)
        df['floor_first'] = np.where(df['floor'] == 1, 1, 0)

        # Replace all negative values with zero
        num = df._get_numeric_data()
        num[num < 0] = 0


        # Count price per meter square for each flat
        df['price_meter_sq'] = df[['price', 'full_sq']].apply(
            lambda row: (row['price'] /
                         row['full_sq']), axis=1)

        # Check if data contains only Moscow offers
        df = df[~(df['latitude'].astype('str').str.contains('59.'))]
        return df


    def new_features(self, data: pd.DataFrame(), full_sq_corridor_percent: float, price_corridor_percent: float, part_data: int):
        df = data
        # No 1. Distance from city center
        Moscow_center_lon = 37.619291
        Moscow_center_lat = 55.751474
        df['to_center'] = abs(Moscow_center_lon - df['longitude']) + abs(Moscow_center_lat - df['latitude'])

        # No 2. Fictive(for futher offer value calculating): yyyy_announc, mm_announc - year and month when flats were announced on market
        df['yyyy_announce'] = df['changed_date'].str[:4].astype('int64')
        df['mm_announce'] = df['changed_date'].str[5:7].astype('int64')

        # No 3. Number of offers were added calculating by months and years
        df['all_offers_added_in_month'] = df.groupby(['yyyy_announce', 'mm_announce'])["flat_id"].transform("count")

        # No 4. Convert changed_date and updated_at to unix timestamp. Convert only yyyy-mm-dd hh
        df['open_date_unix'] = df['changed_date'].apply(
            lambda row: int(time.mktime(ciso8601.parse_datetime(row[:-3]).timetuple())))
        df['close_date_unix'] = df['updated_at'].apply(
            lambda row: int(time.mktime(ciso8601.parse_datetime(row[:-3]).timetuple())))

        # Take just part of data
        if part_data:
            df = df.iloc[:len(df) // part_data]
        # Calculate number of "similar" flats which were on market when each closed offer was closed.
        df['was_opened'] = [np.sum((df['open_date_unix'] < close_time) & (df['close_date_unix'] >= close_time) &
                                   (df['rooms'] == rooms) &
                                   ((df['full_sq'] <= full_sq * (1 + full_sq_corridor_percent / 100)) & (
                                           df['full_sq'] >= full_sq * (1 - full_sq_corridor_percent / 100))) &
                                   ((df['price'] <= price * (1 + price_corridor_percent / 100)) & (
                                           df['price'] >= price * (1 - price_corridor_percent / 100)))) for
                            close_time, rooms, full_sq, price in
                            zip(df['close_date_unix'], df['rooms'], df['full_sq'], df['price'])]

        # Fill missed valeus for secondary flats
        df.loc[:, ['rent_quarter', 'rent_year']] = df[['rent_quarter', 'rent_year']].fillna(0)
        df.loc[:, 'is_rented'] = df[['is_rented']].fillna(1)

        # Get lenght of all df
        len_df = len(df)

        def add_fictive_rows(data: pd.DataFrame(), len_df: int):

            for i in range(len(data), len(data)+8):
                data.loc[i, 'rooms'] = i%len_df

            updated_len_df = len(data)
            for i in range(len(data)+1, len(data)+13):
                data.loc[i, 'mm_announce'] = i%updated_len_df

            updated_len_df = len(data)
            for i in range(len(data), len(data)+130):
                data.loc[i, 'clusters'] = i%updated_len_df

            return data

        df = add_fictive_rows(data=df, len_df=len_df)

        df = df[df.rooms < 7]

        # Transform bool values to int
        df.rooms = df.rooms.fillna(df.rooms.mode()[0])
        df.rooms = df.rooms.astype(int)
        df.mm_announce = df.mm_announce.fillna(df.mm_announce.mode()[0])
        df.mm_announce = df.mm_announce.astype(int)
        df.yyyy_announce = df.yyyy_announce.fillna(df.yyyy_announce.mode()[0])
        df.yyyy_announce = df.yyyy_announce.astype(int)
        return df



    def clustering(self, data: pd.DataFrame(), path_kmeans_models: str):
        # fit k-Means clustering on geo for SECONDARY flats

        data.longitude= data.longitude.fillna(data.longitude.mode()[0])
        data.latitude= data.latitude.fillna(data.latitude.mode()[0])
        kmeans = KMeans(n_clusters=130, random_state=42).fit(data[['longitude', 'latitude']])
        dump(kmeans, path_kmeans_models + '/KMEANS_CLUSTERING_MOSCOW_MAIN.joblib')
        labels = kmeans.labels_
        data['clusters'] = labels

        data.clusters = data.clusters.astype(int)

        # Create dummies from cluster
        df_clusters = pd.get_dummies(data, prefix='cluster_', columns=['clusters'])
        data = pd.merge(data, df_clusters, how='left')
        return data

    # Transform some features (such as mm_announce, rooms, clusters) to dummies
    def to_dummies(self, data: pd.DataFrame):
        df_mm_announce = pd.get_dummies(data, prefix='mm_announce_', columns=['mm_announce'])
        df_rooms = pd.get_dummies(data, prefix='rooms_', columns=['rooms'])
        df = pd.merge(df_mm_announce, df_rooms, how='left')

        df = df.dropna(subset=['full_sq'])
        print(df.columns, flush=True)
        print("After transform to dummies features: ", df.shape)
        return df

    def train_price_model(self, data: pd.DataFrame):

        df = data
        data1 = df[(np.abs(stats.zscore(df.full_sq)) < 3)]
        data2 = df[(np.abs(stats.zscore(df.life_sq)) < 3)]
        data3 = df[(np.abs(stats.zscore(df.kitchen_sq)) < 3)]

        # Merge data1 and data2
        df = pd.merge(data1, data2, on=list(df.columns), how='left')
        # Fill NaN if it appears after merging
        df[['life_sq']] = df[['life_sq']].fillna(df[['life_sq']].mean())

        # Merge df and data3
        df = pd.merge(df, data3, on=list(df.columns), how='left')
        # Fill NaN if it appears after merging
        df[['kitchen_sq']] = df[['kitchen_sq']].fillna(df[['kitchen_sq']].mean())

        # Drop unnecessary columns
        df = data.drop(
            ['close_date_unix', 'open_date_unix', 'all_offers_added_in_month', 'clusters', 'price_meter_sq', 'latitude',
             'longitude',
             'building_type_str', 'max_floor', 'flat_type', 'resource_id', 'rooms',
             'building_id', 'closed', 'floor', 'term', 'nums_of_changing', 'updated_at', 'created_at',
             'flat_id', 'changed_date'], axis=1)

        # Save leaved columns to variable
        columns = list(df.columns)
        # df = df[['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
        # 			 'time_to_metro', 'floor_last', 'floor_first', 'clusters', 'price', 'is_rented', 'rent_quarter',
        # 			 'rent_year']]

        # Log transformation
        # df["longitude"] = np.log1p(df["longitude"])
        # df["latitude"] = np.log1p(df["latitude"])
        df["full_sq"] = np.log1p(df["full_sq"])
        df["life_sq"] = np.log1p(df["life_sq"])
        df["kitchen_sq"] = np.log1p(df["kitchen_sq"])
        df["price"] = np.log1p(df["price"])

        # Create features - predictors
        X = df.drop(['price'], axis=1)

        # Target feature
        y = df[['price']].values.ravel()

        # Split for train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Define Gradient Boosting Machine model
        gbr_model = GradientBoostingRegressor(n_estimators=450, max_depth=8, verbose=1, max_features='sqrt',
                                              random_state=42,
                                              learning_rate=0.07)
        # Train GBR on train dataset
        gbr_model.fit(X_train, y_train)
        gbr_preds = gbr_model.predict(X_test)
        print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds), flush=True)
        print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)

        # Train GBR on full dataset
        gbr_model.fit(X, y)
        return gbr_model, columns

    def calculate_profit(self, data: pd.DataFrame, price_model: GradientBoostingRegressor, list_of_columns: list):
        # price_model = load('/content/drive/My Drive/DDG/developers/models/MOSCOW_VTOR_PRICE_GBR.joblib')

        data = data[data.closed == True]
        data.loc[:, 'pred_price'] = data[list_of_columns].apply(
            lambda row:
            int(np.expm1(price_model.predict(
                [[np.log1p(row.full_sq), np.log1p(row.kitchen_sq), np.log1p(row.life_sq), row.is_apartment,
                  row.renovation, row.has_elevator,
                  row.time_to_metro, row.floor_first, row.floor_last,
                  row.is_rented, row.rent_quarter,
                  row.rent_year, row.to_center, row.yyyy_announce, row.was_opened, row.mm_announce__1,
                  row.mm_announce__2, row.mm_announce__3, row.mm_announce__4,
                  row.mm_announce__5, row.mm_announce__6, row.mm_announce__7, row.mm_announce__8, row.mm_announce__9,
                  row.mm_announce__10, row.mm_announce__11, row.mm_announce__12, row.mm_announce, row.rooms__0,
                  row.rooms__1, row.rooms__2,
                  row.rooms__3, row.rooms__4, row.rooms__5, row.rooms__6,
                  row.cluster__0, row.cluster__1,
                  row.cluster__2, row.cluster__3, row.cluster__4, row.cluster__5, row.cluster__6, row.cluster__8,
                  row.cluster__9, row.cluster__10, row.cluster__11,
                  row.cluster__11, row.cluster__12, row.cluster__13, row.cluster__14, row.cluster__15, row.cluster__16,
                  row.cluster__17, row.cluster__18, row.cluster__19,
                  row.cluster__20, row.cluster__21, row.cluster__22, row.cluster__23, row.cluster__24,
                  row.cluster__25, row.cluster__26, row.cluster__27, row.cluster__28, row.cluster__29, row.cluster__30,
                  row.cluster__31, row.cluster__32,
                  row.cluster__33, row.cluster__34, row.cluster__35, row.cluster__36, row.cluster__37, row.cluster__38,
                  row.cluster__39, row.cluster__40,
                  row.cluster__41, row.cluster__42, row.cluster__43, row.cluster__44, row.cluster__45, row.cluster__46,
                  row.cluster__47,
                  row.cluster__48, row.cluster__49, row.cluster__50, row.cluster__51, row.cluster__52, row.cluster__53,
                  row.cluster__54, row.cluster__55,
                  row.cluster__56, row.cluster__57, row.cluster__58, row.cluster__59, row.cluster__60, row.cluster__61,
                  row.cluster__62, row.cluster__63, row.cluster__64, row.cluster__65, row.cluster__66, row.cluster__67,
                  row.cluster__68, row.cluster__69,
                  row.cluster__70, row.cluster__71, row.cluster__72, row.cluster__73, row.cluster__74, row.cluster__75,
                  row.cluster__76, row.cluster__77,
                  row.cluster__78, row.cluster__79, row.cluster__80, row.cluster__81, row.cluster__82, row.cluster__83,
                  row.cluster__84,
                  row.cluster__85, row.cluster__86, row.cluster__87, row.cluster__88, row.cluster__89, row.cluster__90,
                  row.cluster__91, row.cluster__92,
                  row.cluster__93, row.cluster__94, row.cluster__95, row.cluster__96, row.cluster__97, row.cluster__98,
                  row.cluster__99, row.cluster__100, row.cluster__101, row.cluster__102, row.cluster__103,
                  row.cluster__104, row.cluster__105, row.cluster__106,
                  row.cluster__107, row.cluster__108, row.cluster__109, row.cluster__110, row.cluster__111,
                  row.cluster__112, row.cluster__113, row.cluster__114,
                  row.cluster__115, row.cluster__116, row.cluster__117, row.cluster__119, row.cluster__120,
                  row.cluster__121, row.cluster__122,
                  row.cluster__123, row.cluster__124, row.cluster__125, row.cluster__126, row.cluster__127,
                  row.cluster__128, row.cluster__129]]))[0]), axis=1)

        data.loc[:, 'profit'] = data[['pred_price', 'price']].apply(
            lambda row: ((row.pred_price * 100 / row.price) - 100), axis=1)
        # Handle negative profit values
        data.loc[:, 'profit'] = data['profit'] + 1 - data['profit'].min()
        print(data[['pred_price', 'price', 'profit', 'term', 'changed_date', 'updated_at']].head(2))
        return data


    def secondary_flats(self, data: pd.DataFrame(), path_to_save_data: str):
        # Create df with SECONDARY flats
        df_VTOR = data[(data.flat_type == 'SECONDARY')]


        # Save .csv with SECONDARY flats
        print('Saving secondary to csv', df_VTOR.shape[0], flush=True)
        df_VTOR.to_csv(path_to_save_data + '/MOSCOW_VTOR.csv', index=None, header=True)


    def new_flats(self, data:pd.DataFrame(), path_to_save_data: str):

        # Create df with NEW flats
        df_new_flats = data[((data.flat_type == 'NEW_FLAT')|(data.flat_type == 'NEW_SECONDARY'))]

        # fit k-Means clustering on geo for NEW flats
        # kmeans_NEW_FLAT = KMeans(n_clusters=30, random_state=42).fit(df_new_flats[['longitude', 'latitude']])
        # dump(kmeans_NEW_FLAT, path_kmeans_models + '/KMEAN_CLUSTERING_MOSCOW_NEW_FLAT.joblib')
        # labels = kmeans_NEW_FLAT.labels_
        # df_new_flats['clusters'] = labels

        # Save .csv with NEW flats
        print('Saving new to csv', df_new_flats.shape[0], flush=True)
        df_new_flats.to_csv(path_to_save_data + '/MOSCOW_NEW_FLATS.csv', index=None, header=True)


if __name__ == '__main__':

    full_sq_corridor_percent = 1.5
    price_corridor_percent = 1.5

    # Create obj MainPreprocessing
    mp = MainPreprocessing()

    # Load data
    print("Load data...", flush=True)
    df = mp.load_and_merge(raw_data=raw_data)
    df = df.iloc[:1000]

    # Generate new features
    print("Generate new features...", flush=True)
    features_data = mp.new_features(data=df, full_sq_corridor_percent=full_sq_corridor_percent,
                                    price_corridor_percent=price_corridor_percent, part_data=False)

    # Define clusters
    print("Defining clusters based on lon, lat...")
    cl_data = mp.clustering(features_data, path_kmeans_models=PATH_TO_CLUSTERING_MODELS)

    # Create dummies variables
    print("Transform to dummies...", flush=True)
    cat_data = mp.to_dummies(cl_data)


    # Train price model
    print("Train price model...", flush=True)
    price_model, list_columns = mp.train_price_model(data=cat_data)

    # Calculate profit for each flat
    print("Calculating profit for each offer in dataset...", flush=True)
    test = mp.calculate_profit(data=cat_data, price_model=price_model, list_of_columns=list_columns)

    # Create separate files for secondary flats
    print("Save secondary flats csv.")
    mp.secondary_flats(data=cat_data, path_to_save_data=PREPARED_DATA)

    # Create sepatare files for new flats
    print("Save new flats csv.")
    mp.new_flats(data=cl_data, path_to_save_data=PREPARED_DATA)

