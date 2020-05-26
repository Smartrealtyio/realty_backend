from scipy import stats
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from joblib import dump, load
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os
from math import sin, cos, sqrt, atan2, radians
import json

machine = os.path.abspath(os.getcwd())

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import settings_local as SETTINGS

# Define paths to Moscow and Spb Secondary flats models DUMMIES
PATH_PRICE_GBR_MOSCOW_D = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_GBR_D.joblib'
PATH_PRICE_RF_MOSCOW_D = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_RF_D.joblib'
PATH_PRICE_LGBM_MOSCOW_D = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_LGBM_D.joblib'
PATH_PRICE_GBR_SPB_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_GBR_D.joblib'
PATH_PRICE_RF_SPB_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_RF_D.joblib'
PATH_PRICE_LGBM_SPB_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_LGBM_D.joblib'

# Define paths to Moscow and Spb clustering models
KMEANS_CLUSTERING_MOSCOW_MAIN = SETTINGS.MODEL_MOSCOW + '/KMEANS_CLUSTERING_MOSCOW_MAIN.joblib'
KMEANS_CLUSTERING_SPB_MAIN = SETTINGS.MODEL_SPB + '/KMEANS_CLUSTERING_SPB_MAIN.joblib'

# Define paths to Moscow and Spb data
MOSCOW_DATA_NEW = SETTINGS.DATA_MOSCOW + '/MOSCOW_NEW_FLATS.csv'
MOSCOW_DATA_SECONDARY = SETTINGS.DATA_MOSCOW + '/MOSCOW_VTOR.csv'
SPB_DATA_NEW = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'
SPB_DATA_SECONDARY = SETTINGS.DATA_SPB + '/SPB_VTOR.csv'


# Find profitable offers
def mean_estimation(full_sq_from, full_sq_to, latitude_from, latitude_to, longitude_from, longitude_to, rooms,
                    price_from, price_to, building_type_str, kitchen_sq, life_sq, renovation, has_elevator, floor_first,
                    floor_last, time_to_metro, city_id):
    # Initialize DF
    data_offers = pd.DataFrame()

    # Set paths to data and price prediction models, depending on city:  0 = Moscow, 1 = Spb
    if city_id == 0:
        data_offers = pd.read_csv(MOSCOW_DATA_SECONDARY)
        # data_offers = data_offers[data_offers.flat_type == 'SECONDARY']
        gbr = load(PATH_PRICE_GBR_MOSCOW_D)
        rf = load(PATH_PRICE_RF_MOSCOW_D)
        lgbm = load(PATH_PRICE_LGBM_MOSCOW_D)
    elif city_id == 1:
        data_offers = pd.read_csv(SPB_DATA_SECONDARY)
        # data_offers = data_offers[data_offers.flat_type == 'SECONDARY']
        gbr = load(PATH_PRICE_GBR_SPB_D)
        rf = load(PATH_PRICE_RF_SPB_D)
        lgbm = load(PATH_PRICE_LGBM_SPB_D)

    # Apply filtering flats in database on parameters: full_sq range, coordinates scope
    filter = (((data_offers.full_sq >= full_sq_from) & (data_offers.full_sq <= full_sq_to)) & (
            data_offers.rooms == rooms) &
              ((data_offers.latitude >= latitude_from) & (data_offers.latitude <= latitude_to))
              & ((data_offers.longitude >= longitude_from) & (data_offers.longitude <= longitude_to)))
    data_offers = data_offers[filter]

    # Use only open offers
    data_offers = data_offers[data_offers['closed'] == False]

    print('columns ', data_offers.columns, flush=True)

    if time_to_metro != None:
        data_offers = data_offers[(data_offers.time_to_metro <= time_to_metro)]
    if rooms != None:
        data_offers = data_offers[data_offers.rooms == rooms]
    if building_type_str != None:
        data_offers = data_offers[data_offers.building_type_str == building_type_str]
    if kitchen_sq != None:
        data_offers = data_offers[
            (data_offers.kitchen_sq >= kitchen_sq - 1) & (data_offers.kitchen_sq <= kitchen_sq + 1)]
    if life_sq != None:
        data_offers = data_offers[(data_offers.life_sq >= life_sq - 5) & (data_offers.life_sq <= life_sq + 5)]
    if renovation != None:
        data_offers = data_offers[data_offers.renovation == renovation]
    if has_elevator != None:
        data_offers = data_offers[data_offers.has_elevator == has_elevator]
    if floor_first != None:
        data_offers = data_offers[data_offers.floor_first == 0]
    if floor_last != None:
        data_offers = data_offers[data_offers.floor_last == 0]
    if price_from != None:
        data_offers = data_offers[data_offers.price >= price_from]
    if price_to != None:
        data_offers = data_offers[data_offers.price <= price_to]

    # PRICE PREDICTION
    # data_offers['pred_price'] = data_offers[
    #     ['life_sq', 'to_center', 'mm_announce', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
    #      'time_to_metro', 'floor_last', 'floor_first', 'clusters', 'is_rented', 'rent_quarter', 'rent_year']].apply(
    #     lambda row:
    #     int((np.expm1(
    #         rf.predict([[np.log1p(row.life_sq), np.log1p(row.to_center), row.mm_announce, row.rooms, row.renovation, row.has_elevator, np.log1p(row.longitude),
    #                      np.log1p(row.latitude), np.log1p(row.latitude),
    #                      np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_first, row.floor_last,
    #                      row.clusters, row.is_rented, row.rent_quarter, row.rent_year]])) + np.expm1(
    #         lgbm.predict([[np.log1p(row.life_sq), np.log1p(row.to_center), row.mm_announce, row.rooms, row.renovation, row.has_elevator, np.log1p(row.longitude),
    #                        np.log1p(row.latitude), np.log1p(row.latitude),
    #                        np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_first, row.floor_last,
    #                        row.clusters, row.is_rented, row.rent_quarter, row.rent_year]])))[0] / 2), axis=1)
    #
    # # Calculate the profitability for each flat knowing current and the price that our model predicted
    # data_offers['profit'] = data_offers[['pred_price', 'price']].apply(
    #     lambda row: ((row.pred_price * 100 / row.price) - 100), axis=1)

    # Set threshold for showing profitable offers
    print(data_offers.shape, flush=True)
    data_offers = data_offers[(data_offers.profit >= 5)]
    print(data_offers.shape, flush=True)
    data_offers = data_offers.sort_values(by=['profit'], ascending=False)
    print("Profitable offers: ", data_offers[['pred_price', "price", 'profit']].head(3), flush=True)

    flats = data_offers.to_dict('record')

    return flats


# Predict price and term
def map_estimation(longitude, rooms, latitude, full_sq, kitchen_sq, life_sq, renovation, secondary, has_elevator,
                   floor_first, floor_last, time_to_metro, is_rented, rent_year, rent_quarter, city_id):
    # Get current time
    now = datetime.now()

    # City_id: 0 = Moscow, 1 = Spb

    def define_city(city_id: int, secondary: int):

        city_center_lon = 0
        city_center_lat = 0

        data = pd.DataFrame()
        kmeans, gbr, rf, lgbm = 0, 0, 0, 0
        if city_id == 0:
            # Load data Moscow flats
            data1 = pd.read_csv(MOSCOW_DATA_NEW)
            data2 = pd.read_csv(MOSCOW_DATA_SECONDARY)
            data = pd.concat([data1, data2], ignore_index=True)

            # Load KMean Clustering model
            kmeans = load(KMEANS_CLUSTERING_MOSCOW_MAIN)

            # Load Price Models Moscow Secondary
            gbr = load(PATH_PRICE_GBR_MOSCOW_D)
            rf = load(PATH_PRICE_RF_MOSCOW_D)
            lgbm = load(PATH_PRICE_LGBM_MOSCOW_D)

            city_center_lon = 37.619291
            city_center_lat = 55.751474


        # # Москва вторичка
        # elif city_id == 0 and secondary == 1:
        #     # Load data Moscow secondary
        #     data = pd.read_csv(MOSCOW_DATA_SECONDARY)
        #
        #     # Load KMean Clustering model
        #     kmeans = load(KMEANS_CLUSTERING_MOSCOW_MAIN)
        #
        #     # Load Price Models Moscow Secondary
        #     gbr = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        #     rf = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        #     lgbm = load(PATH_PRICE_GBR_MOSCOW_VTOR)

        # Санкт-Петербург новостройки
        elif city_id == 1:
            # Load data SPb
            data1 = pd.read_csv(SPB_DATA_NEW)
            data2 = pd.read_csv(SPB_DATA_SECONDARY)
            data = pd.concat([data1, data2], ignore_index=True)

            # Load KMean Clustering model
            kmeans = load(KMEANS_CLUSTERING_SPB_MAIN)

            # Load Price Models Spb Secondary
            gbr = load(PATH_PRICE_GBR_SPB_D)
            rf = load(PATH_PRICE_RF_SPB_D)
            lgbm = load(PATH_PRICE_LGBM_SPB_D)

            city_center_lon = 30.315239
            city_center_lat = 59.940735

        # # Санкт-Петербург вторичка
        # elif city_id == 1 and secondary == 1:
        #     data = pd.read_csv(SPB_DATA_SECONDARY)
        #     # Load KMean Clustering model
        #     kmeans = load(KMEANS_CLUSTERING_SPB_MAIN)
        #
        #     # Load Price Models Spb Secondary
        #     gbr = load(PATH_PRICE_GBR_SPB_VTOR)
        #     rf = load(PATH_PRICE_RF_SPB_VTOR)
        #     lgbm = load(PATH_PRICE_LGBM_SPB_VTOR)

        print("Initial shape: ", data.shape, flush=True)
        return data, kmeans, gbr, rf, lgbm, city_center_lon, city_center_lat

    # Call define function
    data, kmeans, gbr, rf, lgbm, city_center_lon, city_center_lat = define_city(city_id=city_id, secondary=secondary)

    ####################
    #                  #
    # PRICE PREDICTION #
    #                  #
    ####################

    # Calculate distance to city_center
    # No 1. Distance from city center in km

    # approximate radius of earth in km
    R = 6373.0

    to_city_center_distance = R * 2 * atan2(sqrt(sin((radians(latitude) - radians(city_center_lat)) / 2)
                                                 ** 2 + cos(radians(city_center_lat)) * cos(radians(city_center_lat))
                                                 * sin((radians(longitude) - radians(city_center_lon)) / 2) ** 2),
                                            sqrt(1 - (sin((radians(latitude) - radians(city_center_lat)) / 2)
                                                      ** 2 + cos(radians(city_center_lat)) * cos(radians(latitude))
                                                      * sin((radians(longitude) - radians(city_center_lon)) / 2) ** 2)))

    # Predict Cluster for current flat
    def define_cluster(km_model: KMeans, lon: float, lat: float):
        current_cluster = km_model.predict([[lon, lat]])
        return current_cluster

    current_cluster = define_cluster(km_model=kmeans, lon=longitude, lat=latitude)

    print("Current cluster is : ", current_cluster, flush=True)

    # Define current month
    mm_announce = now.month

    # Predict Price using gbr, rf, lgmb if not secondary
    def calculate_price(gbr_model: GradientBoostingRegressor, rf_model: RandomForestRegressor,
                        lgbm_model: LGBMRegressor, secondary: int):
        gbr_predicted_price, lgbm_pedicted_price, rf_predicted_price = 0, 0, 0
        # New
        gbr_predicted_price = np.expm1(gbr_model.predict(
            [[np.log1p(life_sq), np.log1p(to_city_center_distance), mm_announce, rooms, renovation, has_elevator,
              np.log1p(longitude), np.log1p(latitude),
              np.log1p(full_sq),
              np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter,
              rent_year]]))
        print("Gbr predicted price ", gbr_predicted_price, flush=True)

        rf_predicted_price = np.expm1(rf_model.predict(
            [[np.log1p(life_sq), np.log1p(to_city_center_distance), mm_announce, rooms, renovation, has_elevator,
              np.log1p(longitude), np.log1p(latitude),
              np.log1p(full_sq),
              np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter,
              rent_year]]))
        print("rf predicted price ", rf_predicted_price, flush=True)

        lgbm_pedicted_price = np.expm1(lgbm_model.predict(
            [[np.log1p(life_sq), np.log1p(to_city_center_distance), mm_announce, rooms, renovation, has_elevator,
              np.log1p(longitude), np.log1p(latitude),
              np.log1p(full_sq),
              np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter,
              rent_year]]))
        print("Lgbm predicted price ", lgbm_pedicted_price, flush=True)

        # Calculate mean price value based on three algorithms
        price_main = (gbr_predicted_price + lgbm_pedicted_price + rf_predicted_price) / 3
        price = int(price_main[0])
        print("Predicted Price: ", price, flush=True)

        price_meter_sq = price / full_sq
        return price, price_meter_sq

    # Calculate price
    price, price_meter_sq = calculate_price(gbr_model=gbr, rf_model=rf, lgbm_model=lgbm, secondary=secondary)

    ####################
    #                  #
    # TERM CALCULATING #
    #                  #
    ####################

    # Remove price and term outliers (out of 3 sigmas)
    data1 = data[(np.abs(stats.zscore(data.price)) < 3)]
    data2 = data[(np.abs(stats.zscore(data.term)) < 3)]

    data = pd.merge(data1, data2, on=list(data.columns), how='left')

    # Fill NaN if it appears after merging
    data[['term']] = data[['term']].fillna(data[['term']].mean())

    # Create subsample of flats from same cluster (from same "geographical" district)
    df_for_current_label = data[data.clusters == current_cluster[0]]
    print('Shape of current cluster: {0}'.format(df_for_current_label.shape))

    # Check if subsample size have more than 3 samples
    if df_for_current_label.shape[0] < 3:
        answ = {'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1}
        return answ

    # Drop flats which sold more than 600 days
    df_for_current_label = df_for_current_label[df_for_current_label.term <= 600]

    # Check if still enough samples
    if df_for_current_label.shape[0] > 1:

        def LinearReg_Term(data: pd.DataFrame):

            # Handle with negative term values
            # way no1
            data = data._get_numeric_data()  # <- this increase accuracy
            data[data < 0] = 0

            # way no2
            # data['profit'] = data['profit'] + 1 - data['profit'].min()

            # Log Transformation
            data['price'] = np.log1p(data['price_meter_sq'])
            data['profit'] = np.log1p(data['profit'])
            data['term'] = np.log1p(data['term'])

            # Create X and y for Linear Model training
            X = data[['profit', 'price_meter_sq']]
            y = data[['term']].values.ravel()

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            # Create LinearModel and fitting
            reg = LinearRegression().fit(X, y)
            return reg

        def larger(p=0, percent=2):
            larger_prices = []
            for _ in range(15):
                new_p = p + p * percent / 100
                larger_prices.append(new_p)
                percent += 2
            return larger_prices

        # Create list of N larger prices than predicted
        list_of_larger_prices = larger(int(price_meter_sq))

        def smaller(p=0, percent=2):
            smaller_prices = []
            for _ in range(15):
                new_p = p - p * percent / 100
                smaller_prices.append(new_p)
                percent += 2
            return smaller_prices[::-1]

        # Create list of N smaller prices than predicted
        list_of_smaller_prices = smaller(int(price_meter_sq))

        # Create list of N prices: which are larger and smaller than predicted
        list_of_prices = list_of_smaller_prices + list_of_larger_prices
        list_of_prices = [int(i) for i in list_of_prices]

        # Call LinearReg on term
        reg = LinearReg_Term(df_for_current_label)

        def CalculateProfit(l: list):
            list_of_terms = []
            for i in l:
                profit = i / price_meter_sq
                # Calculate term based on profit for each price
                term_on_profit = np.expm1(reg.predict([[np.log1p(profit), np.log1p(i)]]))
                list_of_terms.append(term_on_profit)

            return list_of_terms

        # Calculating term for each price from generated list of prices based on associated profit -> returns list of terms
        list_of_terms = CalculateProfit(list_of_prices)

        # Add links to flats
        # term_links = df_for_current_label.to_dict('record')

        list_of_terms = [int(i.tolist()[0]) for i in list_of_terms]
        print("Terms: ", list_of_terms, flush=True)

        prices = list_of_prices
        prices = [int(i * full_sq) for i in prices]
        print("Prices: ", prices, flush=True)

        # Define function for creating list of dicts
        # x=term, y=price
        # Example: [{'x': int, 'y': int}, {'x': int, 'y': int}]
        def createListOfDicts(terms: list, prices: list):
            list_of_dicts = []
            list_of_dicts += ({'y': int(prc), 'x': int(trm)} for prc, trm in zip(prices, terms))
            return list_of_dicts

        # Create list of dicts
        list_of_dicts = createListOfDicts(list_of_terms, prices)

        # Check if list not empty
        if len(list_of_dicts) <= 2:
            answ = {'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1}
            return answ

        print('list_of_dicts: ', list_of_dicts, flush=True)

        # Define current flat with predicted price and initial term = minimal value from list of term
        current_flat = {'x': min(list_of_terms), 'y': price}

        # Iterate over the list of dicts and try to find suitable term based on prices values
        def find_term(l: list, current_flat: dict):
            term = 0
            if l[-1].get('y') > current_flat.get('y') > l[0].get('y'):
                for i in enumerate(l):
                    print(i)
                    if l[i[0]].get('x') <= current_flat.get('x') < l[i[0] + 1].get('x'):
                        print('!')
                        current_flat['x'] = int((l[i[0]].get('x') + l[i[0] + 1].get('x')) / 2)
                        term = int((l[i[0]].get('x') + l[i[0] + 1].get('x')) / 2)
                        break
                print("New term: ", current_flat, flush=True)
            return current_flat, term

        # Find actual term for current flat price
        if (list_of_dicts[-1].get('y') > current_flat.get('y') > list_of_dicts[0].get('y')) and len(current_flat) > 2:
            current_flat, term = find_term(l=list_of_dicts, current_flat=current_flat)
        else:
            answ = {'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1}
            return answ

        # Leave only unique pairs [{'x': int1, 'y': int2}, {'x': int3, 'y': int4}]
        def select_unique_term_price_pairs(list_of_dicts: list):
            terms = []
            result = []

            for i in range(1, len(list_of_dicts)):
                if (list_of_dicts[i].get('x') != list_of_dicts[i - 1].get('x')) and list_of_dicts[i].get(
                        'x') not in terms:
                    if list_of_dicts[i - 1].get('x') not in terms:
                        result.append(list_of_dicts[i - 1])
                        terms.append(list_of_dicts[i - 1].get('x'))
                    result.append(list_of_dicts[i])
                    terms.append(list_of_dicts[i].get('x'))
            return result

        if len(list_of_dicts) > 2:
            list_of_dicts = select_unique_term_price_pairs(list_of_dicts)
        else:
            answ = {'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1}
            return answ

        # Check if list not empty
        # oops = 1 if len(list_of_dicts) <= 2 else 0
        # list_of_dicts = [{"x": 0, 'y': 0}] if oops else list_of_dicts

        def check(l: list, current_flat):
            for i in l:
                if ((i['x'] == current_flat['x']) | (i['y'] == current_flat['y'])):
                    l.remove(i)
            l.append(current_flat)
            return sorted(l, key=lambda k: k['x'])

        # if not oops:
        print('not oops', flush=True)
        # Check if all dict's keys and values in list are unique
        list_of_dicts = check(list_of_dicts, current_flat)

        # # Update list of dicts with current flat
        # list_of_dicts.insert(0, current_flat)
        #
        # # Finally sort
        # list_of_dicts = sorted(list_of_dicts, key=lambda z: z['x'], reverse=False)
        print('Answer: ', list_of_dicts, flush=True)

        # Check if final list have items in it, otherwise set parameter "OOPS" to 1
        oops = 1 if len(list_of_dicts) <= 2 else 0
        term = 0 if len(list_of_dicts) <= 2 else term
        answ = {'Price': price, 'Duration': term, 'PLot': list_of_dicts, 'FlatsTerm': 0, "OOPS": oops}
        print('answ: ', price, term, list_of_dicts, oops, sep='\n', flush=True)
    else:
        print("Not enough data to plot", flush=True)
        answ = {'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1}

    return answ


# Class for developers page
class Developers_API():

    def __init__(self):
        pass

    # Load data from csv
    def load_data(self, spb_new: str, spb_vtor: str, msc_new: str, msc_vtor: str):

        spb_new = pd.read_csv(spb_new)
        spb_vtor = pd.read_csv(spb_vtor)
        msc_new = pd.read_csv(msc_new)
        msc_vtor = pd.read_csv(msc_vtor)

        # Concatenate new flats + secondary flats
        self.all_spb = pd.concat([spb_new, spb_vtor], ignore_index=True, axis=0)
        self.all_msc = pd.concat([msc_new, msc_vtor], ignore_index=True, axis=0)

        # Just new flats
        # self.msc_new_flats = msc_new

        # Group dataset by full_sq
        self.list_of_squares = [38.0, 42.5, 47.0, 51.5, 56.0, 60.5, 65.0, 69.5, 74.0, 78.5, 83.0, 87.5]

        # Initialize full_sq_group values with zero
        msc_new.loc[:, 'full_sq_group'] = 0

        # Create dictionary: key = group number, value = lower threshold value of full_sq
        # Example: {1: 38.0, 2: 42.5}
        full_sq_grouping_dict = {}

        # Update "full_sq_group" column value according to "full_sq" column value
        for i in range(len(self.list_of_squares)):
            # print(i + 1, self.list_of_squares[i])
            full_sq_grouping_dict[i + 1] = self.list_of_squares[i]
            msc_new.loc[:, 'full_sq_group'] = np.where(msc_new['full_sq'] >= self.list_of_squares[i], i + 1,
                                                       msc_new['full_sq_group'])

        # Auxiliary columns to calculate flat_class: econom, comfort, business, elite
        # 0(econom) if price_meter_sq < 0.6 price_meter_sq's quantile within group
        # 1(comfort) if 0.6 price_meter_sq's quantile within group <= price_meter_sq < 0.9 price_meter_sq's quantile within group
        # 1(business) if 0.9 price_meter_sq's quantile within group <= price_meter_sq < 0.95 price_meter_sq's quantile within group
        # 1(elite) if 0.95 price_meter_sq's quantile within group <= price_meter_sq

        msc_new['price_meter_sq_06q'] = \
            msc_new.groupby(['full_sq_group', 'rooms', 'yyyy_sold', 'mm_sold'])[
                'price_meter_sq'].transform(lambda x: x.quantile(.6))
        msc_new['price_meter_sq_09q'] = \
            msc_new.groupby(['full_sq_group', 'rooms', 'yyyy_sold', 'mm_sold'])[
                'price_meter_sq'].transform(lambda x: x.quantile(.9))
        msc_new['price_meter_sq_095q'] = \
            msc_new.groupby(['full_sq_group', 'rooms', 'yyyy_sold', 'mm_sold'])[
                'price_meter_sq'].transform(lambda x: x.quantile(.95))

        # Set new column value: flat_class. 0 = econom, 1 = comfort, 2 = business, 3 = elite
        msc_new.loc[:, 'housing_class'] = 0  # Set to econom by default
        msc_new.loc[:, 'housing_class'] = np.where(
            msc_new['price_meter_sq'] >= msc_new['price_meter_sq_06q'], 1,
            msc_new['housing_class'])  # Set to comfort
        msc_new.loc[:, 'housing_class'] = np.where(
            msc_new['price_meter_sq'] >= msc_new['price_meter_sq_09q'], 2,
            msc_new['housing_class'])  # Set to business
        msc_new.loc[:, 'housing_class'] = np.where(
            msc_new['price_meter_sq'] >= msc_new['price_meter_sq_095q'], 3,
            msc_new['housing_class'])  # Set to elite

        # Remove price outliers within the groups
        std_data_new_msc = msc_new.groupby(['full_sq_group', 'rooms', 'housing_class', 'yyyy_sold', 'mm_sold'])[
            'price'].transform(
            stats.zscore)

        # Construct a Boolean Series to identify outliers: outliers
        outliers = (std_data_new_msc < -3) | (std_data_new_msc > 3)

        # Drop outliers
        msc_new = msc_new[~outliers]
        print('without outliers: ', msc_new.shape, flush=True)

        # Count number of flats in sub-group
        msc_new['mean_price_group_count'] = \
            msc_new.groupby(['full_sq_group', 'rooms', 'housing_class', 'yyyy_sold', 'mm_sold'])[
                'price'].transform('count')

        msc_new.price = msc_new.price.round()

        print("Loaded data shape: {0}".format(msc_new.shape))

        # Transform dtype
        msc_new['mm_sold'] = msc_new['mm_sold'].astype('int')
        msc_new['mm_announce'] = msc_new['mm_announce'].astype('int')
        msc_new['yyyy_sold'] = msc_new['yyyy_sold'].astype('int')
        msc_new['yyyy_announce'] = msc_new['yyyy_announce'].astype('int')

        self.msc_new = msc_new

        self.full_sq_grouping_dict = full_sq_grouping_dict

    def parse_json(self, data=0):
        if "Storage" in machine:
            with open(data, encoding='utf-8') as read_file:
                data = json.load(read_file)
                # city_id = data["city_id"]
                city_id = 0
                longitude = data['longitude']
                latitude = data['latitude']
                is_rented = data['is_rented']
                rent_year = data['rent_year']
                rent_quarter = data['rent_quarter']
                start_timestamp = data['start_timestamp']
                floors_count = data['floors_count']
                has_elevator = data['elevator']
                parking = data['parking']
                time_to_metro = data['time_to_metro']
                flats = [i for i in data['flats_types']]
                sale_start_month = int(
                    datetime.utcfromtimestamp(data['start_timestamp']).strftime('%m'))  # Get month from unix timestamp
                sale_end_month = int(
                    datetime.utcfromtimestamp(data['end_timestamp']).strftime('%m'))  # Get year from unix timestamp
                sale_start_year = int(datetime.utcfromtimestamp(data['start_timestamp']).strftime('%Y'))
                sale_end_year = int(datetime.utcfromtimestamp(data['end_timestamp']).strftime('%Y'))
                schools_500m, schools_1000m, kindergartens_500m, kindergartens_1000m, clinics_500m, clinics_1000m, shops_500m, shops_1000m = \
                data['schools_500m'], data['schools_1000m'], data['kindergartens_500m'], data['kindergartens_1000m'], \
                data['clinics_500m'], data['clinics_1000m'], \
                data['shops_500m'], data['shops_1000m']

        else:
            # city_id = data["city_id"]
            city_id = 0
            longitude = data['longitude']
            latitude = data['latitude']
            is_rented = data['is_rented']
            rent_year = data['rent_year']
            rent_quarter = data['rent_quarter']
            start_timestamp = data['start_timestamp']
            end_timestamp = data['end_timestamp']
            floors_count = data['floors_count']
            has_elevator = data['elevator']
            parking = data['parking']
            time_to_metro = data['time_to_metro']
            flats = [i for i in data['flats_types']]
            sale_start_month = int(
                datetime.utcfromtimestamp(data['start_timestamp']).strftime('%m'))  # Get month from unix timestamp
            sale_end_month = int(
                datetime.utcfromtimestamp(data['end_timestamp']).strftime('%m'))  # Get year from unix timestamp
            sale_start_year = int(datetime.utcfromtimestamp(data['start_timestamp']).strftime('%Y'))
            sale_end_year = int(datetime.utcfromtimestamp(data['end_timestamp']).strftime('%Y'))
            schools_500m, schools_1000m, kindergartens_500m, kindergartens_1000m, clinics_500m, clinics_1000m, \
            shops_500m, shops_1000m = data['schools_500m'], data['schools_1000m'], data['kindergartens_500m'], \
                                      data['kindergartens_1000m'], data['clinics_500m'], data['clinics_1000m'], \
                                      data['shops_500m'], data['shops_1000m']

        return city_id, longitude, latitude, is_rented, rent_year, rent_quarter, floors_count, has_elevator, parking, \
               time_to_metro, flats, sale_start_month, sale_end_month, sale_start_year, sale_end_year, schools_500m, \
               schools_1000m, kindergartens_500m, kindergartens_1000m, clinics_500m, clinics_1000m, shops_500m, \
               shops_1000m

    def predict(self, flats: list, rent_year: int, longitude: float, latitude: float,
                time_to_metro: int, is_rented: int, rent_quarter: int, has_elevator: int, sale_start_month: int,
                sale_end_month: int, sale_start_year: int, sale_end_year: int, housing_class: int, schools_500m=0, schools_1000m=0,
                kindergartens_500m=0, kindergartens_1000m=0, clinics_500m=0, clinics_1000m=0, shops_500m=0,
                shops_1000m=0, city_id=0):

        now = datetime.now()
        price_model = 0
        kmeans = 0

        # list for dicts of term-type
        list_of_terms = []
        first_graphic = []

        # price changes per month
        prices_changes = {1: 0, 2: 1.01, 3: 1.05, 4: 1.07, 5: 1.09, 6: 1.15, 7: 1.12, 8: 1.13, 9: 1.14, 10: 1.17, 11: 1.19, 12: 1.2}
        revenue_s, revenue_1, revenue_2, revenue_3, revenue_4 = 0, 0, 0, 0, 0

        list_of_months = [i for i in range(sale_start_month, 13)]+[i for i in range(1, sale_end_month+1)]
        print(list_of_months, flush=True)
        # mm_announce = list_of_months[idx]
        yyyy_announce= sale_start_year

        for mm_announce in list_of_months:
            print(mm_announce, flush=True)

            # if (mm_announce in [i for i in range(sale_start_month, 13)]) and yyyy_announce != sale_end_year:\
            if mm_announce == 1:
                yyyy_announce += 1

            # get flats parameters for each flat
            for idx, i in enumerate(flats):

                price_meter_sq = i['price_meter_sq']
                # mm_announce = int(datetime.utcfromtimestamp(i['announce_timestamp']).strftime('%m'))  # Get month from unix
                # yyyy_announce = int(datetime.utcfromtimestamp(i['announce_timestamp']).strftime('%Y'))  # Get year from unix
                # life_sq = i['life_sq']
                rooms = i['rooms']
                # renovation = i['renovation']
                # renovation_type = i['renovation_type']
                # longitude = longitude
                # latitude = latitude
                full_sq = i['full_sq']
                # kitchen_sq = i['kitchen_sq']
                # time_to_metro = time_to_metro
                # floor_last = i['floor_last']
                # floor_first = i['floor_first']
                # windows_view = i['windows_view']
                type = i['type']
                # is_rented = is_rented
                # rent_year = rent_year
                # rent_quarter = rent_quarter
                # has_elevator = has_elevator

                # calculate sales values based on prev year
                # current_cluster = kmeans.predict([[longitude, latitude]])

                # Determine appropriate full_sq_group based on full_sq
                full_sq_group = 0

                for idx, item in enumerate(self.list_of_squares):
                    if full_sq >= item:
                        full_sq_group = idx + 1
                        break 

                ### Sales value for current sub-group
                sales_volume_coeff = 1
                n_years = yyyy_announce - sale_start_year
                if n_years > 0:
                    sales_volume_coeff += 0.05*n_years # per one year volume grows by five percent

                # Calculate number of studios
                sales_value_studio = self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                              mm_sold=mm_announce,
                                                                              rooms=0, housing_class=housing_class) * sales_volume_coeff
                # Calculate number of 1-roomed flats
                sales_value_1 = self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                         mm_sold=mm_announce,
                                                                         rooms=1, housing_class=housing_class) * sales_volume_coeff

                # Calculate number of 2-roomed flats
                sales_value_2 = self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                         mm_sold=mm_announce,
                                                                         rooms=2, housing_class=housing_class) * sales_volume_coeff

                # Calculate number of 3-roomed flats
                sales_value_3 = self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                         mm_sold=mm_announce,
                                                                         rooms=3, housing_class=housing_class) * sales_volume_coeff

                # Calculate number of 4-roomed flats
                sales_value_4 = self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                         mm_sold=mm_announce,
                                                                         rooms=4, housing_class=housing_class) * sales_volume_coeff

                # Calculate revenue for each type

                price = price_meter_sq*full_sq
                price = price*prices_changes[mm_announce]
                if rooms == 0:
                    revenue_s += price*sales_value_studio
                if rooms == 1:
                    revenue_1 += price*sales_value_1
                if rooms == 2:
                    revenue_2 += price*sales_value_2
                if rooms == 3:
                    revenue_3 += price*sales_value_3
                if rooms == 4:
                    revenue_4 += price*sales_value_4


                first_graphic.append(
                    {'month_announce': mm_announce, 'year_announce': yyyy_announce, 's': sales_value_studio,
                     '1': sales_value_1, '2': sales_value_2, '3': sales_value_3, '4': sales_value_4, 'revenue_s':
                         revenue_s, 'revenue_1': revenue_1,
                'revenue_2': revenue_2, 'revenue_3': revenue_3, 'revenue_4': revenue_4})
                # list_of_terms.append(
                #     {'type': type, 'mm_announce': mm_announce, 'yyyy_announce': yyyy_announce,
                #      'full_sq_group': full_sq_group})
        print("List of terms: ", first_graphic, flush=True)
        return first_graphic

    # def train_reg(self, city_id: int, use_trained_models=True):
    #
    #     # define regression model variable
    #     reg = 0
    #
    #     # either use pretrained models
    #     if use_trained_models:
    #         if city_id == 0:
    #             reg = load(TERM_MOSCOW)
    #         elif city_id == 1:
    #             reg = load(TERM_SPB)
    #
    #     # or train regression now
    #     else:
    #         # Define city
    #         data = pd.DataFrame()
    #
    #         if city_id == 1:
    #             data = self.all_spb
    #         elif city_id == 0:
    #             data = self.all_msc
    #
    #         # Log Transformation
    #         # data['profit'] = data['profit'] + 1 - data['profit'].min()
    #         data = data._get_numeric_data()
    #         data[data < 0] = 0
    #
    #         # Remove price and term outliers (out of 3 sigmas)
    #         data = data[((np.abs(stats.zscore(data.price)) < 2.5) & (np.abs(stats.zscore(data.term)) < 2.5))]
    #
    #         data['price_meter_sq'] = np.log1p(data['price_meter_sq'])
    #         data['profit'] = np.log1p(data['profit'])
    #         # data['term'] = np.log1p(data['term'])
    #         # data['mode_price_meter_sq'] = np.log1p(data['mode_price_meter_sq'])
    #         # data['mean_term'] = np.log1p(data['mean_term'])
    #
    #         # Create X and y for Linear Model training
    #         X = data[['price_meter_sq', 'profit', 'mm_announce', 'yyyy_announce', 'rent_year', 'windows_view', 'renovation_type', 'full_sq',
    #                   'is_rented']]
    #         y = data[['term']].values.ravel()
    #
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    #
    #         # Create LinearModel and fitting
    #         # reg = LinearRegression().fit(X_train, y_train)
    #         reg = GradientBoostingRegressor(n_estimators=450, max_depth=5, verbose=1, random_state=42,
    #                                     learning_rate=0.07, max_features='sqrt', min_samples_split=5).fit(X_train, y_train)
    #         preds = reg.predict(X_test)
    #         acc = r2_score(y_test, preds)
    #         print(" Term R2 acc: {0}".format(acc))
    #     return reg

    # Расчёт месяца и года продажи при известном сроке(в днях). Предполгается, что квартиры вымещаются на продажу только в начале месяца.

    # Calculate sales volume for each flat sub-group based on its group, number of rooms, sale month
    def calculate_sales_volume_previos_year(self, rooms: int, full_sq_group: int, mm_sold: int, housing_class: int):

        # Only closed offers
        sale_volume_data = self.msc_new[(
            (self.msc_new['closed'] == True))]

        # Get sales volume
        volume_19 = sale_volume_data[
            ((sale_volume_data.rooms == rooms) & (sale_volume_data.yyyy_sold == 19) & (
                    sale_volume_data.full_sq_group == full_sq_group) & (
                     sale_volume_data.mm_sold == mm_sold)& (
                    sale_volume_data.housing_class == housing_class))].shape[0]

        return volume_19

    # def calculate_sale_month_and_year(self, type: int, term: int, yyyy_announce: int, mm_announce: int):
    #
    #     # Sale time in months
    #     n_months = ceil(term / 30)
    #
    #     sale_year = yyyy_announce
    #     # Define sale months
    #     sale_month = mm_announce + n_months - 1
    #     if sale_month % 12 != 0:
    #         if sale_month > 12 and (sale_month % 12) > 0:
    #             sale_month = sale_month % 12
    #             sale_year += 1
    #         else:
    #             sale_month = sale_month % 12
    #
    #     # print(' mm_announce: {2},\n Sale_year: {1}, \n sale_month: {0}'.format(sale_month, sale_year, mm_announce))
    #     return type, sale_year, sale_month
    #
    # def apply_calculate_sale_month_and_year(self, example: list):
    #     list_calculated_months = []
    #     for i in example:
    #         type, sale_year, sale_month = self.calculate_sale_month_and_year(type=i['type'], term=i['term'],
    #                                                                     yyyy_announce=i['yyyy_announce'],
    #                                                                     mm_announce=i['mm_announce'])
    #         list_calculated_months.append({'type': type, 'sale_year': sale_year, 'sale_month': sale_month})
    #     print(list_calculated_months)
    #     return list_calculated_months
    #
    # def create_dataframe(self, list_to_df: list, sale_start_yyyy: int, sale_end_yyyy: int,
    #                      sale_start_m: int, sale_end_m: int):
    #
    #     #  Convert list of dicts to dataframe
    #     df = pd.DataFrame(list_to_df)
    #
    #     # Calculate each group volume
    #     df = df.groupby(['type', 'sale_year', "sale_month"]).size().reset_index(name='volume')
    #
    #     # Create dummies
    #     dummies = pd.get_dummies(df['type'], prefix='flat_type')
    #
    #     # Get dummies names
    #     dummies_columns = list(dummies.columns)
    #
    #     dummies.values[dummies != 0] = df['volume']
    #     df = pd.concat([df, dummies], axis=1)
    #
    #     # Create new column based on sale_month and sale_year : mm.yy
    #     df.sale_month = df.sale_month.astype('int')
    #     df.sale_year = df.sale_year.astype('int')
    #     df['x_axis_labels'] = df[['sale_month', 'sale_year']].apply(
    #         lambda row: "{0}.{1}".format(str(row.sale_month).zfill(2), str(row.sale_year)[-2:]), axis=1)
    #
    #     # Add fictive data
    #     for year in range(sale_start_yyyy, sale_end_yyyy + 1):
    #         for month in range(1, 13):
    #             if '{0}.{1}'.format(str(month).zfill(2), str(year)[-2:]) not in df.x_axis_labels.tolist():
    #                 df.loc[len(df), 'sale_year':'sale_month'] = (year, month)
    #
    #     df.sale_month = df.sale_month.astype('int')
    #     df.sale_year = df.sale_year.astype('int')
    #     df['x_axis_labels'] = df[['sale_month', 'sale_year']].apply(
    #         lambda row: "{0}.{1}".format(str(row.sale_month).zfill(2), str(row.sale_year)[-2:]), axis=1)
    #
    #     # Create new column based on sale_month and sale_year : mm.yy
    #     df = df.fillna(0)
    #
    #     df[dummies_columns] = df.groupby(['x_axis_labels'])[dummies_columns].transform('sum')
    #
    #     df = df.sort_values(['sale_year', 'sale_month'], ascending=True)
    #
    #     df = df.drop_duplicates('x_axis_labels', keep='first')
    #
    #     new_index = df.x_axis_labels.tolist()
    #     df.index = list(new_index)
    #
    #     df = df.drop(['sale_year', 'sale_month', 'volume', 'type', 'x_axis_labels'], axis=1)
    #
    #
    #     # Plotting
    #     img = df.plot.bar(stacked=True, rot=90, title="Sales forecast", figsize=(15, 8))
    #     img.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #     # plt.xlabel('months')
    #     # plt.ylabel('volume')
    #     # img.savefig('test.png')
    #
    #     # plt.show(block=True)
    #
    #     # print(df.pivot_table(index='months', columns='volume', aggfunc='size'))
    #     # df = df.pivot_table(index='months', columns='volume', aggfunc='size')
    #     # df = df.sort_values(by='month', ascending=True)
    #     # img = df.pivot_table(index='months', columns='volume', aggfunc='size').plot.bar(stacked=True)
    #     # print(list(df.type.unique()))
    #     # img.legend(list(df.type.unique()))
    #     # save img
    #     if "Storage" in machine:
    #         img.figure.savefig('test.png')
    #     else:
    #         img.figure.savefig('/home/realtyai/smartrealty/realty/media/test.png')


def predict_developers_term(longitude: float, latitude: float, floors_count: int,
                            has_elevator: int, parking: int, time_to_metro, flats: list, housing_class: int, is_rented=0, rent_year=0,
                            rent_quarter=0, sale_start_month=0, sale_end_month=0,
                            sale_start_year=0, sale_end_year=0, schools_500m=0, schools_1000m=0, kindergartens_500m=0,
                            kindergartens_1000m=0, clinics_500m=0, clinics_1000m=0, shops_500m=0, shops_1000m=0, city_id=0):
    # Create Class
    devAPI = Developers_API()

    # Load CSV data. Check if it's local machine or remote server
    if "Storage" in machine:
        devAPI.load_data(spb_new=SPB_DATA_NEW, spb_vtor=SPB_DATA_SECONDARY, msc_new='None',
                         msc_vtor='None')

    else:
        devAPI.load_data(spb_new=SPB_DATA_NEW, spb_vtor=SPB_DATA_SECONDARY, msc_new=MOSCOW_DATA_NEW,
                         msc_vtor=MOSCOW_DATA_SECONDARY)

    # Parse json
    # city_id, longitude, latitude, is_rented, rent_year, rent_quarter, floors_count, has_elevator, parking, \
    # time_to_metro, flats, sale_start_month, sale_end_month, sale_start_year, sale_end_year, schools_500m, \
    # schools_1000m, kindergartens_500m, kindergartens_1000m, clinics_500m, clinics_1000m, shops_500m, \
    # shops_1000m = devAPI.parse_json(json_file)

    # Train term reg
    # reg = 0
    # if "Storage" in machine:
    #     reg = load('C:/Storage/DDG/DEVELOPERS/models/dev_term_gbr_spb.joblib')
    # else:
    #     reg = devAPI.train_reg(city_id=city_id)

    # Get answer in format: [{'month_announce': mm_announce, 'year_announce': yyyy_announce, '-1': sales_value_studio,
    #                                   '1': sales_value_1, '2': sales_value_2, '3': sales_value_3, '4': sales_value_4}, {...}]
    answer = devAPI.predict(city_id=city_id, flats=flats, has_elevator=has_elevator,
                            is_rented=is_rented,
                            latitude=latitude, longitude=longitude, rent_quarter=rent_quarter, rent_year=rent_year,
                            time_to_metro=time_to_metro, schools_500m=schools_500m, schools_1000m=schools_1000m,
                            kindergartens_500m=kindergartens_500m, kindergartens_1000m=kindergartens_1000m,
                            clinics_500m=clinics_500m,
                            clinics_1000m=clinics_1000m, shops_500m=shops_500m, shops_1000m=shops_1000m,
                            housing_class=housing_class, sale_end_month=sale_end_month, sale_end_year=sale_end_year,
                            sale_start_month=sale_start_month, sale_start_year=sale_start_year)

    return answer
