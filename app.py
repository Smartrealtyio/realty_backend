from flask import Flask, request, jsonify, render_template
from scipy import stats
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LogisticRegression
import psycopg2
import settings_local as SETTINGS
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
#from catboost import Pool, CatBoostRegressor
from joblib import dump, load
import math as m
import math
from datetime import datetime
import requests
import json
import pandas as pd
import statistics
import numpy as np
import math

PATH_PRICE_GBR_MOSCOW_VTOR = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_GBR.joblib'
PATH_PRICE_RF_MOSCOW_VTOR = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_RF.joblib'
PATH_PRICE_LGBM_MOSCOW_VTOR = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_LGBM.joblib'
PATH_PRICE_GBR_SPB_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_GBR.joblib'
PATH_PRICE_RF_SPB_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_RF.joblib'
PATH_PRICE_LGBM_SPB_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_LGBM.joblib'

PATH_PRICE_GBR_MOSCOW_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_GBR.joblib'
PATH_PRICE_RF_MOSCOW_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_RF.joblib'
PATH_PRICE_LGBM_MOSCOW_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_LGBM.joblib'
PATH_PRICE_GBR_SPB_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_GBR.joblib'
PATH_PRICE_RF_SPB_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_RF.joblib'
PATH_PRICE_LGBM_SPB_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_LGBM.joblib'
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/api/mean/', methods=['GET'])
def mean():
    full_sq_from = float(request.args.get('full_sq_from'))
    full_sq_to = float(request.args.get('full_sq_to'))
    latitude_from = float(request.args.get('latitude_from'))
    latitude_to = float(request.args.get('latitude_to'))
    longitude_from = float(request.args.get('longitude_from'))
    longitude_to = float(request.args.get('longitude_to'))
    rooms = float(request.args.get('rooms'))
    price_from = float(request.args.get('price_from')) if request.args.get('price_from') is not None else None
    price_to = float(request.args.get('price_to')) if request.args.get('price_to') is not None else None
    building_type_str = float(request.args.get('building_type_str')) if request.args.get(
        'building_type_str') is not None else None
    kitchen_sq = float(request.args.get('kitchen_sq')) if request.args.get('kitchen_sq') is not None else None
    life_sq = float(request.args.get('life_sq')) if request.args.get('life_sq') is not None else None
    renovation = float(request.args.get('renovation')) if request.args.get('renovation') is not None else None
    has_elevator = float(request.args.get('elevator')) if request.args.get('elevator') is not None else None
    floor_first = float(request.args.get('floor_first')) if request.args.get('floor_first') is not None else None
    floor_last = float(request.args.get('floor_last')) if request.args.get('floor_last') is not None else None
    time_to_metro = float(request.args.get('time_to_metro')) if request.args.get('time_to_metro') is not None else None
    page = int(request.args.get('page')) if request.args.get('page') is not None else 1
    sort_type = int(request.args.get('sort_type')) if request.args.get('sort_type') is not None else 0
    city_id = int(request.args.get('city_id')) if request.args.get('city_id') is not None else 0

    print(latitude_from, latitude_to, longitude_from, longitude_to, flush=True)

    # Initialize DF
    data_offers = pd.DataFrame()

    # 0 = Moscow, 1 = Spb
    if city_id == 0:
        data_offers = pd.read_csv(SETTINGS.DATA_MOSCOW + '/MOSCOW.csv')
        data_offers = data_offers[data_offers.flat_type == 'SECONDARY']
        gbr = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        rf = load(PATH_PRICE_RF_MOSCOW_VTOR)
        lgbm = load(PATH_PRICE_LGBM_MOSCOW_VTOR)
    elif city_id == 1:
        data_offers = pd.read_csv(SETTINGS.DATA_SPB + '/SPB.csv')
        data_offers = data_offers[data_offers.flat_type == 'SECONDARY']
        gbr = load(PATH_PRICE_GBR_SPB_VTOR)
        rf = load(PATH_PRICE_RF_SPB_VTOR)
        lgbm = load(PATH_PRICE_LGBM_SPB_VTOR)

    filter = (((data_offers.full_sq >= full_sq_from)&(data_offers.full_sq <= full_sq_to))&(data_offers.rooms == rooms) &
              ((data_offers.latitude >= latitude_from) & (data_offers.latitude <= latitude_to))
              & ((data_offers.longitude >= longitude_from) & (data_offers.longitude <= longitude_to)))
    data_offers = data_offers[filter]
    print("data offers: ", data_offers.shape, flush=True)
    # Uses only open offers
    data_offers = data_offers[data_offers['closed'] == False]

    print('ds: ', data_offers.head(), flush=True)

    if time_to_metro != None:
        data_offers = data_offers[(data_offers.time_to_metro <= time_to_metro)]
    if rooms != None:
        data_offers = data_offers[data_offers.rooms == rooms]
    if building_type_str != None:
        data_offers = data_offers[data_offers.building_type_str == building_type_str]
    if kitchen_sq != None:
        data_offers = data_offers[(data_offers.kitchen_sq >= kitchen_sq - 1) & (data_offers.kitchen_sq <= kitchen_sq + 1)]
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



    # PRICE




    # Print GradientBoosting Regression features importance
    # feat_imp = pd.Series(gbr.feature_importances_, X1.columns).sort_values(ascending=False)
    # print(feat_imp)


    data_offers['pred_price'] = data_offers[['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                                          'time_to_metro', 'floor_last', 'floor_first', 'clusters']].apply(
        lambda row:
        int((np.expm1(rf.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator, np.log1p(row.longitude),
                                    np.log1p(row.latitude), np.log1p(row.latitude),
                                       np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_first, row.floor_last, row.clusters]]))+np.expm1(lgbm.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator, np.log1p(row.longitude),
                                    np.log1p(row.latitude), np.log1p(row.latitude),
                                       np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_first, row.floor_last, row.clusters]])))[0]/2), axis=1)


    print('Profitable offers using price prediction model: ', data_offers.shape[0])


    data_offers['profit'] = data_offers[['pred_price', 'price']].apply(lambda row: ((row.pred_price*100/row.price)-100), axis=1)
    data_offers = data_offers[(data_offers.profit >= 5)]
    data_offers = data_offers.sort_values(by=['profit'], ascending=False)
    print(data_offers[['pred_price', "price"]].head())


    flats = data_offers.to_dict('record')


    flats_count = len(flats)
    flats_page_count = 10
    # max_page = math.ceil(len(flats) / flats_page_count)
    max_page = 1
    page = page if page <= max_page else 1
    '''
    if sort_type == 0:
        flats = sorted(flats, key=lambda x: x['price'])[(page - 1) * flats_page_count:page * flats_page_count]
    else:
        flats = sorted(flats, key=lambda x: x['price'])[(page - 1) * flats_page_count:page * flats_page_count]
    '''
    conn = psycopg2.connect(host=SETTINGS.host, dbname=SETTINGS.name, user=SETTINGS.user, password=SETTINGS.password)
    cur = conn.cursor()
    for flat in flats:
        # print(flat.keys(), flush=True)
        cur.execute("select metro_id, time_to_metro from time_metro_buildings where building_id=%s",
                    (flat['building_id'],))
        metros_info = cur.fetchall()
        flat['metros'] = []
        for metro in metros_info:
            cur.execute("select name from metros where id=%s", (metro[0],))
            flat['metros'].append({'station': cur.fetchone()[0], 'time_to_metro': metro[1]})

        if flat['resource_id'] == 0:
            flat['link'] = 'https://realty.yandex.ru/offer/' + str(flat['offer_id'])
        else:
            flat['link'] = 'https://www.cian.ru/sale/flat/' + str(flat['offer_id'])

        cur.execute("select address from buildings where id=%s",
                    (flat['building_id'],))
        flat['address'] = cur.fetchone()[0]

        # print(flat['image'], flush=True)

        if type(flat['image']) != str:
            flat['image'] = None
        del flat['offer_id']
        del flat['building_id']
        del flat['time_to_metro']
        # print(flat, flush=True)

    conn.close()

    print('flats', len(flats), flush=True)

    # if math.isnan(mean_price):
    #     mean_price = None
    return jsonify({'flats': flats, 'page': page, 'max_page': max_page, 'count': flats_count})


@app.route('/map')
def map():
    longitude = float(request.args.get('lng'))
    rooms = int(request.args.get('rooms'))
    latitude = float(request.args.get('lat'))
    full_sq = float(request.args.get('full_sq'))
    kitchen_sq = float(request.args.get('kitchen_sq'))
    life_sq = float(request.args.get('life_sq'))
    renovation = int(request.args.get('renovation'))
    secondary = int(request.args.get('secondary'))
    has_elevator = int(request.args.get('elevator'))
    floor_first = int(request.args.get('floor_first'))
    floor_last = int(request.args.get('floor_last'))
    time_to_metro = int(request.args.get('time_to_metro'))

    city_id = int(request.args.get('city_id')) if request.args.get('city_id') is not None else 0

    # initialize dataframe
    data = pd.DataFrame()
    kmeans = 0
    gbr = 0
    lgbm = 0
    rf = 0
    print("PArams: ", city_id, secondary, flush=True)

    # 0 = Moscow, 1 = Spb
    # Москва новостройки
    if city_id == 0 and secondary ==0:
        # Load data Moscow New flats
        data = pd.read_csv(SETTINGS.DATA_MOSCOW + '/MOSCOW_NEW_FLATS.csv')

        # Load KMean Clustering model
        kmeans = load(SETTINGS.MODEL_MOSCOW + '/KMEAN_CLUSTERING_MOSCOW_NEW_FLAT.joblib')

        # Load Price Models Moscow Secondary
        gbr = load(PATH_PRICE_GBR_MOSCOW_NEW)
        rf = load(PATH_PRICE_RF_MOSCOW_NEW)
        lgbm = load(PATH_PRICE_LGBM_MOSCOW_NEW)

    # Москва вторичка
    elif city_id == 0 and secondary == 1:
        # Load data Moscow secondary
        data = pd.read_csv(SETTINGS.DATA_MOSCOW + '/MOSCOW_VTOR.csv')

        # Load KMean Clustering model
        kmeans = load(SETTINGS.MODEL_MOSCOW + '/KMEAN_CLUSTERING_MOSCOW_VTOR.joblib')

        # Load Price Models Moscow Secondary
        gbr = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        rf = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        lgbm = load(PATH_PRICE_GBR_MOSCOW_VTOR)

    # Санкт-Петербург новостройки
    elif city_id == 1 and secondary == 0:
        # Load data SPb New Flats
        data = pd.read_csv(SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv')

        # Load KMean Clustering model
        kmeans = load(SETTINGS.MODEL_SPB + 'KMEAN_CLUSTERING_NEW_FLAT_SPB.joblib')

        # Load Price Models Spb Secondary
        gbr = load(PATH_PRICE_GBR_SPB_NEW)
        rf = load(PATH_PRICE_RF_SPB_NEW)
        lgbm = load(PATH_PRICE_LGBM_SPB_NEW)

    # Санкт-Петербург вторичка
    elif city_id == 1 and secondary == 1:
        data = pd.read_csv(SETTINGS.DATA_SPB + '/SPB_VTOR.csv')
        # Load KMean Clustering model
        kmeans = load(SETTINGS.MODEL_SPB + '/KMEAN_CLUSTERING_SPB_VTOR.joblib')

        # Load Price Models Spb Secondary
        gbr = load(PATH_PRICE_GBR_SPB_VTOR)
        rf = load(PATH_PRICE_RF_SPB_VTOR)
        lgbm = load(PATH_PRICE_LGBM_SPB_VTOR)


    print("Initial shape: ", data.shape, flush=True)



    # Predict Cluster for current flat
    current_claster = kmeans.predict([[longitude, latitude]])
    print("Current label: ", current_claster, flush=True)



    # Reducing skew in data using LogTransformation
    # longitude = np.log1p(longitude)
    # latitude = np.log1p(latitude)
    # full_sq = np.log1p(full_sq)
    # kitchen_sq = np.log1p(kitchen_sq)
    # life_sq = np.log1p(life_sq)
    # rooms = np.log1p(rooms)

    # PRICE PREDICTION


    # price_cat_pred = (np.expm1(gbr.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude), np.log1p(full_sq),
    #                                    np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_claster]]))+np.expm1(rf.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude), np.log1p(full_sq),
    #                                    np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_claster]]))+np.expm1(lgbm.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude), np.log1p(full_sq),
    #                                    np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_claster]]))) / 3

    gbr_predicted_price = np.expm1(gbr.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude),
                           np.log1p(full_sq),
                           np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_claster]]))

    print("Gbr predicted price: ", gbr_predicted_price, flush=True)
    # rf_predicted_price = np.expm1(rf.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude),
    #                       np.log1p(full_sq),
    #                       np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_claster]]))
    # print("Rf predicted price: ", rf_predicted_price, flush=True)

    lgbm_pedicted_price = np.expm1(lgbm.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude), np.log1p(full_sq),
                                       np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_claster]]))
    print("Lgbm predicted price: ", lgbm_pedicted_price, flush=True)

    price_main = (gbr_predicted_price++lgbm_pedicted_price)/ 2

    print("Stacking gbr_lgbm: ", price_main, flush=True)



    # Count mean of Cat and GBR algorithms prediction
    # price = (price_gbr_pred+price_cat_pred)/2
    price = price_main
    price = int(price[0])
    print("Predicted Price: ", price, flush=True)

    price_meter_sq = price / full_sq





    # TERm

    # Create subsample of flats with same cluster label value (from same "geographical" district)
    df_for_current_label = data[data.clusters == current_claster[0]]

    from math import sqrt
    n = int(sqrt(df_for_current_label.shape[0]))

    # Create SUB Classes
    kmeans = KMeans(n_clusters=n, random_state=42).fit(
        df_for_current_label[['full_sq', 'life_sq', 'kitchen_sq', 'clusters', 'time_to_metro', 'longitude', 'latitude', 'renovation', 'nums_of_changing']])

    labels = kmeans.labels_
    df_for_current_label['SUB_cluster'] = labels
    print(df_for_current_label.SUB_cluster.unique(), flush=True)
    df_for_current_label['num_of_flats_in_SUB_cluster'] = df_for_current_label.groupby(['SUB_cluster'])["SUB_cluster"].transform("count")

    # Drop Price and Term Outliers using Z-Score
    df_for_current_label = df_for_current_label[df_for_current_label.price.between(df_for_current_label.term.quantile(.1), df_for_current_label.price.quantile(.9))]


    if df_for_current_label.shape[0] < 2:
        answ = jsonify({'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1})
        return answ

    df_for_current_label['pred_price'] = df_for_current_label[
        ['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
         'time_to_metro', 'floor_last', 'floor_first', 'clusters']].apply(
        lambda row:
        int((gbr.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator,
                                 np.log1p(row.longitude), np.log1p(row.latitude), np.log1p(row.full_sq),
                                 np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_last, row.floor_first,
                                 row.clusters]]) +
             (lgbm.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator,
                                   np.log1p(row.longitude), np.log1p(row.latitude), np.log1p(row.full_sq),
                                   np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_last, row.floor_first,
                                   row.clusters]])))[0] / 2), axis=1)
    df_for_current_label['profit'] = df_for_current_label[['pred_price', 'price']].apply(
        lambda row: ((row.pred_price / row.price)), axis=1)


    df_for_current_label = df_for_current_label[df_for_current_label.term <= 600]
    if df_for_current_label.shape[0] > 1:


        term = 0
        X = df_for_current_label[['profit', 'price']]
        y = df_for_current_label[['term']].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        reg = LinearRegression().fit(X_train, y_train)
        # logreg = LogisticRegression()
        # names = ['renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'full_sq', 'kitchen_sq',
        #      'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y',
        #      'price_meter_sq', 'profit']
        # print("Features importances GBR Term: ", sorted(zip(GBR_TERM_NEW.feature_importances_.tolist(), names)), flush=True)
        # print("Features importances Cat Term: ",sorted(zip(CAT_TERM_NEW.feature_importances_.tolist(), names)), flush=True)

        # Create list of N prices: which are larger and smaller than predicted
        def larger(p=0):
            larger_prices = []
            percent = 2
            for _ in range(15):
                new_p = p + p * percent / 100
                larger_prices.append(new_p)
                percent += 2
            return larger_prices
        list_of_larger_prices = larger(price)

        def smaller(p=0):
            smaller_prices = []
            percent = 2
            for _ in range(15):
                new_p = p - p * percent / 100
                smaller_prices.append(new_p)
                percent += 2
            return smaller_prices[::-1]
        list_of_smaller_prices = smaller(price)


        list_of_prices = list_of_smaller_prices+list_of_larger_prices



        def fn(l: list):
            list_of_terms = []
            for i in l:
                profit = np.log1p(price)/np.log1p(i)
                term_profit = reg.predict([[profit, np.log1p(price)]])


                print("Predicted term: ", term_profit, flush=True)
                list_of_terms.append(term_profit)
            return list_of_terms
        list_of_terms = fn(list_of_prices)





        # Count profit for different prices

        # Add links to flats
        term_links = df_for_current_label.to_dict('record')



        list_of_terms = [i.tolist()[0] for i in list_of_terms]
        print("Terms: ", list_of_terms, flush=True)


        prices = list_of_prices
        print("Prices: ", prices, flush=True)


        # Create list of dictionaries
        a = []
        a += ({'x': int(trm), 'y': prc} for trm, prc in zip(list_of_terms, prices))


        # Sort list by term
        a = [i for i in a if 0 < i.get('x') <600]
        a = sorted(a, key=lambda z: z['x'], reverse=False)
        print("First sort by term: ", a, flush=True)

        def drop_duplicates_term(l: list):
            seen = set()
            new_l = []
            for item in l:
                if item.get('x') not in seen:
                    seen.add(item.get('x'))
                    new_l.append(item)
            return new_l

        new_list_of_dicts = drop_duplicates_term(a)
        print("After drop term duplicates: ", new_list_of_dicts, flush=True)

        b = {'x': int(term), 'y': int(price)}
        print("Predicted raw term, and exact price: ", b, flush=True)

        def drop_duplicates_price(l: list):
            seen_prices = set()
            new_list_of_prices = []
            for item in l:
                if item.get('y') not in seen_prices:
                    seen_prices.add(item.get('y'))
                    new_list_of_prices.append(item)
            return new_list_of_prices
        if len(new_list_of_dicts) > 1:
            if (new_list_of_dicts[-1].get('y')>price > new_list_of_dicts[0].get('y')):
                for i in enumerate(new_list_of_dicts):
                    if new_list_of_dicts[i[0]].get('y') < b.get('y') < new_list_of_dicts[i[0] + 1].get('y'):
                        b['x'] = int((new_list_of_dicts[i[0]].get('x')+new_list_of_dicts[i[0] + 1].get('x'))/2)
                        term = int((new_list_of_dicts[i[0]].get('x')+new_list_of_dicts[i[0] + 1].get('x'))/2)
                        break
                print("New term: ", b, flush=True)
            elif price > new_list_of_dicts[-1].get('y'):
                new_list_of_dicts = [b]

        new_a = drop_duplicates_price(new_list_of_dicts)
        print('Drop price duplicates:', new_a, flush=True)




        new_a.insert(0, b)

        # new_a = drop_duplicates(new_a)
        # new_a += [b]
        new_a = sorted(new_a, key=lambda z: z['x'], reverse=False)
        # new_a = drop_duplicat(new_a)
        print("Sorted finally : ", new_a, flush=True)



        oops = 1 if len(new_a)<=1 else 0
        term = 0 if len(new_a)<=1 else term





        answ = jsonify({'Price': price, 'Duration': term, 'PLot': new_a, 'FlatsTerm': term_links, "OOPS": oops})
    else:
        print("Not enough data to plot", flush=True)
        answ = jsonify({'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS":1})
    return answ



if __name__ == '__main__':
    app.run()
