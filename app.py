from flask import Flask, request, jsonify, render_template
from scipy import stats
import xgboost
import psycopg2
import settings_local as SETTINGS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load
import math as m
import math
from datetime import datetime
import requests
import json
import pandas as pd

import numpy as np
import math

PATH_TO_PRICE_MODEL = SETTINGS.MODEL + '/PriceModel.joblib'
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
    has_elevator = float(request.args.get('has_elevator')) if request.args.get('has_elevator') is not None else None
    floor_first = float(request.args.get('floor_first')) if request.args.get('floor_first') is not None else None
    floor_last = float(request.args.get('floor_last')) if request.args.get('floor_last') is not None else None
    time_to_metro = float(request.args.get('time_to_metro')) if request.args.get('time_to_metro') is not None else None
    page = int(request.args.get('page')) if request.args.get('page') is not None else 1
    sort_type = int(request.args.get('sort_type')) if request.args.get('sort_type') is not None else 0

    print(latitude_from, latitude_to, longitude_from, longitude_to, flush=True)
    '''
    DATA_OUTLIERS = SETTINGS.DATA + '/COORDINATES_OUTLIERS.csv'
    MODEL_OUTLIERS = SETTINGS.MODEL + '/models.joblib'

    data = pd.read_csv(DATA_OUTLIERS)
    data = data[['price_meter_sq', 'full_sq']]
    data = data[data.price_meter_sq < data.price_meter_sq.quantile(0.2)]

    print('data', data.shape, flush=True)

    model = load(MODEL_OUTLIERS)
    # outliers = model.predict(data)
    outliers_it = data[model.predict(data) == -1]
    print('Outliers: ', outliers_it.shape[0], flush=True)
    outliers_it['flat_id'] = outliers_it.index
    new_data = pd.read_csv(DATA_OUTLIERS)
    print(new_data.shape)
    new_data['flat_id'] = new_data.index
    #ds = pd.merge(new_data, outliers_it, left_on="flat_id", right_on="flat_id", suffixes=['', 'double'])
    #ds = ds.drop(['flat_id', 'full_sqdouble', 'price_meter_sqdouble'], axis=1)
    full_data_outliers = new_data[new_data.flat_id.isin(outliers_it.flat_id)]
    sklearn_score_anomalies = model.score_samples(full_data_outliers[['price_meter_sq', 'full_sq']])
    original_paper_score = np.array([(((-1 * s + 0.5) - 1) * 100) for s in sklearn_score_anomalies])
    print(original_paper_score)
    df_f = pd.DataFrame({'profit': original_paper_score}, index=full_data_outliers.index)
    print(df_f.head())
    new_df = pd.concat([full_data_outliers, df_f], axis=1)
    new_df = new_df.sort_values(by=['profit'], ascending=False)


    print('ds shape', new_df.shape, flush=True)
    
    '''


    data = pd.read_csv(SETTINGS.DATA + '/COORDINATES_OUTLIERS.csv')
    new_df = data
    filter = (((new_df.full_sq >= full_sq_from)&(new_df.full_sq <= full_sq_to))&(new_df.rooms == rooms) &
              ((new_df.latitude >= latitude_from) & (new_df.latitude <= latitude_to))
              & ((new_df.longitude >= longitude_from) & (new_df.longitude <= longitude_to)))
    new_df = new_df[filter]

    print('ds', new_df.shape, flush=True)

    if time_to_metro != None:
        new_df = new_df[(new_df.time_to_metro <= time_to_metro)]
    if rooms != None:
        new_df = new_df[new_df.rooms == rooms]
    if building_type_str != None:
        new_df = new_df[new_df.building_type_str == building_type_str]
    if kitchen_sq != None:
        new_df = new_df[(new_df.kitchen_sq >= kitchen_sq - 1) & (new_df.kitchen_sq <= kitchen_sq + 1)]
    if life_sq != None:
        new_df = new_df[(new_df.life_sq >= life_sq - 5) & (new_df.life_sq <= life_sq + 5)]
    if renovation != None:
        new_df = new_df[new_df.renovation == renovation]
    if has_elevator != None:
        new_df = new_df[new_df.has_elevator == has_elevator]
    if floor_first != None:
        new_df = new_df[new_df.floor_first == 0]
    if floor_last != None:
        new_df = new_df[new_df.floor_last == 0]
    if price_from != None:
        new_df = new_df[new_df.price >= price_from]
    if price_to != None:
        new_df = new_df[new_df.price <= price_to]

    # PRICE
    '''

    X1 = new_df[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                 'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']]
    new_df["price"] = np.log1p(new_df["price"])
    y1 = new_df[['price']].values.ravel()
    print(X1.shape, y1.shape)

    clf = GradientBoostingRegressor(n_estimators=350, max_depth=4, verbose=10)
    clf.fit(X1, y1)
    '''
    clf = load(PATH_TO_PRICE_MODEL)

    X1 = new_df[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
               'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']]
    feat_imp = pd.Series(clf.feature_importances_, X1.columns).sort_values(ascending=False)
    print(feat_imp)
    #new_df["price"] = np.log1p(new_df["price"])
    new_df['pred_price'] = new_df[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                                   'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']].apply(
        lambda row:
        int(np.expm1(clf.predict([[row.renovation, row.has_elevator, row.longitude, row.latitude, row.full_sq,
                                   row.kitchen_sq, row.is_apartment, row.time_to_metro, row.floor_last,
                                   row.floor_first, row.X, row.Y]]))[0]), axis=1)
    #new_df["price"] = np.expm1(new_df["price"])

    # Check Profit Offers using Outliers algorithm detection
    outliers_alg = IsolationForest(contamination=0.2)
    outliers_alg.fit(new_df[['longitude', 'latitude', 'price', 'full_sq', 'X', 'Y']])
    outliers_it = new_df[outliers_alg.predict(new_df[['longitude', 'latitude', 'price', 'full_sq', 'X', 'Y']]) == -1]
    print('Outliers: ', outliers_it.shape[0], flush=True)
    outliers_it['flat_id'] = outliers_it.index


    new_df = new_df[new_df.price < new_df.pred_price]
    new_df['flat_id'] = new_df.index
    print('Profitable offers using price prediction model: ', new_df.shape[0])

    new_df = new_df[new_df.flat_id.isin(outliers_it.flat_id)]
    print('After concat: ', new_df.shape[0])
    new_df['profit'] = new_df[['pred_price', 'price']].apply(lambda row: ((row.pred_price*100/row.price)-100), axis=1)
    new_df = new_df.sort_values(by=['profit'], ascending=False)
    print(new_df[['pred_price', "price"]].head())



    # price = np.expm1(pred)
    # price = int(price[0])
    # print("Predicted Price: ", price)
    flats = new_df.to_dict('record')


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
                    (flat['id_building'],))
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
                    (flat['id_building'],))
        flat['address'] = cur.fetchone()[0]

        # print(flat['image'], flush=True)

        if type(flat['image']) != str:
            flat['image'] = None
        del flat['offer_id']
        del flat['id_building']
        del flat['time_to_metro']
        # print(flat, flush=True)

    conn.close()

    print('flats', len(flats), flush=True)

    # if math.isnan(mean_price):
    #     mean_price = None
    return jsonify({'flats': flats, 'page': page, 'max_page': max_page, 'count': flats_count})


@app.route('/map')
def map():
    # building_type_str = request.args.get('building_type_str')
    longitude = float(request.args.get('lng'))
    latitude = float(request.args.get('lat'))
    full_sq = float(request.args.get('full_sq'))
    kitchen_sq = float(request.args.get('kitchen_sq'))
    # life_sq = request.args.get('life_sq')
    is_apartment = int(request.args.get('is_apartment'))
    renovation = int(request.args.get('renovation'))
    has_elevator = int(request.args.get('has_elevator'))
    floor_first = int(request.args.get('floor_first'))
    floor_last = int(request.args.get('floor_last'))
    time_to_metro = int(request.args.get('time_to_metro'))
    X = (m.cos(latitude) * m.cos(longitude))
    Y = (m.cos(latitude) * m.sin(longitude))

    list_of_requested_params_price = [renovation, has_elevator, longitude, latitude, full_sq, kitchen_sq,
                                      is_apartment, time_to_metro, floor_last, floor_first, X, Y]



    # Data
    data = pd.read_csv(SETTINGS.DATA + '/COORDINATES_Pred_Term.csv')
    print("Initial shape: ", data.shape)
    print("Offers with renovation: ", data[data.renovation==1].shape)

    kmeans = load(SETTINGS.MODEL + '/KMEAN_CLUSTERIZATION.joblib')
    current_label = kmeans.predict([[longitude, latitude]])
    print("Current label: ", current_label)

    df_for_current_label = data[data.clusters == current_label[0]]
    df_for_current_label = df_for_current_label[((df_for_current_label.full_sq >= full_sq-2)&(df_for_current_label.full_sq <= full_sq+2))]
    def remove_outlier(df_in, col_name):
        q1 = df_in[col_name].quantile(0.15)
        q3 = df_in[col_name].quantile(0.85)
        # iqr = q3 - q1  # Interquartile range
        # fence_low = q1 - 1.5 * iqr
        # fence_high = q3 + 1.5 * iqr
        df_out = df_in.loc[(df_in[col_name] > q1) & (df_in[col_name] < q3)]
        return df_out


    df_for_current_label = df_for_current_label[(np.abs(stats.zscore(df_for_current_label.price)) < 2.8)]
    # df_for_current_label = remove_outlier(df_for_current_label, 'price')

    X1 = df_for_current_label[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                               'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']]

    # sc = StandardScaler()
    # X1 = sc.fit_transform(X1)

    df_for_current_label["price"] = np.log1p(df_for_current_label["price"])
    y1 = df_for_current_label[['price']].values.ravel()

    # PRICE
    # GBR
    gbr = GradientBoostingRegressor(n_estimators=350, max_depth=4, verbose=10, max_features=5)
    print(X1.shape, y1.shape)
    gbr.fit(X1, y1)
    pred_gbr = gbr.predict([list_of_requested_params_price])
    price_gbr = np.expm1(pred_gbr)

    # XGBoost
    X1_xgb = X1.values
    y1_xgb = df_for_current_label[['price']].values
    best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                                          gamma=0,
                                          learning_rate=0.07,
                                          max_depth=3,
                                          min_child_weight=1,
                                          n_estimators=10000,
                                          reg_alpha=0.75,
                                          reg_lambda=0.45,
                                          subsample=0.6,
                                          seed=42)

    best_xgb_model.fit(X1, y1)
    predict_xgb = np.expm1(best_xgb_model.predict(np.array(list_of_requested_params_price).reshape((1,-1))))

    df_for_current_label["price"] = np.expm1(df_for_current_label["price"])
    price = (pred_gbr+predict_xgb)/2
    price = int(price[0])
    print("Predicted Price: ", price)
    price_meter_sq = price / full_sq
    #list_of_requested_params_term = [renovation, has_elevator, longitude, latitude, price, full_sq, kitchen_sq,
    #                                  is_apartment, time_to_metro,
    #                                 floor_last, floor_first, X, Y]
    term = 0



    '''
    def remove_outlier(df_in, col_name):
        q1 = df_in[col_name].quantile(0.20)
        q3 = df_in[col_name].quantile(0.80)
        # iqr = q3 - q1  # Interquartile range
        # fence_low = q1 - 1.5 * iqr
        # fence_high = q3 + 1.5 * iqr
        df_out = df_in.loc[(df_in[col_name] > q1) & (df_in[col_name] < q3)]
        return df_out
    '''
    df = df_for_current_label[(np.abs(stats.zscore(df_for_current_label.price)) < 2.8)]
    # df = remove_outlier(df_for_current_label, 'price')
    ds = df_for_current_label[(np.abs(stats.zscore(df_for_current_label.term)) < 2.8)]
    # ds = remove_outlier(df_for_current_label, 'term')
    clean_data = pd.merge(df, ds, on=list(ds.columns))
    df_for_current_label = clean_data

    sc = StandardScaler()

    # TERM
    df_for_current_label = df_for_current_label[((df_for_current_label.kitchen_sq <= kitchen_sq+1)&
                                                 (df_for_current_label.kitchen_sq >= kitchen_sq-1))]
    df_for_current_label = df_for_current_label[df_for_current_label.term <= 800]

    reg = GradientBoostingRegressor(n_estimators=350, max_depth=8, max_features=8)
    reg.fit(df_for_current_label[['renovation', 'has_elevator', 'longitude', 'latitude','price', 'full_sq', 'kitchen_sq',
                               'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']], df_for_current_label[['term']])


    term = reg.predict([[renovation, has_elevator, longitude, latitude, price, full_sq, kitchen_sq,
                         is_apartment, time_to_metro, floor_last, floor_first, X, Y]])
    term = int(term.item(0))
    print(term)


    #df_for_current_label = remove_outlier(df_for_current_label, 'price')
    #print("After removing price_outliers: ", df_for_current_label.shape)


    # Add links to flats
    term_links = df_for_current_label.to_dict('record')
    for i in term_links:
        if i['resource_id'] == 0:
            i['link'] = 'https://realty.yandex.ru/offer/' + str(i['offer_id'])
        else:
            i['link'] = 'https://www.cian.ru/sale/flat/' + str(i['offer_id'])

    '''
    filter1 = (((data.full_sq <= full_sq + 3) & (data.full_sq >= full_sq - 3)) & (
            (data.longitude >= longitude - 0.05) & (data.longitude <= longitude + 0.05) &
            (data.latitude >= latitude - 0.05) & (data.latitude <= latitude + 0.05)) &
               (data.term <= term) & ((data.price_meter_sq <= price_meter_sq + 20000) & (
                           data.price_meter_sq >= price_meter_sq - 20000)) & ((data.time_to_metro >= time_to_metro - 2) & (
                        data.time_to_metro <= time_to_metro + 2)))

    ds = data[filter1]
    print(ds.shape)
    '''
    df_for_current_label = df_for_current_label[df_for_current_label.term <= term+100]

    # df_for_current_label = df_for_current_label[((df_for_current_label.price <= price+1500000)& (df_for_current_label.price >= price-1500000))]
    x = df_for_current_label.term
    x = x.tolist()
    x += [term]

    y = df_for_current_label.price
    y = y.tolist()
    y += [price]


    a = []
    a += ({'x': x, 'y': y} for x, y in zip(x, y))
    # Sort Dictionary
    a = sorted(a, key=lambda i: i['x'], reverse=False)
    print(a)

    return jsonify({'Price': price, 'Duration': term, 'PLot': list(a), 'FlatsTerm': term_links})
    # , 'Term': term})
    # return 'Price {0} \n Estimated Sale Time: {1} days'.format(price, term)


if __name__ == '__main__':
    app.run()
