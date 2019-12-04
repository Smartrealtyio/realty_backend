from flask import Flask, request, jsonify, render_template
# import MeanPrice
import FIND_OUTLIERS
import psycopg2
import settings_local as SETTINGS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
import math as m
import math
from datetime import datetime
import requests
import json
import pandas as pd
import numpy as np
import math

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
    original_paper_score = np.array([(((-1 * s + 0.5) - 1) * 100) * 2 for s in sklearn_score_anomalies])
    print(original_paper_score)
    df_f = pd.DataFrame({'profit': original_paper_score}, index=full_data_outliers.index)
    print(df_f.head())
    new_df = pd.concat([full_data_outliers, df_f], axis=1)


    print('ds shape', new_df.shape, flush=True)

    # filter = (((ds.rooms == rooms)|(ds.rooms == 0)|(ds.rooms==1)|(ds.rooms==2)) &
    #          ((ds.latitude >= latitude_from) & (ds.latitude <= latitude_to))
    #           & ((ds.longitude >= longitude_from) & (ds.longitude <= longitude_to)))
    # ds = ds[filter]

    print('ds', new_df.shape, flush=True)

    # if time_to_metro != None:
    #     ds = ds[(ds.time_to_metro <= time_to_metro)]
    # if rooms != None:
    #     ds = ds[ds.rooms == rooms]
    # if building_type_str != None:
    #     ds = ds[ds.building_type_str == building_type_str]
    # if kitchen_sq != None:
    #     ds = ds[(ds.kitchen_sq >= kitchen_sq - 5) & (ds.kitchen_sq <= kitchen_sq + 5)]
    # if life_sq != None:
    #     ds = ds[(ds.life_sq >= life_sq - 5) & (ds.life_sq <= life_sq + 5)]
    # if renovation != None:
    #     ds = ds[ds.renovation == renovation]
    # if has_elevator != None:
    #     ds = ds[ds.has_elevator == has_elevator]
    # if floor_first != None:
    #     ds = ds[ds.floor_first == 0]
    # if floor_last != None:
    #     ds = ds[ds.floor_last == 0]
    # if price_from != None:
    #     ds = ds[ds.price >= price_from]
    # if price_to != None:
    #     ds = ds[ds.price <= price_to]

    print('ds columns', new_df.columns, flush=True)
    print(new_df.head(), flush=True)

    flats = new_df.to_dict('record')


    flats_count = len(flats)
    flats_page_count = 10
    max_page = math.ceil(len(flats) / flats_page_count)
    page = page if page <= max_page else 1
    if sort_type == 0:
        flats = sorted(flats, key=lambda x: x['price'])[(page - 1) * flats_page_count:page * flats_page_count]
    else:
        flats = sorted(flats, key=lambda x: x['price'])[(page - 1) * flats_page_count:page * flats_page_count]

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


def func_pred_price0(params):
    model_price = load(SETTINGS.MODEL + '/GBR_COORDINATES_no_bldgType0.joblib')
    X = params
    pred = model_price.predict([X])
    return np.expm1(pred)


def func_pred_price1(params):
    model_price = load(SETTINGS.MODEL + '/GBR_COORDINATES_no_bldgType1.joblib')
    X = params

    pred = model_price.predict([X])
    return np.expm1(pred)



def func_pred_price2(params):
    model_price = load(SETTINGS.MODEL + '/GBR_COORDINATES_no_bldgType2.joblib')
    X = params

    pred = model_price.predict([X])
    return np.expm1(pred)



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
    price = 0
    data = pd.read_csv(SETTINGS.DATA  + '/COORDINATES_Pred_Price.csv')

    if full_sq < float(data.full_sq.quantile(0.25)):
        print('0')
        price = func_pred_price0(list_of_requested_params_price)
        price = int(price[0])
    elif ((full_sq >= float(data.full_sq.quantile(0.25))) & (full_sq <= float(data.full_sq.quantile(0.8)))):
        print('1')
        price = func_pred_price1(list_of_requested_params_price)
        price = int(price[0])
    elif full_sq > float(data.full_sq.quantile(0.8)):
        print('2')
        price = func_pred_price2(list_of_requested_params_price)
        price = int(price[0])
    price_meter_sq = price / full_sq

    # SALE TERM PREDICTION
    list_of_requested_params_term = [renovation, has_elevator, longitude, latitude, price, full_sq, kitchen_sq,
                                      is_apartment, time_to_metro,
                                     floor_last, floor_first, X, Y]
    term = 0
    # Data
    data = pd.read_csv(SETTINGS.DATA + '/COORDINATES_Pred_Term.csv')
    print("Initial shape: ", data.shape)




    filter1 = (((data.full_sq <= full_sq + 1) & (data.full_sq >= full_sq - 3)) & (
            (data.longitude >= longitude - 0.05) & (data.longitude <= longitude + 0.05) &
            (data.latitude >= latitude - 0.05) & (data.latitude <= latitude + 0.05)) &
               ((data.price_meter_sq <= price_meter_sq + 20000) & (
                           data.price_meter_sq >= price_meter_sq - 20000)) & (
                       (data.time_to_metro >= time_to_metro - 2) & (
                           data.time_to_metro <= time_to_metro + 2)))

    data_term = data[filter1]

    print('SHAPE #2: ', data_term.shape[0])

    reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=50)
    reg.fit(data_term[['price']], data_term[['term']])

    term = reg.predict([[price]])
    term = int(term.item(0))
    print(term)


    # Add links to flats
    term_links = data_term.to_dict('record')
    for i in term_links:
        if i['resource_id'] == 0:
            i['link'] = 'https://realty.yandex.ru/offer/' + str(i['offer_id'])
        else:
            i['link'] = 'https://www.cian.ru/sale/flat/' + str(i['offer_id'])


    '''
    if float(price) < float(data.price.quantile(0.2)):
        print('0')
        term = func_pred_term0(list_of_requested_params_term)

    elif (float(price) >= float(data.price.quantile(0.2))) & (float(price) <= float(data.price.quantile(0.85))):
        print('1')
        term = func_pred_term1(list_of_requested_params_term)

    elif float(price) > float(data.price.quantile(0.85)):
        print('2')
        term = func_pred_term2(list_of_requested_params_term)
    '''
    filter1 = (((data.full_sq <= full_sq + 3) & (data.full_sq >= full_sq - 3)) & (
            (data.longitude >= longitude - 0.05) & (data.longitude <= longitude + 0.05) &
            (data.latitude >= latitude - 0.05) & (data.latitude <= latitude + 0.05)) &
               (data.term <= term) & ((data.price_meter_sq <= price_meter_sq + 20000) & (
                           data.price_meter_sq >= price_meter_sq - 20000)) & ((data.time_to_metro >= time_to_metro - 2) & (
                        data.time_to_metro <= time_to_metro + 2)))
    # ds = data_term[data_term.term <= term]
    ds = data[filter1]
    print(ds.shape)

    ds = ds[ds.price <= price+100000]
    x = ds.term
    x = x.tolist()
    x += [term]

    y = ds.price
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
