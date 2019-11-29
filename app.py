from flask import Flask, request, jsonify, render_template
# import MeanPrice
import FIND_OUTLIERS
import psycopg2
import settings_local as SETTINGS
from sklearn import linear_model
from joblib import dump, load
import math as m
import math
from datetime import datetime
import requests
import json
import pandas as pd
import numpy as np

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
    # flats_page_count = int(request.args.get('flats_page_count')) if request.args.get('flats_page_count') is not None else 10

    # mean_price, flats = FIND_OUTLIERS.OutliersSearch(full_sq_from, full_sq_to, rooms, latitude_from, latitude_to,
    #                                                  longitude_from, longitude_to, price_from, price_to, building_type_str,
    #                                                  kitchen_sq, life_sq, renovation, has_elevator, floor_first, floor_last,
    #
    #
    #                           time_to_metro)

    DATA_OUTLIERS = SETTINGS.DATA + '/COORDINATES_OUTLIERS.csv'
    MODEL_OUTLIERS = SETTINGS.MODEL + '/models.joblib'

    data = pd.read_csv(DATA_OUTLIERS)
    data = data[['price_meter_sq', 'full_sq']]
    data = data[data.price_meter_sq < data.price_meter_sq.quantile(0.2)]

    model = load(MODEL_OUTLIERS)
    # outliers = model.predict(data)
    outliers_it = data[model.predict(data) == -1]
    print('Outliers: ', outliers_it.shape[0], flush=True)
    outliers_it['flat_id'] = outliers_it.index
    new_data = pd.read_csv(DATA_OUTLIERS)
    print(new_data.shape)
    new_data['flat_id'] = new_data.index
    ds = pd.merge(new_data, outliers_it, left_on="flat_id", right_on="flat_id", suffixes=['', 'double'])
    ds = ds.drop(['flat_id', 'full_sqdouble', 'price_meter_sqdouble'], axis=1)

    print('ds shape', ds.shape, flush=True)

    filter = (((ds.full_sq >= full_sq_from) & (ds.full_sq <= full_sq_to)) & (ds.rooms == rooms) &
              ((ds.latitude >= latitude_from) & (ds.latitude <= latitude_to)
               & (ds.longitude >= longitude_from) & (ds.longitude <= longitude_to)))
    ds = ds[filter]

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

    print('ds columns', ds.columns, flush=True)
    print(ds.head(), flush=True)

    flats = ds.to_dict('record')


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
        print(flat.keys(), flush=True)
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
        del flat['offer_id']
        del flat['id_building']
        del flat['time_to_metro']

    conn.close()

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


def func_pred_term0(params):
    model_term = load(SETTINGS.MODEL + '/GBR_COORDINATES_TERM0.joblib')
    X = params
    pred = model_term.predict([X])
    return pred


def func_pred_term1(params):
    model_term = load(SETTINGS.MODEL + '/GBR_COORDINATES_TERM1.joblib')
    X = params
    pred = model_term.predict([X])
    return pred


def func_pred_term2(params):
    model_term = load(SETTINGS.MODEL + '/GBR_COORDINATES_TERM2.joblib')
    X = params
    pred = model_term.predict([X])
    return pred


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
    data = pd.read_csv(SETTINGS.DATA  + '/COORDINATES_Pred_Term.csv')

    if full_sq < float(data.full_sq.quantile(0.1)):
        print('0')
        price = func_pred_price0(list_of_requested_params_price)
        price = price[0]
    elif ((full_sq >= float(data.full_sq.quantile(0.1))) & (full_sq <= float(data.full_sq.quantile(0.8)))):
        print('1')
        price = func_pred_price1(list_of_requested_params_price)
        price = price[0]
    elif full_sq > float(data.full_sq.quantile(0.8)):
        print('2')
        price = func_pred_price2(list_of_requested_params_price)
        price = price[0]
    price_meter_sq = price / full_sq

    # SALE TERM PREDICTION
    list_of_requested_params_term = [renovation, has_elevator, longitude, latitude, price, full_sq, kitchen_sq,
                                      is_apartment, time_to_metro,
                                     floor_last, floor_first, X, Y]
    term = 0
    # Data
    data = pd.read_csv(SETTINGS.DATA  + '/COORDINATES_Pred_Term.csv')
    filter_term = (((data.full_sq <= full_sq + 8) & (data.full_sq >= full_sq - 8)) & (
            (data.longitude >= longitude - 0.01) & (data.longitude <= longitude + 0.01) &
            (data.latitude >= latitude - 0.01) & (data.latitude <= latitude + 0.01)) &
                   ((data.price_meter_sq <= price_meter_sq + 2000) & (data.price_meter_sq >= price_meter_sq - 2000))
                   & ((data.time_to_metro >= time_to_metro - 2) & (data.time_to_metro <= time_to_metro + 2)))
    data_term = data[(filter_term)]


    if data_term.shape[0] < 1:
        filter_term = (((data.full_sq <= full_sq + 8) & (data.full_sq >= full_sq - 8)) & (
                (data.longitude >= longitude - 0.08) & (data.longitude <= longitude + 0.08) &
                (data.latitude >= latitude - 0.08) & (data.latitude <= latitude + 0.08)) &
                       ((data.price_meter_sq <= price_meter_sq + 2000) & (data.price_meter_sq >= price_meter_sq - 2000))
                       & ((data.time_to_metro >= time_to_metro - 2) & (data.time_to_metro <= time_to_metro + 2)))
        data_term = data[(filter_term)]


    reg = linear_model.LinearRegression().fit(data_term[['price']], data_term[['term']])


    term = reg.predict([[price]])

    if term < 0:
        filter_term = (((data.full_sq <= full_sq + 8) & (data.full_sq >= full_sq - 8)) & (
                (data.longitude >= longitude - 0.08) & (data.longitude <= longitude + 0.08) &
                (data.latitude >= latitude - 0.08) & (data.latitude <= latitude + 0.08)) &
                       ((data.price_meter_sq <= price_meter_sq + 2000) & (data.price_meter_sq >= price_meter_sq - 2000))
                       & ((data.time_to_metro >= time_to_metro - 2) & (data.time_to_metro <= time_to_metro + 2)))
        data_term = data[(filter_term)]
        reg = linear_model.LinearRegression().fit(data_term[['price']], data_term[['term']])

        # sns.lmplot(x='term', y='price', data=data_term)
        # plt.show()
        term = reg.predict([[price]])

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
    filter1 = ((data.full_sq <= full_sq + 1) & (
            (data.longitude >= longitude - 0.08) & (data.longitude <= longitude + 0.08) &
            (data.latitude >= latitude - 0.08) & (data.latitude <= latitude + 0.08)) &
               ((data.price_meter_sq <= price_meter_sq + 5000) & (data.price_meter_sq >= price_meter_sq - 5000)) &
               (data.term < 450) & (
                           (data.time_to_metro >= time_to_metro - 2) & (data.time_to_metro <= time_to_metro + 2)))
    ds = data[filter1]
    print(ds.shape)

    x = ds.term
    x = np.sort(x).tolist()

    y = ds.price
    y = np.sort(y).tolist()
    a = []
    a += ({'x{0}'.format(k): x, 'y{0}'.format(k): y} for k, x, y in zip(list(range(len(x))), x, y))
    print(a)

    return jsonify({'Price': price, 'Duration': term.tolist()[0], 'PLot': list(a)})
    # , 'Term': term})
    # return 'Price {0} \n Estimated Sale Time: {1} days'.format(price, term)


if __name__ == '__main__':
    app.run()
