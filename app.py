from flask import Flask, request, jsonify, render_template
import MeanPrice
import psycopg2
import settings_local as SETTINGS
from joblib import dump, load
import math
from datetime import datetime
import requests
import json

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

    mean_price, flats = MeanPrice.MeanPrices(full_sq_from, full_sq_to, rooms, latitude_from, latitude_to,
                                             longitude_from, longitude_to, price_from, price_to, building_type_str,
                                             kitchen_sq, life_sq, renovation, has_elevator, floor_first, floor_last,
                                             time_to_metro)
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
        cur.execute("select metro_id, time_to_metro from time_metro_buildings where building_id=%s",
                    (flat['id_building'],))
        metros_info = cur.fetchall()
        flat['metros'] = []
        for metro in metros_info:
            cur.execute("select name from metros where id=%s", (metro[0],))
            flat['metros'].append({'station': cur.fetchone()[0], 'time_to_metro': metro[1]})

        if 'cian' not in str(flat['offer_id']):
            flat['link'] = 'https://realty.yandex.ru/offer/' + str(flat['offer_id'])
        else:
            flat['link'] = 'https://www.cian.ru/sale/flat/' + str(str(flat['offer_id']).split('cian')[1])
        del flat['offer_id']
        del flat['id_building']
        del flat['time_to_metro']

    conn.close()

    if math.isnan(mean_price):
        mean_price = None
    return jsonify({'mean_price': mean_price, 'flats': flats, 'page': page, 'max_page': max_page, 'count': flats_count})


def func_pred_price(params: list):
    model_price = load(SETTINGS.PRICE_MODEL_PATH)
    X = params
    # X = [1, 1, 1, 23, 100, 20, 70, 0, 5, 1, 0, 0, 0]0
    pred = model_price.predict([X])
    return int(pred)


def func_pred_term(params: list):
    model_term = load(SETTINGS.SALE_TIME_MODEL_PATH)
    X = params
    pred = model_term.predict([X])
    return int(pred)


@app.route('/map')
def map():
    building_type_str = request.args.get('building_type_str')
    longitude = request.args.get('lng')
    latitude = request.args.get('lat')
    full_sq = request.args.get('full_sq')
    kitchen_sq = request.args.get('kitchen_sq')
    life_sq = request.args.get('life_sq')
    is_apartment = request.args.get('is_apartment')
    renovation = request.args.get('renovation')
    has_elevator = request.args.get('has_elevator')
    floor_first = request.args.get('floor_first')
    floor_last = request.args.get('floor_last')
    time_to_metro = request.args.get('time_to_metro')

    list_of_requested_params_price = [building_type_str, renovation, has_elevator, longitude, latitude, full_sq,
                                      kitchen_sq,
                                      life_sq, is_apartment, time_to_metro, floor_last, floor_first]

    price = func_pred_price(list_of_requested_params_price)

    # SALE TERM PREDICTION
    list_of_requested_params_term = [building_type_str, renovation, has_elevator, longitude, latitude, price, full_sq,
                                     kitchen_sq,
                                     life_sq, is_apartment, time_to_metro,
                                     floor_last, floor_first]
    term = func_pred_term(list_of_requested_params_term)
    return {'Price': price, 'Duration': term}
    # return 'Price {0} \n Estimated Sale Time: {1} days'.format(price, term)


@app.route('/api/save/', methods=['POST'])
def save():
    print(request.json)
    flat = json.loads(request.data)
    for price in flat['prices']:
        date = price[0].split(' ')[0]
        time = price[0].split(' ')[1]

        price[0] = datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]),
                        int(time.split(':')[0]), int(time.split(':')[1]), int(time.split(':')[2][:2]))
    try:
        conn = psycopg2.connect(host=SETTINGS.host, dbname=SETTINGS.name, user=SETTINGS.user,
                                password=SETTINGS.password)
        cur = conn.cursor()
    except:
        print('fail connection')
        return jsonify({'result': False})

    cur.execute("select id from districts where name=%s;", (flat['district'],))
    try:
        district_id = cur.fetchone()[0]
    except:
        print('district does not exist')
        conn.close()
        return jsonify({'result': False})
    print('district_id' + str(district_id))

    metro_ids = {}
    for metro in flat['metros']:
        try:
            cur.execute("select id from metros where name=%s;", (metro,))
            metro_id = cur.fetchone()[0]
            metro_ids.update({metro: metro_id})
        except:
            # logging.info('metro' + str(metro) + 'does not exist')
            # try:
            #     metro_location = 'Москва,метро '+ metro
            #     coords_response = requests.get(
            #         f'https://geocode-maps.yandex.ru/1.x/?apikey={self.yand_api_token}&format=json&geocode={metro_location}', timeout=5).text
            #     coords = \
            #     json.loads(coords_response)['response']['GeoObjectCollection']['featureMember'][0]['GeoObject'][
            #         'Point']['pos']
            #     longitude, latitude = coords.split(' ')
            #     longitude = float(longitude)
            #     latitude = float(latitude)
            #
            #     cur.execute("""insert into metros (longitude, latitude, city_id, created_at, updated_at, metro_id, name)
            #                    values (%s, %s, %s, %s, %s, %s, %s)""", (
            #         longitude,
            #         latitude,
            #         1,
            #         datetime.now(),
            #         datetime.now(),
            #         0,
            #         metro
            #     ))
            #     print('udated', metro)
            # except:
            # logging.info('fail in updating' + str(metro))
            continue

    try:
        coords_response = requests.get(
            'https://geocode-maps.yandex.ru/1.x/?apikey={}&format=json&geocode={}'.format(SETTINGS.yand_api_token,
                                                                                          flat["address"]),
            timeout=5).text
        coords = \
            json.loads(coords_response)['response']['GeoObjectCollection']['featureMember'][0]['GeoObject'][
                'Point'][
                'pos']
        longitude, latitude = coords.split(' ')
        longitude = float(longitude)
        latitude = float(latitude)
    except IndexError:
        print('bad address for yandex-api' + flat['address'])
        conn.close()
        return jsonify({'result': False})

    cur.execute("select id from buildings where address=%s or longitude=%s and latitude=%s;",
                (flat['address'], longitude, latitude))
    is_building_exist = cur.fetchone()
    if not is_building_exist:

        cur.execute(
            """insert into buildings
               (max_floor, building_type_str, built_year, flats_count, address, renovation,
                has_elevator, longitude, latitude, district_id, created_at, updated_at)
               values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""", (
                flat['max_floor'],
                flat['building_type_str'],
                flat['built_year'],
                flat['flats_count'],
                flat['address'],
                flat['renovation'],
                flat['has_elevator'],
                longitude,
                latitude,
                district_id,
                datetime.now(),
                datetime.now()
            ))
        cur.execute("select id from buildings where address=%s;", (flat['address'],))
        building_id = cur.fetchone()[0]
        print('building_id' + str(building_id))
        for metro, metro_id in metro_ids.items():
            try:
                cur.execute(
                    """insert into time_metro_buildings (building_id, metro_id, time_to_metro, transport_type, created_at, updated_at)
                       values (%s, %s, %s, %s, %s, %s);""", (
                        building_id,
                        metro_id,
                        flat['metros'][metro]['time_to_metro'],
                        flat['metros'][metro]['transport_type'],
                        datetime.now(),
                        datetime.now()
                    ))
            except:
                print('some new error')
                conn.close()
                return jsonify({'result': False})
    else:
        building_id = is_building_exist[0]
        print('building already exist' + str(building_id))

    cur.execute('select * from flats where offer_id=%s', (flat['offer_id'],))
    is_offer_exist = cur.fetchone()
    if not is_offer_exist:
        cur.execute(
            """insert into flats (full_sq, kitchen_sq, life_sq, floor, is_apartment, building_id, created_at, updated_at, offer_id, closed, rooms_total, image, resource_id)
               values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", (
                flat['full_sq'],
                flat['kitchen_sq'],
                flat['life_sq'],
                flat['floor'],
                flat['is_apartment'],
                building_id,
                datetime.now(),
                datetime.now(),
                'cian' + flat['offer_id'],
                flat['closed'],
                flat['rooms_count'],
                flat['image'],
                1
            ))
        cur.execute('select id from flats where offer_id=%s;', (flat['offer_id'],))
        flat_id = cur.fetchone()[0]
        print('flat_id' + str(flat_id))
    else:
        flat_id = is_offer_exist[0]
        print('flat already exist' + str(flat_id))

        cur.execute("""update flats
                       set full_sq=%s, kitchen_sq=%s, life_sq=%s, floor=%s, is_apartment=%s, building_id=%s, updated_at=%s, closed=%s, rooms_total=%s, image=%s
                       where id=%s""", (
            flat['full_sq'],
            flat['kitchen_sq'],
            flat['life_sq'],
            flat['floor'],
            flat['is_apartment'],
            building_id,
            datetime.now(),
            flat['closed'],
            flat['rooms_count'],
            flat['image'],
            flat_id
        ))
        print('updated' + str(flat_id))

    for price_info in flat['prices']:
        cur.execute('select * from prices where changed_date=%s', (price_info[0],))
        is_price_exist = cur.fetchone()
        if not is_price_exist:
            cur.execute("""insert into prices (price, changed_date, flat_id, created_at, updated_at)
                           values (%s, %s, %s, %s, %s);""", (
                price_info[1],
                price_info[0],
                flat_id,
                datetime.now(),
                datetime.now()
            ))

    conn.commit()
    cur.close()

    return jsonify({'result': True})


if __name__ == '__main__':
    app.run()
