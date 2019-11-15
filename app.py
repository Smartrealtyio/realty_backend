from flask import Flask, request, jsonify
import MeanPrice
import math
import psycopg2
import settings_local as SETTINGS

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


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
    building_type_str = float(request.args.get('building_type_str')) if request.args.get('building_type_str') is not None else None
    kitchen_sq = float(request.args.get('kitchen_sq')) if request.args.get('kitchen_sq') is not None else None
    life_sq = float(request.args.get('life_sq')) if request.args.get('life_sq') is not None else None
    renovation = float(request.args.get('renovation')) if request.args.get('renovation') is not None else None
    has_elevator = float(request.args.get('has_elevator')) if request.args.get('has_elevator') is not None else None
    floor_first = float(request.args.get('floor_first')) if request.args.get('floor_first') is not None else None
    floor_last = float(request.args.get('floor_last')) if request.args.get('floor_last') is not None else None
    time_to_metro = float(request.args.get('time_to_metro')) if request.args.get('time_to_metro') is not None else None

    mean_price, flats = MeanPrice.MeanPrices(full_sq_from, full_sq_to, rooms, latitude_from, latitude_to,
                                             longitude_from, longitude_to, price_from, price_to, building_type_str,
                                             kitchen_sq, life_sq, renovation, has_elevator, floor_first, floor_last, time_to_metro)

    conn = psycopg2.connect(host=SETTINGS.host, dbname=SETTINGS.name, user=SETTINGS.user, password=SETTINGS.password)
    cur = conn.cursor()
    for id, flat in flats.items():
        cur.execute("select metro_id, time_to_metro from time_metro_buildings where building_id=%s", (flat['id_building'],))
        metros_info = cur.fetchall()
        flat['metros'] = []
        for metro in metros_info:
            cur.execute("select name from metros where id=%s", (metro[0],))
            flat['metros'].append({'station': cur.fetchone()[0], 'time_to_metro': metro[1]})

        flat['link'] = 'https://realty.yandex.ru/offer/' + str(flat['offer_id'])
        del flat['offer_id']
        del flat['id_building']
        del flat['time_to_metro']

    conn.close()

    if math.isnan(mean_price):
        mean_price = None
    return jsonify({'mean_price': mean_price, 'flats': flats})


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    return jsonify({'key': 5})


if __name__ == '__main__':
    app.run()
