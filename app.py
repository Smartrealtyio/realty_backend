from flask import Flask, request, jsonify
import MeanPrice
import json
import math

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api/mean/', methods=['POST'])
def mean():
    data = json.loads(request.data)
    print(data)
    if 'price_from' not in data:
        data['price_from'] = None
    if 'price_to' not in data:
        data['price_to'] = None
    mean_price, flats = MeanPrice.MeanPrices(data['full_sq'], data['rooms'], data['latitude_from'], data['latitude_to'],
                         data['longitude_from'], data['longitude_to'], data['price_from'], data['price_to'])
    if math.isnan(mean_price):
        mean_price = None
    return jsonify({'mean_price': mean_price, 'flats': flats})


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    return jsonify({'key':5})


if __name__ == '__main__':
    app.run()
