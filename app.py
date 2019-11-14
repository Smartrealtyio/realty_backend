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

    mean_price, flats = MeanPrice.MeanPrices(**data)

    if math.isnan(mean_price):
        mean_price = None

    return jsonify({'mean_price': mean_price, 'flats': flats})


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    return jsonify({'key':5})


if __name__ == '__main__':
    app.run()
