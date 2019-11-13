from flask import Flask, request, jsonify
import MeanPrice
import json

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/mean/', methods=['POST'])
def mean():
    data = json.loads(request.data)
    print(data)
    if 'price_from' not in data:
        data['price_from'] = None
    if 'price_to' not in data:
        data['price_to'] = None
    response = MeanPrice.MeanPrices(data['full_sq'], data['rooms'], data['latitude_from'], data['latitude_to'],
                         data['longitude_from'], data['longitude_to'], data['price_from'], data['price_to'])
    return jsonify(response)


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    return jsonify({'key':5})


if __name__ == '__main__':
    app.run()
