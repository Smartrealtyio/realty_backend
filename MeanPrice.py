import pandas as pd
import settings_local as SETTINGS
import MainPreprocessing as MPrep


def MeanPrices(full_sq, rooms, latitude_from, latitude_to,
               longitude_from, longitude_to, price_from=None, price_to=None):
    MPrep.main_preprocessing()
    df = pd.read_csv(SETTINGS.DATA + '/COORDINATES_MEAN_PRICE.csv')
    price_from = float(price_from) if price_from != None else float(df.price.min())
    price_to = float(price_to) if price_to != None else float(df.price.max())
    # Running main preprocessing

    # Apply requested parameters as filter for flats database
    filter = (((df.full_sq >= full_sq - 5) & (df.full_sq <= full_sq + 5)) & (df.rooms == rooms) &
              ((df.latitude >= latitude_from) & (df.latitude <= latitude_to)
               & (df.longitude >= longitude_from) & (df.longitude <= longitude_to)) &
              ((df.price >= price_from) & (df.price <= price_to)))

    df = df[filter]

    # Mean price per meter across all filtered flats
    mean_price = (df['price'] / df['full_sq']).mean()

    # Find all flats below mean price
    flats_below_mean_price = df[(df['price'] / df['full_sq']) <= mean_price]
    print('mean_price', type(mean_price))
    print('flats_below_mean_price', type(flats_below_mean_price.to_dict('index')))
    print('mean_price', mean_price)
    print('flats_below_mean_price', flats_below_mean_price.to_dict('index'))
    # return str(mean_price)
    return mean_price, flats_below_mean_price.to_dict('index')


if __name__ == '__main__':
    MeanPrices(full_sq=75.0, latitude_from=55.81, latitude_to=55.815, longitude_from=37.59,
               longitude_to=37.595, rooms=3, price_from=None, price_to=None)