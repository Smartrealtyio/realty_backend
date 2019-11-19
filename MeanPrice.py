import pandas as pd
import settings_local as SETTINGS
import MainPreprocessing as MPrep


def MeanPrices(full_sq_from, full_sq_to, rooms, latitude_from, latitude_to,
               longitude_from, longitude_to, price_from=None, price_to=None, building_type_str=None, kitchen_sq=None,
               life_sq=None, renovation=None, has_elevator=None, floor_first=None, floor_last=None, time_to_metro=None):
    df = pd.read_csv(SETTINGS.DATA + '/COORDINATES_MEAN_PRICE.csv')
    price_from = float(price_from) if price_from != None else float(df.price.min())
    price_to = float(price_to) if price_to != None else float(df.price.max())
    full_sq_from = float(full_sq_from) if full_sq_from != None else float(df.full_sq.min())
    full_sq_to = float(full_sq_to) if full_sq_to != None else float(df.full_sq.max())
    # building_type_str = int(building_type_str) if building_type_str != None else df.building_type_str\
    #    .isin(df.building_type_str.tolist())
    # Running main preprocessing

    # print('List: ', df.building_type_str.tolist())
    # print("Check:", df.building_type_str.isin((df.building_type_str.tolist())))
    # Apply requested parameters as filter for flats database
    filter = (((df.full_sq >= full_sq_from) & (df.full_sq <= full_sq_to))
              & (df.rooms == rooms) &
              ((df.latitude >= latitude_from) & (df.latitude <= latitude_to))
              & ((df.longitude >= longitude_from) & (df.longitude <= longitude_to)) &
              ((df.price >= price_from) & (df.price <= price_to)))  # & (df.building_type_str == building_type_str))
    filter1 = (df.rooms == True)
    # df = df.query('rooms == ')
    df = df[filter]

    if time_to_metro != None:
        df = df[(df.time_to_metro <= time_to_metro)]
    if rooms != None:
        df = df[df.rooms == rooms]
    if building_type_str != None:
        df = df[df.building_type_str == building_type_str]
    if kitchen_sq != None:
        df = df[(df.kitchen_sq >= kitchen_sq - 5) & (df.kitchen_sq <= kitchen_sq + 5)]
    if life_sq != None:
        df = df[(df.life_sq >= life_sq - 5) & (df.life_sq <= life_sq + 5)]
    if renovation != None:
        df = df[df.renovation == renovation]
    if has_elevator != None:
        df = df[df.has_elevator == has_elevator]
    if floor_first != None:
        df = df[df.floor_first == 0]
    if floor_last != None:
        df = df[df.floor_last == 0]


    # Mean price per meter across all filtered flats
    mean_price = (df['price'] / df['full_sq']).mean()

    # Find all flats below mean price
    flats_below_mean_price = df[(df['price'] / df['full_sq']) <= mean_price]
    print('mean_price', type(mean_price))
    print('flats_below_mean_price', type(flats_below_mean_price.to_dict('index')))
    print('mean_price', mean_price)
    print('flats_below_mean_price', flats_below_mean_price.to_dict('index'))
    # return str(mean_price)
    return mean_price, flats_below_mean_price.to_dict('records')


if __name__ == '__main__':
    # MeanPrices(full_sq=70, latitude_from=10.0, latitude_to=11.0, longitude_from=10.0,
    #            longitude_to=11.0, rooms=2, price_from=None, price_to=None)
    pass