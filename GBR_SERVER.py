import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
# import Realty.config as cf
from joblib import dump, load
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA

PATH_TO_PRICE_MODEL = SETTINGS.MODEL + '/PriceModel.joblib'


def Model_0(data: pd.DataFrame):
    data = data
    print("Data: ", data.shape)
    ds0 = data[((data.full_sq < data.full_sq.quantile(0.25)))]
    print('Data #0 length: ', ds0.shape)
    X0 = ds0.drop(['price'], axis=1)
    sc = StandardScaler()
    # X0 = sc.fit_transform(X0)
    ds0["price"] = np.log1p(ds0["price"])
    y0 = ds0[['price']].values.ravel()
    clf = GradientBoostingRegressor(n_estimators=150, max_depth=4, verbose=10)
    print(X0.shape, y0.shape)
    clf.fit(X0, y0)
    '''
    clf = GradientBoostingRegressor() # {'n_estimators': 150, 'max_depth': 4}
    param_grid = {
        #'min_samples_split': [10, 30, 70, 100],
        'n_estimators': [50, 150, 250, 350, 500],
        'max_depth': [2, 4, 6, 8, 10]
    }
    n_iter_search = 25
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                       n_iter=n_iter_search, cv=3, verbose=5)


    random_search.fit(X0, y0)
    print("RandomizedSearchCV" )
    print("Best0 : ", random_search.best_params_)
    '''
    print('Saving ModelMain0')

    dump(clf, PATH_TO_PRICE_MODEL + '/GBR_COORDINATES_no_bldgType0.joblib')

def Model_1(data: pd.DataFrame):
    ds1 = data[((data.full_sq >= data.full_sq.quantile(0.25)) & (data.full_sq <= data.full_sq.quantile(0.8)))]
    print('Data #1 length: ', ds1.shape)
    X1 = ds1.drop(['price'], axis=1)
    print(X1.columns)
    sc = StandardScaler()
    # X1 = sc.fit_transform(X1)

    ds1["price"] = np.log1p(ds1["price"])
    y1 = ds1[['price']].values.ravel()

    clf = GradientBoostingRegressor(n_estimators=350, max_depth=4, verbose=10)
    print(X1.shape, y1.shape)

    clf.fit(X1, y1)
    """
    clf = GradientBoostingRegressor() # {'n_estimators': 50, 'max_depth': 6}
    param_grid = {
        #'min_samples_split': [10, 30, 70, 100],
        'n_estimators': [50, 150, 250, 350, 500],
        'max_depth': [2, 4, 6, 8, 10]
    }
    n_iter_search = 30
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                       n_iter=n_iter_search, cv=3, verbose=5)

    random_search.fit(X1, y1)
    print("RandomizedSearchCV")
    print("Best1 : ", random_search.best_params_)
    """
    print('Saving ModelMain1')

    #if not os.path.exists(cf.base_dir + '/models'):
    #    os.makedirs(cf.base_dir + '/models')
    dump(clf, PATH_TO_PRICE_MODEL + '/GBR_COORDINATES_no_bldgType1.joblib')

def Model_2(data: pd.DataFrame):
    from scipy import stats

    data = data[(np.abs(stats.zscore(data.price)) < 2.7)]
    data = data[(np.abs(stats.zscore(data.term)) < 2.7)]
    X1 = data[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
               'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']]
    data["price"] = np.log1p(data["price"])
    y1 = data[['price']].values.ravel()
    print(X1.shape, y1.shape)

    clf = GradientBoostingRegressor(n_estimators=170, max_depth=4, verbose=10)
    clf.fit(X1, y1)
    dump(clf, PATH_TO_PRICE_MODEL)
    '''
    clf = GradientBoostingRegressor() #{'n_estimators': 50, 'max_depth': 6}
    param_grid = {
        #'min_samples_split': [10, 30, 70, 100],
        'n_estimators': [50, 150, 250, 350, 500],
        'max_depth': [2, 4, 6, 8, 10]
    }
    n_iter_search = 30
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                       n_iter=n_iter_search, cv=3, verbose=5)

    random_search.fit(X2, y2)
    print("RandomizedSearchCV")
    print("Best2 : ", random_search.best_params_)
    '''
    print('Saving ModelMain2')

    #if not os.path.exists(cf.base_dir + '/models'):
    #    os.makedirs(cf.base_dir + '/models')


def model():
    data = pd.read_csv(prepared_data + '/COORDINATES_Pred_Term.csv')
    #Model_0(data)

    #Model_1(data)

    Model_2(data)


if __name__ == '__main__':
    model()



'''
{'Алтуфьевский': 1, 'Южное Медведково': 114, 'Лосиноостровский': 49, 'Ярославский': 117,
 'Марьина Роща': 52, 'Марфино': 51, 'Бабушкинский': 3, 'Свиблово': 84, 'Останкинский': 70,
  'Северный': 87, 'Алексеевский': 0, 'Ростокино': 80, 'Вешняки': 13, 'Восточное Измайлово': 19,
   'Гольяново': 23, 'Ивановское': 31, 'Северное Измайлово': 85, 'Сокольники': 92, 'Новогиреево': 64,
    'Перово': 73, 'Преображенское': 76, 'Восточный': 20, 'Соколиная Гора': 91, 'Метрогородок': 54,
     'Богородское': 10, 'Новокосино': 65, 'Очаково-Матвеевское': 71, 'Солнцево': 94,
      'Ново-Переделкино': 63, 'Крылатское': 44, 'Внуково': 15, 'Бескудниковский': 6, 'Аэропорт': 2,
       'Хорошёвский': 103, 'Беговой': 5, 'Савёловский': 83, 'Молжаниновский': 58, 'Царицыно': 104,
        'Чертаново Центральное': 106, 'Чертаново Северное': 105, 'Нагатино-Садовники': 60, 'Нагорный': 61,
         'Орехово-Борисово Северное': 69, 'Бирюлёво Западное': 9, 'Бирюлёво Восточное': 8, 'Куркино': 46,
          'Строгино': 96, 'Митино': 56, 'Северное Тушино': 86, 'Покровское-Стрешнево': 75, 'Печатники': 74,
           'Лефортово': 47, 'Гагаринский': 21, 'Южное Бутово': 113, 'Котловка': 41, 'Замоскворечье': 27,
            'Басманный': 4, 'Молодёжный': 59, 'Власиха': 14, 'Ивантеевка': 32, 'Котельники': 40,
             'Красноармейск': 42, 'Лосино-Петровский': 48, 'Серпухов': 88, 'Электрогорск': 112,
              'Дмитровский': 26, 'Истринский': 33, 'Клинский': 35, 'Коломенский район': 37,
               'Красногорский': 43, 'Лотошинский': 50, 'Ногинский': 67, 'Одинцовский': 68,
                'Павлово-Посадский': 72, 'Серпуховский': 89, 'Солнечногорский': 93, 'Шаховской': 109,
                 'Щелковский': 110, 'Бутырский': 12, 'Бибирево': 7, 'Тропарёво-Никулино': 100,
                  'Раменки': 78, 'Войковский': 16, 'Сокол': 90, 'Щукино': 111, 'Южное Тушино': 115,
                   'Восточное Дегунино': 18, 'Головинский': 22, 'Западное Дегунино': 28, 'Коптево': 39,
                    'Тимирязевский': 98, 'Братеево': 11, 'Даниловский': 24, 'Зябликово': 30,
                     'Чертаново Южное': 107, 'Капотня': 34, 'Кузьминки': 45, 'Нижегородский': 62,
                      'Рязанский': 82, 'Зюзино': 29, 'Черёмушки': 108, 'Тёплый Стан': 101, 'Ясенево': 118,
                       'Коньково': 38, 'Таганский': 97, 'Мещанский': 55, 'Якиманка': 116, 'Пресненский': 77,
                        'Троицк': 99, 'Михайлово-Ярцевское': 57, 'Клёновское': 36, 'Вороновское': 17,
                         'Новофёдоровское': 66, 'Роговское': 79, 'Матушкино': 53, 'Сосенское': 95, 
                         'Филимонковское': 102, 'Рязановское': 81, 'Десёновское': 25}
'''