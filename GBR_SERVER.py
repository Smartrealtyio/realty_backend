import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
from joblib import dump, load
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA
PATH_TO_PRICE_MODEL = SETTINGS.MODEL


def model():
    # Load and split dataset.
    ds = pd.read_csv(prepared_data + '/COORDINATES_Pred_Price.csv')

    # 0
    ds0 = ds[((ds.full_sq < ds.full_sq.quantile(0.1)))]
    print('Data #0 length: ', ds0.shape)

    X0 = ds0.drop(['price'], axis=1)
    y0 = ds0[['price']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.01, random_state=42)
    clf = GradientBoostingRegressor(n_estimators=300, max_depth=12, verbose=10)

    clf.fit(X_train, y_train)
    print('Saving ModelMain0')

    dump(clf, PATH_TO_PRICE_MODEL + '/GBR_COORDINATES_no_bldgType0.joblib')

    # 1
    ds1 = ds[((ds.full_sq >= ds.full_sq.quantile(0.1)) & (ds.full_sq <= ds.full_sq.quantile(0.8)))]
    print('Data #1 length: ', ds1.shape)

    X1 = ds1.drop(['price'], axis=1)
    y1 = ds1[['price']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.01, random_state=42)
    clf = GradientBoostingRegressor(n_estimators=300, max_depth=12, verbose=10)

    clf.fit(X_train, y_train)
    print('Saving ModelMain1')

    dump(clf, PATH_TO_PRICE_MODEL + '/GBR_COORDINATES_no_bldgType1.joblib')

    # 2
    ds2 = ds[((ds.full_sq > ds.full_sq.quantile(0.8)))]
    print('Data #2 length: ', ds0.shape)

    X2 = ds2.drop(['price'], axis=1)
    y2 = ds2[['price']].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.01, random_state=42)
    clf = GradientBoostingRegressor(n_estimators=300, max_depth=12, verbose=10)

    clf.fit(X_train, y_train)
    print('Saving ModelMain2')

    dump(clf, PATH_TO_PRICE_MODEL + '/GBR_COORDINATES_no_bldgType2.joblib')

    '''
    param_grid = {
    'max_depth': [5, 8, 10, 12, 15],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
    }
    clf = GradientBoostingRegressor(learning_rate=0.1)
    grid_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid,
                               cv=3, n_jobs=-1, verbose=8, n_iter=10)
    grid_search.fit(X_train, y_train)
    print("Best Score: ", grid_search.best_score_)
    print('Best Params: ', grid_search.best_params_)
    '''
    # clf = GradientBoostingRegressor(learning_rate=0.1, n_estimators=600, min_samples_split=8,
    # min_samples_leaf=5, max_features=2, max_depth=10, verbose=10)

    # clf = GradientBoostingRegressor(learning_rate=0.07, n_estimators=1000, min_samples_split=8,
    #                                min_samples_leaf=5, max_features=2, max_depth=10, verbose=10)

    # r2_score_val = r2_score(y_test, pd.Series(pred))

    # print("\nR2_score: ", r2_score_val)


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
