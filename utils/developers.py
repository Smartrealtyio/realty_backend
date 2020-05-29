import pandas as pd
from scipy import stats
from datetime import datetime
import numpy as np
import os
import json
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import settings_local as SETTINGS

machine = os.path.abspath(os.getcwd())

# Define paths to Moscow and Spb data
MOSCOW_DATA_NEW = SETTINGS.DATA_MOSCOW + '/MOSCOW_NEW_FLATS.csv'
MOSCOW_DATA_SECONDARY = SETTINGS.DATA_MOSCOW + '/MOSCOW_VTOR.csv'
SPB_DATA_NEW = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'
SPB_DATA_SECONDARY = SETTINGS.DATA_SPB + '/SPB_VTOR.csv'


# Class for developers page
class Developers_API():

    def __init__(self):
        pass

    # Load data from csv
    def load_data(self, spb_new: str, spb_vtor: str, msc_new: str, msc_vtor: str):

        spb_new = pd.read_csv(spb_new)
        spb_vtor = pd.read_csv(spb_vtor)
        msc_new = pd.read_csv(msc_new)
        msc_vtor = pd.read_csv(msc_vtor)

        # Concatenate new flats + secondary flats
        self.all_spb = pd.concat([spb_new, spb_vtor], ignore_index=True, axis=0)
        self.all_msc = pd.concat([msc_new, msc_vtor], ignore_index=True, axis=0)

        # Just new flats
        # self.msc_new_flats = msc_new

        # Group dataset by full_sq
        self.list_of_squares = [38.0, 42.5, 47.0, 51.5, 56.0, 60.5, 65.0, 69.5, 74.0, 78.5, 83.0, 87.5]

        # Initialize full_sq_group values with zero
        msc_new.loc[:, 'full_sq_group'] = 0

        # Create dictionary: key = group number, value = lower threshold value of full_sq
        # Example: {1: 38.0, 2: 42.5}
        full_sq_grouping_dict = {}

        # Update "full_sq_group" column value according to "full_sq" column value
        for i in range(len(self.list_of_squares)):
            # print(i + 1, self.list_of_squares[i])
            full_sq_grouping_dict[i + 1] = self.list_of_squares[i]
            msc_new.loc[:, 'full_sq_group'] = np.where(msc_new['full_sq'] >= self.list_of_squares[i], i + 1,
                                                       msc_new['full_sq_group'])

        # Auxiliary columns to calculate flat_class: econom, comfort, business, elite
        # 0(econom) if price_meter_sq < 0.6 price_meter_sq's quantile within group
        # 1(comfort) if 0.6 price_meter_sq's quantile within group <= price_meter_sq < 0.9 price_meter_sq's quantile within group
        # 1(business) if 0.9 price_meter_sq's quantile within group <= price_meter_sq < 0.95 price_meter_sq's quantile within group
        # 1(elite) if 0.95 price_meter_sq's quantile within group <= price_meter_sq

        msc_new['price_meter_sq_06q'] = \
            msc_new.groupby(['full_sq_group', 'rooms', 'yyyy_sold', 'mm_sold'])[
                'price_meter_sq'].transform(lambda x: x.quantile(.6))
        msc_new['price_meter_sq_09q'] = \
            msc_new.groupby(['full_sq_group', 'rooms', 'yyyy_sold', 'mm_sold'])[
                'price_meter_sq'].transform(lambda x: x.quantile(.9))
        msc_new['price_meter_sq_095q'] = \
            msc_new.groupby(['full_sq_group', 'rooms', 'yyyy_sold', 'mm_sold'])[
                'price_meter_sq'].transform(lambda x: x.quantile(.95))

        # Set new column value: flat_class. 0 = econom, 1 = comfort, 2 = business, 3 = elite
        msc_new.loc[:, 'housing_class'] = 0  # Set to econom by default
        msc_new.loc[:, 'housing_class'] = np.where(
            msc_new['price_meter_sq'] >= msc_new['price_meter_sq_06q'], 1,
            msc_new['housing_class'])  # Set to comfort
        msc_new.loc[:, 'housing_class'] = np.where(
            msc_new['price_meter_sq'] >= msc_new['price_meter_sq_09q'], 2,
            msc_new['housing_class'])  # Set to business
        msc_new.loc[:, 'housing_class'] = np.where(
            msc_new['price_meter_sq'] >= msc_new['price_meter_sq_095q'], 3,
            msc_new['housing_class'])  # Set to elite

        # Remove price outliers within the groups
        std_data_new_msc = msc_new.groupby(['full_sq_group', 'rooms', 'housing_class', 'yyyy_sold', 'mm_sold'])[
            'price'].transform(
            stats.zscore)

        # Construct a Boolean Series to identify outliers
        outliers = (std_data_new_msc < -3) | (std_data_new_msc > 3)

        # Drop outliers
        msc_new = msc_new[~outliers]
        print('without outliers: ', msc_new.shape, flush=True)

        # Count number of flats in sub-group
        msc_new['mean_price_group_count'] = \
            msc_new.groupby(['full_sq_group', 'rooms', 'housing_class', 'yyyy_sold', 'mm_sold'])[
                'price'].transform('count')

        # Round price values
        msc_new.price = msc_new.price.round()

        # Transform dtype
        msc_new['mm_sold'] = msc_new['mm_sold'].astype('int')
        msc_new['mm_announce'] = msc_new['mm_announce'].astype('int')
        msc_new['yyyy_sold'] = msc_new['yyyy_sold'].astype('int')
        msc_new['yyyy_announce'] = msc_new['yyyy_announce'].astype('int')

        self.msc_new = msc_new

        self.full_sq_grouping_dict = full_sq_grouping_dict

    def parse_json(self, data=0):
        if "Storage" in machine:
            with open(data, encoding='utf-8') as read_file:
                data = json.load(read_file)
                # city_id = data["city_id"]
                city_id = 0
                longitude = data['longitude']
                latitude = data['latitude']
                is_rented = data['is_rented']
                rent_year = data['rent_year']
                rent_quarter = data['rent_quarter']
                start_timestamp = data['start_timestamp']
                floors_count = data['floors_count']
                has_elevator = data['elevator']
                parking = data['parking']
                time_to_metro = data['time_to_metro']
                flats = [i for i in data['flats_types']]
                sale_start_month = int(
                    datetime.utcfromtimestamp(data['start_timestamp']).strftime('%m'))  # Get month from unix timestamp
                sale_end_month = int(
                    datetime.utcfromtimestamp(data['end_timestamp']).strftime('%m'))  # Get year from unix timestamp
                sale_start_year = int(datetime.utcfromtimestamp(data['start_timestamp']).strftime('%Y'))
                sale_end_year = int(datetime.utcfromtimestamp(data['end_timestamp']).strftime('%Y'))
                schools_500m, schools_1000m, kindergartens_500m, kindergartens_1000m, clinics_500m, clinics_1000m, shops_500m, shops_1000m = \
                    data['schools_500m'], data['schools_1000m'], data['kindergartens_500m'], data[
                        'kindergartens_1000m'], \
                    data['clinics_500m'], data['clinics_1000m'], \
                    data['shops_500m'], data['shops_1000m']

        else:
            # city_id = data["city_id"]
            city_id = 0
            longitude = data['longitude']
            latitude = data['latitude']
            is_rented = data['is_rented']
            rent_year = data['rent_year']
            rent_quarter = data['rent_quarter']
            start_timestamp = data['start_timestamp']
            end_timestamp = data['end_timestamp']
            floors_count = data['floors_count']
            has_elevator = data['elevator']
            parking = data['parking']
            time_to_metro = data['time_to_metro']
            flats = [i for i in data['flats_types']]
            sale_start_month = int(
                datetime.utcfromtimestamp(data['start_timestamp']).strftime('%m'))  # Get month from unix timestamp
            sale_end_month = int(
                datetime.utcfromtimestamp(data['end_timestamp']).strftime('%m'))  # Get year from unix timestamp
            sale_start_year = int(datetime.utcfromtimestamp(data['start_timestamp']).strftime('%Y'))
            sale_end_year = int(datetime.utcfromtimestamp(data['end_timestamp']).strftime('%Y'))
            schools_500m, schools_1000m, kindergartens_500m, kindergartens_1000m, clinics_500m, clinics_1000m, \
            shops_500m, shops_1000m = data['schools_500m'], data['schools_1000m'], data['kindergartens_500m'], \
                                      data['kindergartens_1000m'], data['clinics_500m'], data['clinics_1000m'], \
                                      data['shops_500m'], data['shops_1000m']

        return city_id, longitude, latitude, is_rented, rent_year, rent_quarter, floors_count, has_elevator, parking, \
               time_to_metro, flats, sale_start_month, sale_end_month, sale_start_year, sale_end_year, schools_500m, \
               schools_1000m, kindergartens_500m, kindergartens_1000m, clinics_500m, clinics_1000m, shops_500m, \
               shops_1000m

    def predict(self, flats: list, rent_year: int, longitude: float, latitude: float,
                time_to_metro: int, is_rented: int, rent_quarter: int, has_elevator: int, sale_start_month: int,
                sale_end_month: int, sale_start_year: int, sale_end_year: int, housing_class: int, schools_500m=0,
                schools_1000m=0,
                kindergartens_500m=0, kindergartens_1000m=0, clinics_500m=0, clinics_1000m=0, shops_500m=0,
                shops_1000m=0, city_id=0):

        now = datetime.now()
        price_model = 0

        # lists for answer
        first_graphic = []
        second_graphic = []
        third_graphic = []

        # price changes per month for each flat type
        prices_changes_studio = {1: 1, 2: 1.045, 3: 1.095, 4: 1.12, 5: 1.17, 6: 1.19, 7: 1.12, 8: 1.13, 9: 1.14,
                                 10: 1.155, 11: 1.175, 12: 1.2}
        prices_changes_1 = {1: 1, 2: 1.035, 3: 1.085, 4: 1.1, 5: 1.15, 6: 1.185, 7: 1.12, 8: 1.14, 9: 1.165, 10: 1.175,
                            11: 1.195, 12: 1.22}
        prices_changes_2 = {1: 1, 2: 1.025, 3: 1.35, 4: 1.12, 5: 1.17, 6: 1.191, 7: 1.12, 8: 1.13, 9: 1.14, 10: 1.17,
                            11: 1.19, 12: 1.21}
        prices_changes_3 = {1: 1, 2: 1.015, 3: 1.015, 4: 1.11, 5: 1.14, 6: 1.175, 7: 1.12, 8: 1.13, 9: 1.14, 10: 1.17,
                            11: 1.195, 12: 1.2}
        prices_changes_4 = {1: 1, 2: 1.005, 3: 1.01, 4: 1.06, 5: 1.09, 6: 1.15, 7: 1.13, 8: 1.16, 9: 1.17, 10: 1.175,
                            11: 1.18, 12: 1.85}

        # INITIALIZE VARIABLES FOR GRAPHICS
        # Accumulated revenue for each flat type
        revenue_s, revenue_one_roomed, revenue_two_roomed, revenue_three_roomed, revenue_four_roomed = 0, 0, 0, 0, 0
        # Initial price_meter_sq for each flat_type
        s_price_meter_sq, one_roomed_price_meter_sq, two_roomed_price_meter_sq, three_roomed_price_meter_sq, \
        four_roomed_price_meter_sq = 0, 0, 0, 0, 0
        # Initial sales value for each flat_type
        sales_value_studio, sales_value_1, sales_value_2, sales_value_3, sales_value_4 = [], [], [], [], []
        # Accumulated sales value for each flat_type
        sales_value_studio_acc, sales_value_1_acc, sales_value_2_acc, sales_value_3_acc, sales_value_4_acc = 0, 0, 0, 0, 0
        # Initial flats_count parameter value for each flat type
        flats_count_s, flats_count_1, flats_count_2, flats_count_3, flats_count_4 = 0, 0, 0, 0, 0
        flats_count = 0
        # Growth rate depending on flat_type
        sales_volume_coeff_s, sales_volume_coeff_1, sales_volume_coeff_2, sales_volume_coeff_3, \
        sales_volume_coeff_4 = 1, 1, 1, 1, 1
        rooms = ''

        print('sale_start={0}.{1}, sale_end={2}.{3}'.format(sale_start_month, sale_start_year, sale_end_month,
                                                            sale_end_year), flush=True)
        # Calculate number of sale years
        n_years = sale_end_year - sale_start_year

        # Create sequence of months depending on start sale date and end sale date
        list_of_months = ([i for i in range(sale_start_month, 13)] + [i for i in range(1, sale_end_month + 1)]) \
            if n_years != 0 else [i for i in range(sale_start_month, 13)]
        list_of_months += ([i for i in range(1, 13)]) * int(n_years - 1)
        print('List of months: ', list_of_months, flush=True)

        yyyy_announce = sale_start_year
        # if yyyy_announce not in [2019, 2020, 2021, 2022, 2023]:
        #     return ['Error'], ['Error']

        # For each month in month sequence define sales volume
        for idx_month, mm_announce in enumerate(list_of_months):

            # Check if current month is January, change year + 1
            if mm_announce == 1 and idx_month != 0:
                yyyy_announce += 1

            # Get flat parameters for each flat
            for idx_flats, i in enumerate(flats):
                print(20 * '-', 'check front: ', idx_flats, i, flush=True)
                price_meter_sq = int(i['price_meter_sq'])
                # mm_announce = int(datetime.utcfromtimestamp(i['announce_timestamp']).strftime('%m'))  # Get month from unix
                # yyyy_announce = int(datetime.utcfromtimestamp(i['announce_timestamp']).strftime('%Y'))  # Get year from unix
                # life_sq = i['life_sq']
                flats_count = int(i['flats_count'])
                rooms = int(i['rooms']) if i['rooms'] != 's' else i['rooms']
                # print('Rooms: ', rooms, flush=True)
                # renovation = i['renovation']
                # renovation_type = i['renovation_type']
                # longitude = longitude
                # latitude = latitude
                full_sq = int(i['full_sq'])
                # kitchen_sq = i['kitchen_sq']
                # time_to_metro = time_to_metro
                # floor_last = i['floor_last']
                # floor_first = i['floor_first']
                # windows_view = i['windows_view']
                type = int(i['type'])
                # is_rented = is_rented
                # rent_year = rent_year
                # rent_quarter = rent_quarter
                # has_elevator = has_elevator

                # current_cluster = kmeans.predict([[longitude, latitude]])

                # Determine appropriate full_sq_group based on full_sq
                full_sq_group = 0
                for idx, item in enumerate(self.list_of_squares):
                    if full_sq >= item:
                        full_sq_group = idx + 1
                        break
                # print('full_sq_group={0}, full_sq={1}'.format(full_sq_group, full_sq), flush=True)

                # CALCULATE SALES VOLUME FOR EACH FLAT TYPE
                # Determine the growth rate depending on year
                n_years = yyyy_announce - sale_start_year
                if n_years >= 0:
                    sales_volume_coeff_s += 0.1  # per one year volume grows by five percent
                    sales_volume_coeff_1 += 0.09  # per one year volume grows by five percent
                    sales_volume_coeff_2 += 0.07  # per one year volume grows by five percent
                    sales_volume_coeff_3 += 0.06  # per one year volume grows by five percent
                    sales_volume_coeff_4 += 0.05  # per one year volume grows by five percent

                    # Calculate number of studios
                    if rooms == 's':
                        sales_value = round(self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                                     mm_sold=mm_announce,
                                                                                     rooms=0,
                                                                                     housing_class=housing_class) * sales_volume_coeff_s)
                        sales_value_studio.append(sales_value)

                    # Calculate number of 1-roomed flats
                    if rooms == 1:
                        sales_value = round(self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                                     mm_sold=mm_announce,
                                                                                     rooms=1,
                                                                                     housing_class=housing_class) * sales_volume_coeff_1)
                        print('mm_graphic={0}, sales_value_1={1}'.format(idx_month + 1, sales_value), flush=True)
                        sales_value_1.append(sales_value)

                    # Calculate number of 2-roomed flats
                    if rooms == 2:
                        sales_value = round(self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                                     mm_sold=mm_announce,
                                                                                     rooms=2,
                                                                                     housing_class=housing_class) * sales_volume_coeff_2)
                        print('mm_graphic={0}, sales_value_2={1}'.format(idx_month + 1, sales_value), flush=True)
                        sales_value_2.append(sales_value)

                    # Calculate number of 3-roomed flats
                    if rooms == 3:
                        sales_value = round(self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                                     mm_sold=mm_announce,
                                                                                     rooms=3,
                                                                                     housing_class=housing_class) * sales_volume_coeff_3)
                        print('mm_graphic={0}, sales_value_3={1}'.format(idx_month + 1, sales_value), flush=True)
                        sales_value_3.append(sales_value)

                    # Calculate number of 4-roomed flats
                    if rooms == 4:
                        sales_value = round(self.calculate_sales_volume_previos_year(full_sq_group=full_sq_group,
                                                                                     mm_sold=mm_announce,
                                                                                     rooms=4,
                                                                                     housing_class=housing_class) * sales_volume_coeff_4)
                        sales_value_4.append(sales_value)

                    # Calculate revenue for each type and change price depeneding on the month
                    if rooms == 's':
                        s_price_meter_sq = price_meter_sq * prices_changes_studio[mm_announce]
                        s_full_price = s_price_meter_sq * full_sq
                        revenue_s += s_full_price * sales_value_studio[-1] if len(sales_value_studio) != 0 else 0
                    if rooms == 1:
                        one_roomed_price_meter_sq = price_meter_sq * prices_changes_1[mm_announce]
                        one_roomed_full_price = one_roomed_price_meter_sq * full_sq
                        revenue_one_roomed += one_roomed_full_price * sales_value_1[-1] if len(
                            sales_value_1) != 0 else 0
                    if rooms == 2:
                        two_roomed_price_meter_sq = price_meter_sq * prices_changes_2[mm_announce]
                        two_roomed_full_price = two_roomed_price_meter_sq * full_sq
                        revenue_two_roomed += two_roomed_full_price * sales_value_2[-1] if len(
                            sales_value_2) != 0 else 0
                    if rooms == 3:
                        three_roomed_price_meter_sq = price_meter_sq * prices_changes_3[mm_announce]
                        three_roomed_full_price = three_roomed_price_meter_sq * full_sq
                        revenue_three_roomed += three_roomed_full_price * sales_value_3[-1] if len(
                            sales_value_3) != 0 else 0
                    if rooms == 4:
                        four_roomed_price_meter_sq = price_meter_sq * prices_changes_4[mm_announce]
                        four_roomed_full_price = four_roomed_price_meter_sq * full_sq
                        revenue_four_roomed += four_roomed_full_price * sales_value_4[-1] if len(
                            sales_value_4) != 0 else 0

                flats_count_s = flats_count if rooms == 's' else flats_count_s
                flats_count_1 = flats_count if rooms == 1 else flats_count_1
                flats_count_2 = flats_count if rooms == 2 else flats_count_2
                flats_count_3 = flats_count if rooms == 3 else flats_count_3
                flats_count_4 = flats_count if rooms == 4 else flats_count_4

            print('\nFirst graphic: \nMonth_graphic={0} '.format(idx_month))
            print({'month_announce': mm_announce, 'year_announce': yyyy_announce, 'month_graphic': idx_month + 1,
                   's': sum(sales_value_studio),
                   '1': sum(sales_value_1), '2': sum(sales_value_2), '3': sum(sales_value_3), '4': sum(sales_value_4),
                   'revenue_s':
                       float('{:.2f}'.format(revenue_s / 1000000)),
                   'revenue_1': float('{:.2f}'.format(revenue_one_roomed / 1000000)),
                   'revenue_2': float('{:.2f}'.format(revenue_two_roomed / 1000000)),
                   'revenue_3': float('{:.2f}'.format(revenue_three_roomed / 1000000)),
                   'revenue_4': float('{:.2f}'.format(revenue_four_roomed / 1000000))})

            # Collect data for first graphic
            first_graphic.append(
                {'month_announce': mm_announce, 'year_announce': yyyy_announce, 'month_graphic': idx_month + 1,
                 's': sum(sales_value_studio) if sum(sales_value_studio) <= flats_count_s else flats_count_s,
                 '1': sum(sales_value_1) if sum(sales_value_1) <= flats_count_1 else flats_count_1,
                 '2': sum(sales_value_2) if sum(sales_value_2) <= flats_count_2 else flats_count_2,
                 '3': sum(sales_value_3) if sum(sales_value_3) <= flats_count_3 else flats_count_3,
                 '4': sum(sales_value_4)
                 if sum(sales_value_4) <= flats_count_4 else flats_count_4,
                 'revenue_s':
                     float('{:.2f}'.format(revenue_s / 1000000)),
                 'revenue_1': float('{:.2f}'.format(revenue_one_roomed / 1000000)),
                 'revenue_2': float('{:.2f}'.format(revenue_two_roomed / 1000000)),
                 'revenue_3': float('{:.2f}'.format(revenue_three_roomed / 1000000)),
                 'revenue_4': float('{:.2f}'.format(revenue_four_roomed / 1000000))})

            # print('\nFirst graphic: ', first_graphic, flush=True)
            # Convert mm and year to datetime format

            dt_stamp = datetime(yyyy_announce, mm_announce, 1)

            print('\nSecond graphic: \nMonth_graphic={0} '.format(idx_month))
            print({'date': dt_stamp.strftime('%Y.%m.%d'),
                   's_price': s_price_meter_sq,
                   '1_price': one_roomed_price_meter_sq,
                   '2_price': two_roomed_price_meter_sq,
                   '3_price': three_roomed_price_meter_sq,
                   '4_price': four_roomed_price_meter_sq})

            # Collect data for second graphic
            second_graphic.append({'date': dt_stamp.strftime('%Y.%m.%d'),
                                   's_price': s_price_meter_sq,
                                   '1_price': one_roomed_price_meter_sq,
                                   '2_price': two_roomed_price_meter_sq,
                                   '3_price': three_roomed_price_meter_sq,
                                   '4_price': four_roomed_price_meter_sq})

            # print('\nSecond graphic: ', second_graphic, flush=True)

            # Collect data for third graphic
            # Accumulated sales values for each flat_type
            sales_value_studio_acc += sum(sales_value_studio)
            sales_value_1_acc += sum(sales_value_1)
            sales_value_2_acc += sum(sales_value_2)
            sales_value_3_acc += sum(sales_value_3)
            sales_value_4_acc += sum(sales_value_4)


            third_graphic.append({'date': dt_stamp.strftime('%Y.%m.%d'),
                                  's_sold': sales_value_studio_acc,
                                  's_all': flats_count_s})
            dt_stamp = datetime(yyyy_announce, mm_announce, 2)
            third_graphic.append({'date': dt_stamp.strftime('%Y.%m.%d'),
                                  '1_sold': sales_value_1_acc,
                                  '1_all': flats_count_1})
            dt_stamp = datetime(yyyy_announce, mm_announce, 3)
            third_graphic.append({'date': dt_stamp.strftime('%Y.%m.%d'),
                                  '2_sold': sales_value_2_acc,
                                  '2_all': flats_count_2})
            dt_stamp = datetime(yyyy_announce, mm_announce, 4)
            third_graphic.append({'date': dt_stamp.strftime('%Y.%m.%d'),
                                  '3_sold': sales_value_3_acc,
                                  '3_all': flats_count_3})
            dt_stamp = datetime(yyyy_announce, mm_announce, 5)
            third_graphic.append({'date': dt_stamp.strftime('%Y.%m.%d'),
                                  '4_sold': sales_value_4_acc,
                                  '4_all': flats_count_4})

            print('\nThird graphic: \nMonth_graphic={0} '.format(idx_month))
            print({'date': dt_stamp.strftime('%Y.%m.%d'),
                   's_sold': sales_value_studio_acc,
                   's_all': flats_count_s}, {'date': dt_stamp.strftime('%Y.%m.%d'),
                                             '1_sold': sales_value_1_acc,
                                             '1_all': flats_count_1}, {'date': dt_stamp.strftime('%Y.%m.%d'),
                                                                       '2_sold': sales_value_2_acc,
                                                                       '2_all': flats_count_2},
                  {'date': dt_stamp.strftime('%Y.%m.%d'),
                   '3_sold': sales_value_3_acc,
                   '3_all': flats_count_3}, {'date': dt_stamp.strftime('%Y.%m.%d'),
                                             '4_sold': sales_value_4_acc,
                                             '4_all': flats_count_4})
            # print('\nThird graphic: ', third_graphic, flush=True)

            # '1_sold': sales_value_1_acc,
            #                                   '1_all': flats_count_1,
            #                                   '2_sold': sales_value_2_acc,
            #                                   '2_all': flats_count_2,
            #                                   '3_sold': sales_value_3_acc,
            #                                   '3_all': flats_count_3,
            #                                   '4_sold': sales_value_4_acc,
            #                                   '4_all': flats_count_4})

            # Update
            sales_value_studio, sales_value_1, sales_value_2, sales_value_3, sales_value_4 = [], [], [], [], []

        return first_graphic, second_graphic, third_graphic

    # def train_reg(self, city_id: int, use_trained_models=True):
    #
    #     # define regression model variable
    #     reg = 0
    #
    #     # either use pretrained models
    #     if use_trained_models:
    #         if city_id == 0:
    #             reg = load(TERM_MOSCOW)
    #         elif city_id == 1:
    #             reg = load(TERM_SPB)
    #
    #     # or train regression now
    #     else:
    #         # Define city
    #         data = pd.DataFrame()
    #
    #         if city_id == 1:
    #             data = self.all_spb
    #         elif city_id == 0:
    #             data = self.all_msc
    #
    #         # Log Transformation
    #         # data['profit'] = data['profit'] + 1 - data['profit'].min()
    #         data = data._get_numeric_data()
    #         data[data < 0] = 0
    #
    #         # Remove price and term outliers (out of 3 sigmas)
    #         data = data[((np.abs(stats.zscore(data.price)) < 2.5) & (np.abs(stats.zscore(data.term)) < 2.5))]
    #
    #         data['price_meter_sq'] = np.log1p(data['price_meter_sq'])
    #         data['profit'] = np.log1p(data['profit'])
    #         # data['term'] = np.log1p(data['term'])
    #         # data['mode_price_meter_sq'] = np.log1p(data['mode_price_meter_sq'])
    #         # data['mean_term'] = np.log1p(data['mean_term'])
    #
    #         # Create X and y for Linear Model training
    #         X = data[['price_meter_sq', 'profit', 'mm_announce', 'yyyy_announce', 'rent_year', 'windows_view', 'renovation_type', 'full_sq',
    #                   'is_rented']]
    #         y = data[['term']].values.ravel()
    #
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    #
    #         # Create LinearModel and fitting
    #         # reg = LinearRegression().fit(X_train, y_train)
    #         reg = GradientBoostingRegressor(n_estimators=450, max_depth=5, verbose=1, random_state=42,
    #                                     learning_rate=0.07, max_features='sqrt', min_samples_split=5).fit(X_train, y_train)
    #         preds = reg.predict(X_test)
    #         acc = r2_score(y_test, preds)
    #         print(" Term R2 acc: {0}".format(acc))
    #     return reg

    # Расчёт месяца и года продажи при известном сроке(в днях). Предполгается, что квартиры вымещаются на продажу только в начале месяца.

    # Calculate sales volume for each flat sub-group based on its group, number of rooms, sale month
    def calculate_sales_volume_previos_year(self, rooms: int, full_sq_group: int, mm_sold: int, housing_class: int):

        # Only closed offers
        sale_volume_data = self.msc_new[(
            (self.msc_new['closed'] == True))]

        # Get sales volume
        volume_19 = sale_volume_data[
            ((sale_volume_data.rooms == rooms) & (sale_volume_data.yyyy_sold == 19) & (
                    sale_volume_data.full_sq_group == full_sq_group) & (
                     sale_volume_data.mm_sold == mm_sold) & (
                     sale_volume_data.housing_class == housing_class))].shape[0]

        return volume_19

    # def calculate_sale_month_and_year(self, type: int, term: int, yyyy_announce: int, mm_announce: int):
    #
    #     # Sale time in months
    #     n_months = ceil(term / 30)
    #
    #     sale_year = yyyy_announce
    #     # Define sale months
    #     sale_month = mm_announce + n_months - 1
    #     if sale_month % 12 != 0:
    #         if sale_month > 12 and (sale_month % 12) > 0:
    #             sale_month = sale_month % 12
    #             sale_year += 1
    #         else:
    #             sale_month = sale_month % 12
    #
    #     # print(' mm_announce: {2},\n Sale_year: {1}, \n sale_month: {0}'.format(sale_month, sale_year, mm_announce))
    #     return type, sale_year, sale_month
    #
    # def apply_calculate_sale_month_and_year(self, example: list):
    #     list_calculated_months = []
    #     for i in example:
    #         type, sale_year, sale_month = self.calculate_sale_month_and_year(type=i['type'], term=i['term'],
    #                                                                     yyyy_announce=i['yyyy_announce'],
    #                                                                     mm_announce=i['mm_announce'])
    #         list_calculated_months.append({'type': type, 'sale_year': sale_year, 'sale_month': sale_month})
    #     print(list_calculated_months)
    #     return list_calculated_months
    #
    # def create_dataframe(self, list_to_df: list, sale_start_yyyy: int, sale_end_yyyy: int,
    #                      sale_start_m: int, sale_end_m: int):
    #
    #     #  Convert list of dicts to dataframe
    #     df = pd.DataFrame(list_to_df)
    #
    #     # Calculate each group volume
    #     df = df.groupby(['type', 'sale_year', "sale_month"]).size().reset_index(name='volume')
    #
    #     # Create dummies
    #     dummies = pd.get_dummies(df['type'], prefix='flat_type')
    #
    #     # Get dummies names
    #     dummies_columns = list(dummies.columns)
    #
    #     dummies.values[dummies != 0] = df['volume']
    #     df = pd.concat([df, dummies], axis=1)
    #
    #     # Create new column based on sale_month and sale_year : mm.yy
    #     df.sale_month = df.sale_month.astype('int')
    #     df.sale_year = df.sale_year.astype('int')
    #     df['x_axis_labels'] = df[['sale_month', 'sale_year']].apply(
    #         lambda row: "{0}.{1}".format(str(row.sale_month).zfill(2), str(row.sale_year)[-2:]), axis=1)
    #
    #     # Add fictive data
    #     for year in range(sale_start_yyyy, sale_end_yyyy + 1):
    #         for month in range(1, 13):
    #             if '{0}.{1}'.format(str(month).zfill(2), str(year)[-2:]) not in df.x_axis_labels.tolist():
    #                 df.loc[len(df), 'sale_year':'sale_month'] = (year, month)
    #
    #     df.sale_month = df.sale_month.astype('int')
    #     df.sale_year = df.sale_year.astype('int')
    #     df['x_axis_labels'] = df[['sale_month', 'sale_year']].apply(
    #         lambda row: "{0}.{1}".format(str(row.sale_month).zfill(2), str(row.sale_year)[-2:]), axis=1)
    #
    #     # Create new column based on sale_month and sale_year : mm.yy
    #     df = df.fillna(0)
    #
    #     df[dummies_columns] = df.groupby(['x_axis_labels'])[dummies_columns].transform('sum')
    #
    #     df = df.sort_values(['sale_year', 'sale_month'], ascending=True)
    #
    #     df = df.drop_duplicates('x_axis_labels', keep='first')
    #
    #     new_index = df.x_axis_labels.tolist()
    #     df.index = list(new_index)
    #
    #     df = df.drop(['sale_year', 'sale_month', 'volume', 'type', 'x_axis_labels'], axis=1)
    #
    #
    #     # Plotting
    #     img = df.plot.bar(stacked=True, rot=90, title="Sales forecast", figsize=(15, 8))
    #     img.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #     # plt.xlabel('months')
    #     # plt.ylabel('volume')
    #     # img.savefig('test.png')
    #
    #     # plt.show(block=True)
    #
    #     # print(df.pivot_table(index='months', columns='volume', aggfunc='size'))
    #     # df = df.pivot_table(index='months', columns='volume', aggfunc='size')
    #     # df = df.sort_values(by='month', ascending=True)
    #     # img = df.pivot_table(index='months', columns='volume', aggfunc='size').plot.bar(stacked=True)
    #     # print(list(df.type.unique()))
    #     # img.legend(list(df.type.unique()))
    #     # save img
    #     if "Storage" in machine:
    #         img.figure.savefig('test.png')
    #     else:
    #         img.figure.savefig('/home/realtyai/smartrealty/realty/media/test.png')


def predict_developers_term(longitude: float, latitude: float, floors_count: int,
                            has_elevator: int, parking: int, time_to_metro, flats: list, housing_class: int,
                            is_rented=0, rent_year=0,
                            rent_quarter=0, sale_start_month=0, sale_end_month=0,
                            sale_start_year=0, sale_end_year=0, schools_500m=0, schools_1000m=0, kindergartens_500m=0,
                            kindergartens_1000m=0, clinics_500m=0, clinics_1000m=0, shops_500m=0, shops_1000m=0,
                            city_id=0):
    # Create Class
    devAPI = Developers_API()

    # Load CSV data. Check if it's local machine or remote server
    if "Storage" in machine:
        devAPI.load_data(spb_new=SPB_DATA_NEW, spb_vtor=SPB_DATA_SECONDARY, msc_new='None',
                         msc_vtor='None')

    else:
        devAPI.load_data(spb_new=SPB_DATA_NEW, spb_vtor=SPB_DATA_SECONDARY, msc_new=MOSCOW_DATA_NEW,
                         msc_vtor=MOSCOW_DATA_SECONDARY)

    # Parse json
    # city_id, longitude, latitude, is_rented, rent_year, rent_quarter, floors_count, has_elevator, parking, \
    # time_to_metro, flats, sale_start_month, sale_end_month, sale_start_year, sale_end_year, schools_500m, \
    # schools_1000m, kindergartens_500m, kindergartens_1000m, clinics_500m, clinics_1000m, shops_500m, \
    # shops_1000m = devAPI.parse_json(json_file)

    # Train term reg
    # reg = 0
    # if "Storage" in machine:
    #     reg = load('C:/Storage/DDG/DEVELOPERS/models/dev_term_gbr_spb.joblib')
    # else:
    #     reg = devAPI.train_reg(city_id=city_id)

    # Get answer in format: [{'month_announce': mm_announce, 'year_announce': yyyy_announce, '-1': sales_value_studio,
    #                                   '1': sales_value_1, '2': sales_value_2, '3': sales_value_3, '4': sales_value_4}, {...}]
    first_graphic, second_graphic, third_graphic = devAPI.predict(city_id=city_id, flats=flats,
                                                                  has_elevator=has_elevator,
                                                                  is_rented=is_rented,
                                                                  latitude=latitude, longitude=longitude,
                                                                  rent_quarter=rent_quarter, rent_year=rent_year,
                                                                  time_to_metro=time_to_metro,
                                                                  schools_500m=schools_500m,
                                                                  schools_1000m=schools_1000m,
                                                                  kindergartens_500m=kindergartens_500m,
                                                                  kindergartens_1000m=kindergartens_1000m,
                                                                  clinics_500m=clinics_500m,
                                                                  clinics_1000m=clinics_1000m, shops_500m=shops_500m,
                                                                  shops_1000m=shops_1000m,
                                                                  housing_class=housing_class,
                                                                  sale_end_month=sale_end_month,
                                                                  sale_end_year=sale_end_year,
                                                                  sale_start_month=sale_start_month,
                                                                  sale_start_year=sale_start_year)

    return first_graphic, second_graphic, third_graphic
