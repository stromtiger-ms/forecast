import pandas as pd
import requests
import os
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
import urllib
import psycopg2
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from keras import layers
import keras


class MLModel:
    def __init__(self):
        load_dotenv()
        params = urllib.parse.quote_plus \
            (r'DRIVER={SQL Server};'
             r'SERVER=tcp:stromtiger.database.windows.net;'
             r'DATABASE=backenddb;'
             r'UID=tigeradmin;'
             r'PWD=Supertigerpassword$123;'
             r'Encrypt=yes;'
             r'TrustServerCertificate=no;'
             r'Connection Timeout=30;')

        user = os.getenv('USER')
        password = os.getenv('PASSWORD')
        host = os.getenv('HOST')
        port = os.getenv('PORT')
        hostname = host + ':' + port
        database_name = os.getenv('DATABASE')

        conn_str = f'postgresql+psycopg2://{user}:{password}@{hostname}/{database_name}'

        cnx = create_engine(conn_str, echo=True)
        self.cnx = cnx

    def upload_holidays(self, file_path: str, table_name: str):
        holiday_data = pd.read_excel(file_path, engine='openpyxl')
        holiday_data.to_sql(name=table_name, con=self.cnx, if_exists='replace', index=False, chunksize=10000)

    def _get_holidays(self, table_names: list(), date_col: str) -> pd.DataFrame:
        holidays = pd.read_sql(sql=f"""SELECT * FROM public.{table_names[0]}
                                       UNION
                                       SELECT * FROM public.{table_names[1]}
                                       UNION
                                       SELECT * FROM public.{table_names[2]};""",
                               parse_dates=[date_col],
                               con=self.cnx)
        return holidays

    def _feature_extraction(self, table_name: str, date_col: str, holiday_tables: list()):
        data = pd.read_sql(sql=f"SELECT * FROM {table_name}",
                           con=self.cnx,
                           parse_dates=[date_col])
        # get weekdays
        data['Weekday'] = data[date_col].apply(lambda x: x.weekday())
        # check if is weekend
        weekdays = data['Weekday'].unique().sort()
        data['isWeekend'] = data['Weekday'].apply(lambda x: 1 if x in [5,6] else 0)
        # get all holidays
        holidays = self._get_holidays(table_names=holiday_tables, date_col='Datum')
        # convert weekdays to number equivalent
        weekdays_dict = {'Montag': 0,
                         'Dienstag': 1,
                         'Mittwoch': 2,
                         'Donnerstag': 3,
                         'Freitag': 4,
                         'Samstag': 5,
                         'Sonntag': 6}
        holidays['Wochentag'] = holidays['Wochentag'].apply(lambda x: weekdays_dict[x])
        # get "Ferien"
        years = [2020, 2021, 2022]
        ferien_lst = list()
        for year in years:
            url = f"https://ferien-api.de/api/v1/holidays/NW/{year}"
            response = requests.get(url)
            ferien_lst.append(response.json())
        keys = ['start', 'end', 'year', 'name']
        start, end, year, name = [], [], [], []
        ferien_dict = dict.fromkeys(keys, [])

        for year_idx, _ in enumerate(years):
            for idx, _ in enumerate(years):
                start.append(ferien_lst[year_idx][idx]['start'])
                end.append(ferien_lst[year_idx][idx]['end'])
                year.append(ferien_lst[year_idx][idx]['year'])
                name.append(ferien_lst[year_idx][idx]['name'])

        ferien_dict['start'] = start
        ferien_dict['end'] = end
        ferien_dict['year'] = year
        ferien_dict['name'] = name

        ferien = pd.DataFrame.from_dict(ferien_dict)

        # merge dfs
        data['Day'] = data['zeit'].apply(lambda x: datetime.strptime(str(x).split()[0], '%Y-%m-%d'))
        data = data.merge(holidays[['Datum']], how='left', left_on='Day', right_on='Datum')
        data['IstFeiertag'] = data['Day'] == data['Datum']
        data['IstFeiertag'] = data['IstFeiertag'].apply(lambda x: int(x))

        # get IstFerientag
        ferien['TagStart'] = ferien['start'].apply(lambda x: datetime.strptime(str(x).split('T')[0], '%Y-%m-%d'))
        ferien['TagEnd'] = ferien['end'].apply(lambda x: datetime.strptime(str(x).split('T')[0], '%Y-%m-%d'))

        ferientage = []
        for idx, row in ferien.iterrows():
            start_date = row['TagStart']
            end_date = row['TagEnd']
            ferientage.extend(pd.date_range(start=start_date, end=end_date))
        data['IstFerientag'] = data['Day'].isin(ferientage)
        data['IstFerientag'] = data['IstFerientag'].apply(lambda x: int(x))

        return data[['zeit', 'kw', 'status', 'verbraucherid', 'Weekday', 'isWeekend', 'IstFeiertag', 'IstFerientag']]

    def train_model(self):

        def _create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), :]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 2])
            return np.array(dataX), np.array(dataY)

        # feature extraction
        preprocessed_data = self._feature_extraction(table_name='stromlastdaten',
                                        date_col='zeit',
                                        holiday_tables=['feiertage2020nrw',
                                                        'feiertage2021nrw',
                                                        'feiertage2022nrw'])
        preprocessed_data.drop(['zeit', 'status', 'Weekday'], axis=1, inplace=True)
        # get unique customerIds
        customers = preprocessed_data['verbraucherid'].unique()
        # create df for for predictions
        preds = pd.DataFrame(columns=customers)

        print('Start training')
        for customer in customers:
            print(f'Start training for customer {customer}')
            data = preprocessed_data[preprocessed_data['verbraucherid'] == customer].drop('verbraucherid', axis=1)

            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(data)

            # split into train and test sets
            train_size = int(len(dataset) * 0.67)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

            # reshape into X=t and Y=t+1
            look_back = 1
            trainX, trainY = _create_dataset(train, look_back)
            testX, testY = _create_dataset(test, look_back)

            # reshape input to be  [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], look_back, 4))
            testX = np.reshape(testX, (testX.shape[0], look_back, 4))

            # create and train model
            model = keras.Sequential()
            model.add(layers.LSTM(100, return_sequences=True, input_shape=(trainX.shape[1], 4)))
            model.add(layers.LSTM(100, return_sequences=False))
            model.add(layers.Dense(25))
            model.add(layers.Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(trainX, trainY, batch_size=1, epochs=3)

            # make predictions
            train_predict = model.predict(trainX)
            test_predict = model.predict(testX)

            # save predictions
            preds[str(customer)] = test_predict.flatten()

        return preds

    def export_predictions(self, predictions: pd.DataFrame):
        predictions['sum'] = predictions.sum(axis=1)
        return predictions
