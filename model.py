import pandas as pd
import requests
import os
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
import urllib
import psycopg2


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
        # get holidays
        # feature extraction
        # train model
        # save trained model
        pass

    def make_predictions(self):
        # model.predict
        pass
