import pyodbc
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy as db
from datetime import datetime
import urllib
load_dotenv()

host = os.getenv('HOST')
port = os.getenv('PORT')
database = os.getenv('DATABASE')
user = os.getenv('USER')
password = os.getenv('PASSWORD')
driver = '{ODBC Driver 17 for SQL Server}'

# create engine
conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=tcp:stromtiger.database.windows.net;'
    r'DATABASE=backenddb;'
    r'UID=tigeradmin;'
    r'PWD=Supertigerpassword$123;'
)
cnxn = pyodbc.connect(conn_str)


# write function for executing queries
def execute_query(conn: str, query: str, commit: bool):
    engine = pyodbc.connect(conn)
    cursor = engine.cursor()
    cursor.execute(query)
    if commit:
        engine.commit()
    # row = cursor.fetchone()
    # while row:
    #     print(row)
    #     row = cursor.fetchone()
    pass


params = urllib.parse.quote_plus\
    (r'DRIVER={SQL Server};'
    r'SERVER=tcp:stromtiger.database.windows.net;'
    r'DATABASE=backenddb;'
    r'UID=tigeradmin;'
    r'PWD=Supertigerpassword$123;'
    r'Encrypt=yes;'
    r'TrustServerCertificate=no;'
    r'Connection Timeout=30;')
conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
cnx = create_engine(conn_str, echo=True)

if __name__ == '__main__':
    data = pd.read_csv('stromverbrauch_stadtwerke.csv', sep=',')
    start = datetime.now()
    print(start)
    print('Upload in progress..')
    data.to_sql(name='stromlastdaten', con=cnx, if_exists='replace', index=False, chunksize=10000)
    print('done')
    print(round((datetime.now() - start).total_seconds()))
