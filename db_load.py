import pyodbc
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy as db
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


test_query = """
select [name] as database_name,
    database_id,
    create_date
from sys.databases
order by name
"""

#execute_query(engine=cnxn, query=test_query)

#engine = create_engine('jdbc:sqlserver://stromtiger.database.windows.net:1433;database=backenddb')
#engine.connect()
#metadata = db.MetaData()

data = pd.read_csv('stromverbrauch_stadtwerke.csv', sep=',')

for id, row in data.iterrows():
    date = row['Zeit']
    kw = row['kW']
    status = row['Status']
    kunde = row['Kunde']
    # create string
    raw_query = f"INSERT INTO stromlastdaten(zeit, kw, status, kundeId) VALUES ('{date}', {kw}, '{status}', {kunde})"
    print(f"Progrss: {id/len(data)}")
    # run query
    execute_query(conn=conn_str, query=raw_query, commit=True)
