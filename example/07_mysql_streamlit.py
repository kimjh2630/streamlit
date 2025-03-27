import pandas as pd
import streamlit as st

from sqlalchemy import create_engine

DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'tabledb'
DB_USER = 'root'
DB_PASS = '1234'

print("DB_HOST : ", DB_HOST)
print("DB_PORT : ", DB_PORT)
print("DB_NAME : ", DB_NAME)
print("DB_USER : ", DB_USER)
print("DB_PASS : ", DB_PASS)

#SQLAlchemy 엔진 생성 (MySQL용)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

#데이터 가져오기 함수
def fetch_data():
    query = "SELECT * FROM usertbl;"
    df = pd.read_sql(query, engine)     #SQLAlchemy 엔진 사용
    return df

#Streamlit 앱
st.title("MySQL 데이터 조회")

if st.button("데이터 불러오기"):
    df = fetch_data()
    st.dataframe(df)                    #데이터 출력