from faker import Faker
from mysql.connector import Error

import mysql.connector
import pandas as pd
import streamlit as st

#MySQL 연결 정보 (secrets.toml에서 가져옴)
def get_db_connection():
    try:
        secrets = st.secrets["mysql"]
        connection = mysql.connector.connect(
            host = secrets["host"],
            database = secrets["database"],
            user = secrets["user"],
            password = secrets["password"]
        )
        if connection.is_connected():
            print("MySQL 데이터베이스에 성공적으로 연결되었습니다.")
        return connection
    except Error as e:
        st.error(f"데이터베이스 연결 실패: {e}")
        return None

#users 테이블을 생성하는 함수 정의
def create_users_table():
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                customer_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                age INT
            );
        """)
        conn.commit()
        st.success("users 테이블이 생성되었습니다.")
    except Error as e:
        st.error(f"users 테이블 생성 실패: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

#더미 데이터 삽입 함수
def insert_fake_data(num_rows):
    conn = get_db_connection()
    if conn is None:
        return
    
    fake = Faker('ko_KR')
    try:
        cursor = conn.cursor()
        for _ in range(num_rows):
            name = fake.name()
            age = fake.random_int(min = 18, max = 80)
            cursor.execute(
                "INSERT INTO users (name, age) VALUES (%s, %s)",
                (name, age)
            )
        conn.commit()
        st.success(f"{num_rows}개의 가짜 데이터가 추가되었습니다.")
    except Error as e:
        st.error(f"데이터 삽입 실패: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

#데이터 조회 함수
def fetch_users():
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        result = cursor.fetchall()
        columns = ["customer_id", "name", "age"]
        df = pd.DataFrame(result, columns = columns)
        return df
    except Error as e:
        st.error(f"데이터 조회 실패: {e}")
        return pd.DataFrame()
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

#Streamlit UI
st.title("MySQL + Faker 데이터 생성")
st.write(st.secrets)  #secrets.toml 내용 확인용

if st.button("users 테이블 생성"):
    create_users_table()

num_rows = st.number_input("추가할 가짜 데이터 개수", min_value = 1, max_value = 1000, value = 10)
if st.button("가짜 데이터 추가"):
    insert_fake_data(num_rows)

if st.button("데이터 조회"):
    df = fetch_users()
    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("조회된 데이터가 없습니다.")