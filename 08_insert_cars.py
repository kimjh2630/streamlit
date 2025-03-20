import mysql.connector
import pandas as pd
import pymysql
import streamlit as st

from mysql.connector import Error
from sqlalchemy import create_engine

#MySQL 연결 정보
DB_HOST = 'localhost'           #호스트주소
DB_NAME = 'tabledb'             #데이터베이스 이름
DB_USER = 'root'                #MySQL 사용자 이름(기본 root, 필요할 경우 변경)
DB_PASSWORD = '1234'            #MySQL 비밀번호
DB_TABLE = 'cars'               #테이블 이름

#CSV 파일 경로
CSV_FILE_PATH = "./data/cars.csv"

#데이터베이스 연결 함수
def create_database_connection():
    try:
        connection = mysql.connector.connect(
            host = DB_HOST,
            database = DB_NAME,
            user = DB_USER,
            password = DB_PASSWORD
        )
        if connection.is_connected():
            print("MySQL 데이터베이스에 성공적으로 연결되었습니다.")
        return connection
    except Error as e:
        print(f"데이터베이스 연결 중 오류 발생: {e}")
        return None
    
#테이블 생성 함수
def create_table(connection):
    try:
        cursor = connection.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            foreign_local_used VARCHAR(20),
            color VARCHAR(20),
            wheel_drive INT,
            automation VARCHAR(20),
            seat_make VARCHAR(20),
            price BIGINT,
            description VARCHAR(100),
            make_year INT,
            manufacturer VARCHAR(50)
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        print(f"'{DB_TABLE}' 테이블이 성공적으로 생성되었습니다.")
    except Error as e:
        print(f"테이블 생성 중 오류 발생: {e}")
    finally:
        cursor.close()

#데이터 삽입 함수
def insert_data(connection, df):
    try:
        cursor = connection.cursor()
        insert_query = f"""
        INSERT INTO {DB_TABLE}(
            foreign_local_used, color, wheel_drive, automation, seat_make,
            price, description, make_year, manufacturer
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        #판다스 데이터 프레임을 numpy배열로 만들어서
        #배열을 하나씩(1 record) 꺼내어 튜플로 변환하여 리스트로 변환
        data = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data)
        connection.commit()
        print(f"{cursor.rowcount}개의 레코드가 '{DB_TABLE}' 테이블에 삽입되었습니다.")
    except Error as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
    finally:
        cursor.close()

#메인 함수
def main():
    #CSV 파일 읽기 (index_col = 0으로 첫 번째 열을 인덱스로 설정)
    try:
        df = pd.read_csv(CSV_FILE_PATH, index_col = 0)
        print("CSV 파일을 성공적으로 읽었습니다.")
    except FileNotFoundError:
        print(f"파일 '{CSV_FILE_PATH}'을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"CSV 파일 읽기 중 오류 발생: {e}")
        return

    #열 이름 조정
    expected_columns = [
        "foreign_local_used", "color", "wheel_drive", "automation",
        "seat_make", "price", "description", "make_year", "manufacturer"
    ]
    df.columns = expected_columns  #열 이름 매핑

    #데이터베이스 연결
    connection = create_database_connection()
    if connection is None:
        return

    #테이블 생성
    create_table(connection)

    #데이터 삽입
    insert_data(connection, df)

    #연결 종료
    if connection.is_connected():
        connection.close()
        print("MySQL 연결이 종료되었습니다.")
    
    #SQLAlchemy 엔진 생성 (MySQL용)
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

    #데이터 가져오기 함수
    def select_data():
        query = "SELECT * FROM cars;"
        df = pd.read_sql(query, engine)     #SQLAlchemy 엔진 사용
        return df

    #Streamlit 앱
    st.title("MySQL 데이터 조회")

    if st.button("데이터 불러오기"):
        df = select_data() 
        st.dataframe(df)                    #데이터 출력

if __name__ == "__main__":
    main()