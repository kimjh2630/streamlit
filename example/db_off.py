import mysql.connector
import numpy as np
import pandas as pd
import pymysql
import streamlit as st

from mysql.connector import Error
from sqlalchemy import create_engine

#데이터베이스 정보
DB_HOST = 'localhost'           #호스트주소
DB_NAME = 'prjdb'               #데이터베이스 이름
DB_USER = 'root'                #MySQL 사용자 이름(기본 root, 필요할 경우 변경)
DB_PASSWORD = '1234'            #MySQL 비밀번호
DB_TABLE = 'offtbl'             #사용할 테이블 이름
DB_PORT = '3306'                #MySQL 포트번호

#CSV 파일 경로 설정
CSV_FILE_PATH = "./data/recycling_off.csv"

#DB 연결함수
def create_database_connection():
    try:
        connection = mysql.connector.connect(
            host = DB_HOST,
            database = DB_NAME,
            user = DB_USER,
            password = DB_PASSWORD
        )
        if connection.is_connected():
            print("DB연결에 성공하였습니다.")
        return connection
    except Error as e:
        print(f"데이터베이스 연결 중 오류 발생: {e}")
        return None
    
#테이블 생성 함수
def create_table(connection):
    try:
        cursor = connection.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE}(
            id INT AUTO_INCREMENT PRIMARY KEY,
            날짜 DATE,
            지역 VARCHAR(20),
            방문자수 INT,
            연령대 VARCHAR(20),
            성별 VARCHAR(10),
            `이벤트 종류` VARCHAR(20),
            참여자수 INT,
            참여비율 FLOAT
        );
        """

        cursor.execute(create_table_query)
        connection.commit()
        print(f"'{DB_TABLE}' 테이블이 성공적으로 생성되었습니다.")
    except Error as e:
        print(f"테이블 생성 중 오류 발생:{e}")
    finally:
        cursor.close()

#데이터 삽입 함수
def insert_data(connection, df):
    try:
        cursor = connection.cursor()
        insert_query = f"""
        INSERT INTO {DB_TABLE}(
        날짜,
        지역,
        방문자수,
        연령대,
        성별,
        `이벤트 종류`,
        참여자수,
        참여비율
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        data = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data)
        connection.commit()
        print(f"{cursor.rowcount}개의 레코드가 '{DB_TABLE}'테이블에 삽입되엇습니다.")
    except Error as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
    finally:
        cursor.close()

#메인 함수
def main():
    try:
        df = pd.read_csv(CSV_FILE_PATH, index_col = 0, encoding = 'cp949')
        print("CSV 파일을 성공적으로 읽었습니다.")
        df.replace([np.inf, -np.inf], np.nan, inplace = True)
    except FileNotFoundError:
        print(f"파일 '{CSV_FILE_PATH}'을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"CSV 파일 읽기 중 오류 발생: {e}")
        return
    
    #CSV 파일 컬럼명을 데이터베이스의 테이블 컬럼명과 동일하게 변경
    expected_columns = [
        "날짜",
        "지역",
        "방문자수",
        "연령대",
        "성별",
        "이벤트 종류",
        "참여자수",
        "참여비율"
    ]
    df.columns = expected_columns

    #데이터베이스 연결
    connection = create_database_connection()
    if connection is None:
        return
    
    #테이블 생성
    create_table(connection)

    #데이터 삽입
    insert_data(connection, df)

    #데이터베이스 연결 종료
    if connection.is_connected():
        connection.close()
        print("MySQL 연결이 종료되었습니다.")

    #SQLALchemy 엔진 생성(Streamlit에서 데이터 조회 시 사용)
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    #데이터 조회 함수
    def select_data():
        query = "SELECT * FROM offtbl;"
        df = pd.read_sql(query, engine)
        return df
    
    #Streamlit UI 구성
    st.title("데이터베이스 조회")

    if st.button("데이터 불러오기"):
        df = select_data()
        st.dataframe(df, use_container_width = True)

#메인 함수 실행 
if __name__ == "__main__":
    main()