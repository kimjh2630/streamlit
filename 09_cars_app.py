import mysql.connector
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mysql.connector import Error

#1. MySQL 연결 정보
DB_HOST = 'localhost'           #호스트주소
DB_NAME = 'tabledb'             #데이터베이스 이름
DB_USER = 'root'                #MySQL 사용자 이름(기본 root, 필요할 경우 변경)
DB_PASSWORD = '1234'            #MySQL 비밀번호
DB_TABLE = 'cars'               #테이블 이름

#2. 페이지 설정
st.set_page_config(page_title = "자동차 재고 현황", page_icon = ":bar_chart:", layout = "wide")

#3. MySQL 연결 함수
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
    
#4. 데이터 가져오기 (캐싱 적용)
@st.cache_data
def get_data():
    connection = create_database_connection()
    if connection is None:
        return pd.DataFrame()
    
    try:
        query = f"SELECT * FROM {DB_TABLE}"
        df = pd.read_sql(query, connection)
        #열 이름 조정
        df = df.rename(columns = {
            "foreign_local_used" : "Foreign_Local_Used",
            "seat_make" : "seat-make",
            "make_year" : "make-year",
            "automation" : "Automation"
        })
        return df
    except Error as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            connection.close()

#5. 함수 정의
#5-1. 필터링된 데이터 캐싱
@st.cache_data
def filter_data(df, manufacturer, automation, use_category):
    return df.query(
        "manufacturer == @manufacturer & Automation == @automation & Foreign_Local_Used == @use_category"
    )

#데이터 가져오기
df = get_data()

#데이터가 없는 경우
if df.empty:
    st.error("데이터베이스에서 해당 데이터를 가져올 수 없습니다. 데이터베이스를 확인하세요.")
    st.stop()

#6. 화면 레이아웃
#6-1. 사이드바 영역
st.sidebar.header("선택하세요")

#6-1-1. 제조사 멀티셀렉트
manufacturer = st.sidebar.multiselect(
    "제조사를 선택하세요",
    options = df['manufacturer'].unique(),
    default = df['manufacturer'].unique()
)

#6-1-2. 변속기 라디오 버튼
automation = st.sidebar.radio(
    "변속기를 선택하세요:",
    options = df['Automation'].unique()
)

#6-1-3. 사용 카테고리 라디오 버튼
use_category = st.sidebar.radio(
    "카테고리를 선택하세요:",
    options = df["Foreign_Local_Used"].unique()
)

#데이터 필터링
df_select = filter_data(df, manufacturer, automation, use_category)

#필터링된 데이터가 비어있는 경우 처리
if df_select.empty:
    st.warning("현재 필터 설정에 해당하는 데이터가 없습니다.")
    st.stop()

#6-2. 메인화면
#6-2-1. 제목
st.title(":car: 자동차 재고 현황")
st.markdown("#####")
st.dataframe(df_select)

#KPI(핵심 성과 지표) 계산
avg_price = int(df_select['price'].mean() / 1000)
car_count = df_select.shape[0]
first_make_year = df_select['make-year'].min()

st.markdown("#####")

st.divider()

#6-2-2. KPI(핵심 성과 지표) 표시
first_column, second_column, third_column = st.columns(3)

with first_column:
    st.subheader("평균가격")
    st.subheader(f"US $ {avg_price:,}")

with second_column:
    st.subheader("수량")
    st.subheader(f"{car_count} Cars")

with third_column:
    st.subheader("최초 생산 년도")
    st.subheader(f"{first_make_year}년")

st.divider()

#색상별 가격 막대 차트(막대 차트)
price_per_color = df_select.groupby(by = ["color"])[["price"]].sum().sort_values(by = "price")
fig_color_price = px.bar(
    price_per_color / 1000,
    x = "price",
    y = price_per_color.index,
    orientation = "h",
    title = "색상별 가격 추이",
    color_discrete_sequence = ["#0083B8"] * len(price_per_color),
    template = "plotly_white",
)

fig_color_price.update_layout(
    plot_bgcolor = "rgba(0, 0, 0, 0)",
    xaxis = dict(showgrid = False)
)

#제조사별 가격 차트(막대 차트)
price_per_make = df_select.groupby(by = ["manufacturer"])[["price"]].sum().sort_values(by = "price")
fig_make_price = px.bar(
    price_per_make / 1000,
    x = price_per_make.index,
    y = "price",
    orientation = "v",
    title = "제조사별 가격 추이",
    color_discrete_sequence = ["#0083B8"] * len(price_per_color),
    template = "plotly_white",
)

fig_make_price.update_layout(
    plot_bgcolor = "rgba(0, 0, 0, 0)",
    xaxis = dict(showgrid = False)
)

#6-2-3. 차트 표시(2열)
left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_color_price, use_container_width = True)
right_column.plotly_chart(fig_make_price, use_container_width = True)

st.divider()

#시트 유형별 차트(파이 차트)
seat_make_dist = df_select.groupby(by = ["seat-make"])[["price"]].agg('count').sort_values(by = "seat-make")
fig_seat_dist = px.pie(
    seat_make_dist,
    values = "price",
    title = "시트 유형별 현황",
    names = seat_make_dist.index,
    color_discrete_sequence = px.colors.sequential.RdBu,
    hole = 0.4
)

fig_seat_dist.update_layout(
    plot_bgcolor = "rgba(0, 0, 0, 0)",
    xaxis = dict(showgrid = False)
)

#가격 지표
max_price = df_select['price'].max()
min_price = df_select['price'].min()
mid_price = df_select['price'].median()

#6-2-4. 3열 레이아웃
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.markdown("##")
    left_column.metric(label = "선택된 차량의 최소 가격(US $)", value = int(min_price / 1454))
    left_column.metric(label = "선택된 차량의 최대 가격(US $)", value = int(max_price / 1454))
    left_column.metric(label = "선택된 차량의 중간 가격(US $)", value = int(mid_price / 1454))

middle_column.plotly_chart(fig_seat_dist, use_container_width = True)

#생산 연도별 차트(히스토그램)
fig_make_year = px.histogram(
    df_select,
    x = "make-year",
    title = "생산 연도별 현황"
)

fig_make_year.update_layout(
    plot_bgcolor = "rgba(0, 0, 0, 0)",
    bargap = 0.1,
    xaxis = dict(showgrid = False)
)

right_column.plotly_chart(fig_make_year, use_container_width = True)

st.divider()

#Streamlit 스타일 숨기기 및 푸터 추가
st.markdown(
    """
    <style>
    footer {visibility : hidden;}
    footer:after{
        content : 'Created by Samson Afolabi';
        visibility : visible;
        position : relative;
        right : 115px;
    }
    </style>
    """,
    unsafe_allow_html = True,
)
