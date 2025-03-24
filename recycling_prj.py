from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
from mysql.connector import Error

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
import streamlit as st

#한글 폰트 설정 (맑은 고딕 적용)
plt.rc("font", family = "Malgun Gothic")

#Streamlit 페이지 설정
st.set_page_config(page_title = "Recycling Dashboard", 
                   page_icon = ":chart:",
                   layout = "wide",
                   initial_sidebar_state = 'collapsed')

#Streamlit 앱 제목
st.title(":bar_chart: 유입 경로 및 지역별 방문자 분석")

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

#데이터 로드 함수(캐싱을 사용하여 최적화)
@st.cache_data
def on_data():
    conn_on = get_db_connection()
    if conn_on is None:
        return pd.DataFrame()
    
    try:
        cursor_on = conn_on.cursor()
        cursor_on.execute("SELECT * FROM ontbl")
        result_on = cursor_on.fetchall()
        columns = [desc[0] for desc in cursor_on.description]         #컬럼명 가져오기
        
        if not result_on:                                             #데이터가 비어 있는지 확인
            st.error("온라인 데이터가 비어 있습니다.")
            return pd.DataFrame(columns = columns)                    #빈 데이터프레임 반환
        
        df_on = pd.DataFrame(result_on, columns = columns)            #컬럼 개수 맞추기
        return df_on
    except Error as e:
        st.error(f"데이터 조회 실패: {e}")
        return pd.DataFrame()
    finally:
        if conn_on.is_connected():
            cursor_on.close()
            conn_on.close()

@st.cache_data
def off_data():
    conn_off = get_db_connection()
    if conn_off is None:
        return pd.DataFrame()
    
    try:
        cursor_off = conn_off.cursor()
        cursor_off.execute("SELECT * FROM offtbl")
        result_off = cursor_off.fetchall()
        columns = [desc[0] for desc in cursor_off.description]         #컬럼명 가져오기
        
        if not result_off:                                             #데이터가 비어 있는지 확인
            st.error("오프라인 데이터가 비어 있습니다.")
            return pd.DataFrame(columns = columns)                     #빈 데이터프레임 반환
        
        df_off = pd.DataFrame(result_off, columns = columns)           #컬럼 개수 맞추기
        return df_off
    except Error as e:
        st.error(f"데이터 조회 실패: {e}")
        return pd.DataFrame()
    finally:
        if conn_off.is_connected():
            cursor_off.close()
            conn_off.close()

#데이터 로드
df_on = on_data()
df_off = off_data()
st.success("데이터 로드 완료!")

#날짜 데이터를 'datetime'형식으로 변환하여 연도 및 월 추출
df_on['날짜'] = pd.to_datetime(df_on['날짜'])
df_on['year'] = df_on['날짜'].dt.year
df_on['month'] = df_on['날짜'].dt.month

df_off['날짜'] = pd.to_datetime(df_off['날짜'])
df_off['year'] = df_off['날짜'].dt.year
df_off['month'] = df_off['날짜'].dt.month

#사이드바에 연도 및 월 선택기능 추가
st.sidebar.header("사이드바")
select_year = st.sidebar.radio(
    "2023년과 2024년 중 선택하세요",
    df_on['year'].unique()
)
select_month = st.sidebar.radio(
    "1월 ~ 12월에서 선택하세요",
    df_on['month'].unique()
)

#선택한 연도와 월에 해당하는 데이터 필터링
df_on_date = df_on[(df_on['year'] == select_year) & (df_on['month'] == select_month)]
df_off_date = df_off[(df_off['year'] == select_year) & (df_off['month'] == select_month)]

#유입 경로별 데이터
data = {
    "유입경로": ["직접 유입", "키워드 검색", "블로그", "인스타그램", "유튜브", "배너 광고", "트위터 X", "기타 SNS"],
    "유입수": [55, 80, 6, 9, 93, 88, 0, 62]
}

#PC, 모바일 유입 수 변환
device_counts = df_on_date["디바이스"].value_counts()

#지역별 방문자수 데이터
region_visitors = df_off_date.groupby("지역")["방문자수"].sum()

#지역별 참여자수 데이터
region_event = df_off_date.groupby("이벤트 종류")["참여자수"].sum()

#성별에 따른 참여 이벤트 데이터
gender_event = df_off_date.groupby(["이벤트 종류", "성별"])["참여자수"].sum().reset_index()

df_part = df_off_date['참여자수'].sum()
df_visit = df_off_date['방문자수'].sum()
df_leave = df_on_date['이탈률(%)'].mean()
df_conv = df_on_date['전환율(%)'].mean()

first_column, second_column, third_column, fourth_column = st.columns(4)

with first_column:
    st.subheader(f"{select_year}년 {select_month}월의 방문자수(단위 : 명)")
    st.subheader(f"{df_part:,}")

with second_column:
    st.subheader(f"{select_year}년 {select_month}월의 참여자수(단위 : 명)")
    st.subheader(f"{df_visit:,}")

with third_column:
    st.subheader(f"{select_year}년 {select_month}월의 이탈률(단위 : %)")
    st.subheader(f"{df_leave:.2f}")

with fourth_column:
    st.subheader(f"{select_year}년 {select_month}월의 전환율(단위 : %)")
    st.subheader(f"{df_conv:.2f}")

st.divider()

#데이터프레임 표시
with st.expander(":floppy_disk:사용된 데이터"):
    on_data, off_data = st.tabs(["온라인", "오프라인"])

    #온라인 탭
    with on_data:
        st.write(":floppy_disk:**온라인 유입 데이터**")
        st.dataframe(df_on_date)

    #오프라인 탭
    with off_data:
        st.write(":floppy_disk:**오프라인 방문 데이터**")
        st.dataframe(df_off_date)

st.divider()

#Streamlit UI 구성
#차트 표시
with st.expander(":chart: 온/오프라인 차트"):
    on_chart, off_chart = st.tabs(["온라인", "오프라인"])

    #온라인 유입 데이터 시각화
    with on_chart:
        #유입 경로별 유입수 막대그래프
        st.subheader(":bar_chart: 유입 경로별 유입자 수")
        fig1, ax1 = plt.subplots(figsize = (11, 5))
        sns.barplot(
            x = '유입경로',
            y = '유입수',
            data = df_on_date,
            palette = "pastel",
            ax = ax1,
            err_kws = {'linewidth' : 0}
        )
        ax1.set_title("유입 경로별 유입수 비율", fontsize=9)
        ax1.set_xlabel = ("유입 경로")
        ax1.set_ylabel = ("유입수")
        plt.xticks(rotation =45)
        st.pyplot(fig1)

        #유입 디바이스 파이차트
        st.subheader(":chart: 디바이스 비율")
        fig3, ax3 = plt.subplots(figsize = (7, 7))
        plt.pie(
            x = device_counts,
            labels = device_counts.index,
            autopct = "%.2f%%",
            colors = sns.color_palette("pastel")
        )
        ax3.set_title("디바이스별 유입 비율")
        plt.xticks(rotation = 45)
        st.pyplot(fig3)

        #체류시간과 전환율의 상관관계 히트맵
        st.subheader(":chart: 체류시간 VS 전환율 히트맵")
        fig5, ax5 = plt.subplots(figsize = (9, 6))
        sns.heatmap(
            df_on_date[["체류시간(min)", "전환율(가입)", "전환율(앱)", "전환율(구독)"]].corr(),
            annot = True,
            cmap = "coolwarm",
            ax = ax5
        )
        ax5.set_title("체류시간과 전환율의 상관관계")
        plt.yticks(rotation = 45)
        st.pyplot(fig5)

    #오프라인 방문 데이터 시각화
    with off_chart:
        #지역별 방문자수 막대그래프
        st.subheader(":chart: 지역별 방문자 수")
        fig2, ax2 = plt.subplots(figsize = (7, 7))
        plt.pie(
            x = region_visitors,
            labels = region_visitors.index,
            autopct = "%.2f%%",
            explode = [0.03] * len(region_visitors.index),  #조각 분리 효과
            wedgeprops = {"width" : 0.95, "edgecolor" : "w", "linewidth" : 1},
            colors = sns.color_palette("pastel")
        )
        ax2.set_title("지역별 방문자 수", fontsize = 9)
        ax2.set_xlabel = ("지역")
        ax2.set_ylabel = ("방문자 수")
        plt.xticks(rotation = 45)
        st.pyplot(fig2)

        #이벤트별 참여자수 파이차트
        st.subheader(":chart: 이벤트별 참여자 수")
        fig4, ax4 = plt.subplots(figsize = (7, 7))
        plt.pie(
            x = region_event,
            labels = region_event.index,
            autopct = "%.2f%%",
            explode = [0.03] * len(region_event.index),      #조각 분리 효과
            wedgeprops = {"width" : 0.95, "edgecolor" : "w", "linewidth" : 1},
            colors = sns.color_palette("pastel")
        )
        plt.xticks(rotation = 45)
        st.pyplot(fig4)

        #성별에 따른 참여 이벤트 라인차트
        st.subheader(":bar_chart: 성별에 따른 참여 이벤트")
        fig6, ax6 = plt.subplots(figsize = (11, 5))
        sns.barplot(
            x = '이벤트 종류',
            y = '참여자수',
            data = gender_event,
            hue = "성별",
            palette = {"남": "#0000ff", "여": "#ff0000"},
            ax = ax6
        )
        ax6.set_title("성별에 따른 참여 이벤트", fontsize=9)
        ax6.set_xlabel = ("이벤트 종류")
        ax6.set_ylabel = ("참여자 수")
        ax6.legend(loc = "upper left")
        plt.xticks(rotation = 45)
        st.pyplot(fig6)

st.divider()
with st.sidebar.expander("온라인"):
    select_device = st.multiselect("디바이스", df_on["디바이스"].unique())
    select_path = st.multiselect("유입경로", df_on["유입경로"].unique())
    time_input = st.slider("체류 시간(분)", min_value = 0, max_value = 100, value = 0, step = 5)

#머신러닝 부분
with st.expander(":computer: 온라인 전환율 & 오프라인 방문자 수 예측"):
    on_machine, off_machine = st.tabs(["온라인", "오프라인"])

    with on_machine:
        df_ml_on = df_on.copy()
        df_ml_on = pd.get_dummies(df_ml_on, columns = ["디바이스", "유입경로"])

        features = ["체류시간(min)"] + [col for col in df_ml_on.columns if "디바이스_" in col or "유입경로_" in col]
        target = "전환율(가입)"

        X = df_ml_on[features]
        y = df_ml_on[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        y_train.fillna(y_train.median(), inplace=True)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        fig_ml_on, ax_ml_on = plt.subplots(figsize = (9, 6))
        sns.scatterplot(
            x = y_test,
            y = y_pred,
            alpha = 0.5,
            ax = ax_ml_on
        )
        ax_ml_on.plot([0, 1], [0, 1], "r--")
        ax_ml_on.set_title("전환율 예측 결과 비교")
        ax_ml_on.set_xlabel("실제 전환율")
        ax_ml_on.set_ylabel("예측 전환율")
        st.pyplot(fig_ml_on)
    
        input_data = pd.DataFrame(np.zeros((1, len(features))), columns = features)
        input_data["체류시간(min)"] = time_input

        for device in select_device:
            if f"디바이스_{device}" in input_data.columns:
                input_data[f"디바이스_{device}"] = 1

        for path in select_path:
            if f"유입경로_{path}" in input_data.columns:
                input_data[f"유입경로_{path}"] = 1

        predicted_conversion = model.predict(input_data)[0]
        st.write(f"예상 전환율 : {predicted_conversion:.2f}%")

    with off_machine:
        st.write("방문자 수 예측")

