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

#✅MySQL 연결 정보 설정
def get_db_connection():
    try:
        secrets = st.secrets["mysql"]                                  #secrets.toml에서 가져옴
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

#✅데이터 로드 함수(캐싱을 사용하여 최적화)
@st.cache_data
def on_data():                                                        #MySQL에서 온라인 데이터를 가져오는 함수
    conn_on = get_db_connection()
    if conn_on is None:
        return pd.DataFrame()
    
    try:
        cursor_on = conn_on.cursor()
        cursor_on.execute("SELECT * FROM ontbl")
        result_on = cursor_on.fetchall()
        columns = [desc[0] for desc in cursor_on.description]         #컬럼명 가져오기
        
        if not result_on:                                             #데이터가 비어 있는 경우 예외 처리
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
def off_data():                                                        #MySQL에서 오프라인 데이터를 가져오는 함수
    conn_off = get_db_connection()
    if conn_off is None:
        return pd.DataFrame()
    
    try:
        cursor_off = conn_off.cursor()
        cursor_off.execute("SELECT * FROM offtbl")
        result_off = cursor_off.fetchall()
        columns = [desc[0] for desc in cursor_off.description]         #컬럼명 가져오기
        
        if not result_off:                                             #데이터가 비어 있는 경우 예외 처리
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

#✅데이터 로드
df_on = on_data()
df_off = off_data()
st.success("데이터 로드 완료!")

#✅날짜 데이터를 'datetime'형식으로 변환하여 연도 및 월 추출
df_on['날짜'] = pd.to_datetime(df_on['날짜'])
df_on['year'] = df_on['날짜'].dt.year
df_on['month'] = df_on['날짜'].dt.month

df_off['날짜'] = pd.to_datetime(df_off['날짜'])
df_off['year'] = df_off['날짜'].dt.year
df_off['month'] = df_off['날짜'].dt.month

#✅사이드바에 연도 및 월 선택기능 추가
st.sidebar.header(":clipboard: 데이터 선택 영역")
select_year = st.sidebar.radio(
    "2023년과 2024년 중 선택하세요",
    df_on['year'].unique()
)
select_month = st.sidebar.radio(
    "1월 ~ 12월에서 선택하세요",
    df_on['month'].unique()
)

with st.sidebar.expander("온라인"):
    select_all_device = st.checkbox("디바이스 전체 선택")
    device_options = df_on["디바이스"].unique().tolist()
    select_all_path = st.checkbox("유입경로 전체 선택")
    path_options = df_on["유입경로"].unique().tolist()

    if select_all_device:
        select_device = st.multiselect("디바이스", device_options, default = device_options)
    else:
        select_device = st.multiselect("디바이스", device_options)

    if select_all_path:
        select_path = st.multiselect("유입경로", path_options, default = path_options)
    else:
        select_path = st.multiselect("유입경로", path_options)
    time_input = st.slider("체류 시간(분)", min_value = 0, max_value = 100, value = 0, step = 5)

with st.sidebar.expander("오프라인"):
    select_region = st.selectbox("지역을 선택하세요.", df_off["지역"].unique())


#✅선택한 연도와 월에 해당하는 데이터 필터링
df_on_date = df_on[(df_on['year'] == select_year) & (df_on['month'] == select_month)]
df_off_date = df_off[(df_off['year'] == select_year) & (df_off['month'] == select_month)]

#✅유입 경로별 데이터
data = {
    "유입경로": ["직접 유입", "키워드 검색", "블로그", "인스타그램", "유튜브", "배너 광고", "트위터 X", "기타 SNS"],
    "유입수": [55, 80, 6, 9, 93, 88, 0, 62]
}

#✅PC, 모바일 유입 수 변환
device_counts = df_on_date["디바이스"].value_counts()

#✅지역별 방문자수 데이터
region_visitors = df_off_date.groupby("지역")["방문자수"].sum()

#✅지역별 참여자수 데이터
region_event = df_off_date.groupby("이벤트 종류")["참여자수"].sum()

#✅성별에 따른 참여 이벤트 데이터
gender_event = df_off_date.groupby(["이벤트 종류", "성별"])["참여자수"].sum().reset_index()

#✅주요 통계 데이터 계산
df_visit = df_off_date["방문자수"].sum()            #방문자 수 총합
df_part = df_off_date["참여자수"].sum()             #참여자 수 총합
df_leave = df_on_date['이탈률(%)'].mean()           #평균 이탈률
df_conv = df_on_date['전환율(%)'].mean()            #평균 전환율

#✅주요 지표 대시보드 표시
first_column, second_column, third_column, fourth_column = st.columns(4)

with first_column:
    st.subheader(f"{select_year}년 {select_month}월의 방문자수(단위 : 명)")
    st.subheader(f"{df_visit:,}")

with second_column:
    st.subheader(f"{select_year}년 {select_month}월의 참여자수(단위 : 명)")
    st.subheader(f"{df_part:,}")

with third_column:
    st.subheader(f"{select_year}년 {select_month}월의 이탈률(단위 : %)")
    st.subheader(f"{df_leave:.2f}")

with fourth_column:
    st.subheader(f"{select_year}년 {select_month}월의 전환율(단위 : %)")
    st.subheader(f"{df_conv:.2f}")

#📌구분선 추가
st.divider()

#✅데이터프레임 표시(온라인 / 오프라인 데이터)
with st.expander(":floppy_disk:사용된 데이터"):
    on_data, off_data = st.tabs(["온라인", "오프라인"])

    with on_data:
        st.write(":floppy_disk:**온라인 유입 데이터**")
        st.dataframe(df_on_date)

    with off_data:
        st.write(":floppy_disk:**오프라인 방문 데이터**")
        st.dataframe(df_off_date)

#📌구분선 추가
st.divider()

#✅Streamlit UI 구성
#📌온/오프라인 차트 표시 영역(Expander 사용)
with st.expander(":chart: 온/오프라인 차트"):
    #온라인, 오프라인 데이터 시각화 탭 구성
    on_chart, off_chart = st.tabs(["온라인", "오프라인"])

    #✅온라인 유입 데이터 시각화
    with on_chart:
        #📌유입 경로별 유입자 수 (막대그래프)
        st.subheader(":bar_chart: 유입 경로별 유입자 수")
        fig1, ax1 = plt.subplots(figsize = (11, 5))
        sns.barplot(
            x = '유입경로',
            y = '유입수',
            data = df_on_date,
            palette = "pastel",
            ax = ax1,
            err_kws = {'linewidth' : 0}     #에러바 제거
        )
        ax1.set_title("유입 경로별 유입수 비율", fontsize=9)
        ax1.set_xlabel = ("유입 경로")
        ax1.set_ylabel = ("유입수")
        plt.xticks(rotation =45)
        st.pyplot(fig1)

        #📌디바이스별 유입 비율 (파이 차트)
        st.subheader(":chart: 디바이스 비율")
        fig3, ax3 = plt.subplots(figsize = (7, 7))
        plt.pie(
            x = device_counts,
            labels = device_counts.index,
            autopct = "%.2f%%",             #백분율 표시
            colors = sns.color_palette("pastel")
        )
        ax3.set_title("디바이스별 유입 비율")
        plt.xticks(rotation = 45)
        st.pyplot(fig3)

        #📌체류시간과 전환율의 상관관계 (히트맵)
        st.subheader(":chart: 체류시간 VS 전환율 히트맵")
        fig5, ax5 = plt.subplots(figsize = (9, 6))
        sns.heatmap(
            df_on_date[["체류시간(min)", "전환율(가입)", "전환율(앱)", "전환율(구독)"]].corr(),
            annot = True,       #상관계수 표시
            cmap = "coolwarm",
            ax = ax5
        )
        ax5.set_title("체류시간과 전환율의 상관관계")
        plt.yticks(rotation = 45)
        st.pyplot(fig5)

    #✅오프라인 방문 데이터 시각화
    with off_chart:
        #📌지역별 방문자 수 (파이 차트)
        st.subheader(":chart: 지역별 방문자 수")
        fig2, ax2 = plt.subplots(figsize = (7, 7))
        plt.pie(
            x = region_visitors,
            labels = region_visitors.index,
            autopct = "%.2f%%",
            explode = [0.03] * len(region_visitors.index),          #조각 분리 효과
            wedgeprops = {"width" : 0.95, "edgecolor" : "w", "linewidth" : 1},      #시각적 구분
            colors = sns.color_palette("pastel")
        )
        ax2.set_title("지역별 방문자 수", fontsize = 9)
        ax2.set_xlabel = ("지역")
        ax2.set_ylabel = ("방문자 수")
        plt.xticks(rotation = 45)
        st.pyplot(fig2)

        #📌이벤트별 참여자 수 (파이 차트)
        st.subheader(":chart: 이벤트별 참여자 수")
        fig4, ax4 = plt.subplots(figsize = (7, 7))
        plt.pie(
            x = region_event,
            labels = region_event.index,
            autopct = "%.2f%%",         #백분율 표시
            explode = [0.03] * len(region_event.index),      #조각 분리 효과
            wedgeprops = {"width" : 0.95, "edgecolor" : "w", "linewidth" : 1},      #시각적 구분
            colors = sns.color_palette("pastel")
        )
        plt.xticks(rotation = 45)
        st.pyplot(fig4)

        #📌성별에 따른 참여 이벤트 (막대 그래프)
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

#📌구분선 추가
st.divider()
    
#✅머신러닝을 활용한 예측
with st.expander(":computer: 온라인 전환율 & 오프라인 방문자 수 예측"):
    on_machine, off_machine = st.tabs(["온라인", "오프라인"])

    #✅온라인 전환율 예측 모델(랜덤 포레스트 사용)
    with on_machine:
        #온라인 데이터 복사 및 원-핫 인코딩
        df_ml_on = df_on.copy()
        df_ml_on = pd.get_dummies(df_ml_on, columns = ["디바이스", "유입경로"])

        #체류시간 및 원-핫 인코딩된 디바이스, 유입경로 및 타겟 변수 설정
        features = ["체류시간(min)"] + [col for col in df_ml_on.columns if "디바이스_" in col or "유입경로_" in col]
        target = "전환율(가입)"

        #입력(X), 출력(y) 데이터 정의
        X = df_ml_on[features]
        y = df_ml_on[target]

        #학습 데이터와 테스트 데이터 분할(학습 데이터 : 80%, 테스트 데이터 : 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        #결측값 처리
        y_train.fillna(y_train.median(), inplace = True)

        #랜덤 포레스트 회귀 모델 생성 및 학습
        on_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
        on_model.fit(X_train, y_train)

        #테스트 데이터 예측
        y_pred = on_model.predict(X_test)

        #✅예측 결과 시각화(실제 전환율 VS 예측 전환율 비교)
        fig_ml_on, ax_ml_on = plt.subplots(figsize = (9, 6))
        sns.lineplot(
            x = y_test,         #실제 값
            y = y_pred,         #예측 값
            marker = "o",
            ax = ax_ml_on,
            linestyle = "-"
        )
        ax_ml_on.grid(visible = True, linestyle = "-", linewidth = 0.5)
        ax_ml_on.set_title("전환율 예측 결과 비교")
        ax_ml_on.set_xlabel("실제 전환율")
        ax_ml_on.set_ylabel("예측 전환율")
        ax_ml_on.legend()
        st.pyplot(fig_ml_on)
    
        #✅사용자가 입력한 값을 기반으로 전환율 예측
        input_data = pd.DataFrame(np.zeros((1, len(features))), columns = features)
        input_data["체류시간(min)"] = time_input    #선택된 체류 시간 입력

        #선택된 디바이스 및 유입 경로에 대한 원-핫 인코딩 적용
        for device in select_device:
            if f"디바이스_{device}" in input_data.columns:
                input_data[f"디바이스_{device}"] = 1

        for path in select_path:
            if f"유입경로_{path}" in input_data.columns:
                input_data[f"유입경로_{path}"] = 1

        #전환율 예측 결과 출력
        predicted_conversion = on_model.predict(input_data)[0]
        st.write(f"예상 전환율 : {predicted_conversion:.2f}%")

    #✅오프라인 방문자 수 예측 모델
    with off_machine:
        #✅날짜별 방문자 수 데이터 그룹화 및 전처리
        df_ml_off = df_off.groupby(["날짜", "지역"])["방문자수"].sum().reset_index()
        df_ml_off["날짜"] = pd.to_datetime(df_ml_off["날짜"])

        #날짜 데이터를 연도, 월, 일 요일로 변환
        df_ml_off["year"] = df_ml_off["날짜"].dt.year
        df_ml_off["month"] = df_ml_off["날짜"].dt.month
        df_ml_off["day"] = df_ml_off["날짜"].dt.day
        df_ml_off["day_of_week"] = df_ml_off["날짜"].dt.weekday

        #선택한 지역에 대한 데이터 필터링
        df_region = df_ml_off[df_ml_off["지역"] == select_region]

        #모델 학습을 위한 입력(X) 및 출력(y) 데이터 정의
        features = ["year", "month", "day", "day_of_week"]

        X = df_region[features]
        y = df_region["방문자수"]

        #학습 데이터와 테스트 데이터 분할(학습 데이터 : 80%, 테스트 데이터 : 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        #랜덤 포레스트 회귀 모델 생성 및 학습
        off_model = RandomForestRegressor(n_estimators=100, random_state=42)
        off_model.fit(X_train, y_train)

        #테스트 데이터 예측
        y_pred = off_model.predict(X_test)

        #✅향후 12개월 방문자 수 예측을 위한 데이터 생성
        f_dates = pd.date_range(start = df_region["날짜"].max() + pd.Timedelta(days = 1), periods = 12, freq = "M")
        f_df = pd.DataFrame({"year" : f_dates.year, 
                             "month" : f_dates.month, 
                             "day" : f_dates.day, 
                             "day_of_week" : f_dates.weekday
                            })

        #방문자 수 예측
        f_pred = off_model.predict(f_df)
        f_df["예측 방문자 수"] = f_pred
        f_df["날짜"] = f_dates

        #✅예측 결과 시각화(향후 방문자 수 예측)
        fig_ml_off, ax_ml_off = plt.subplots(figsize = (9, 6))
        ax_ml_off.plot(f_df["날짜"], f_df["예측 방문자 수"], label = "예측 방문자 수", color = "red", marker = "o")
        ax_ml_off.set_xlabel("날짜")
        ax_ml_off.set_ylabel("방문자 수")
        ax_ml_off.set_title(f"{select_region}지역 방문자 수 예측")
        ax_ml_off.legend()
        plt.xticks(rotation = 45)
        st.pyplot(fig_ml_off)

        #✅예측 방문자 수 데이터프레임 출력
        #날짜 데이터를 "YYYY-MM-01" 형식으로 변환
        f_df["날짜"] = pd.to_datetime(f_df["날짜"]).apply(lambda x: x.replace(day = 1))
        
        #방문자 수를 정수로 변환 후 "명" 추가
        f_df["예측 방문자 수"] = f_df["예측 방문자 수"].astype(int).astype(str) + "명"
        st.write(f_df[["날짜","예측 방문자 수"]])