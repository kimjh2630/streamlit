from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import datetime
import time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

#초기 설정 : Streamlit의 페이지 레이아웃을 "wide"로 설정하여 넓게 사용
st.set_page_config(layout = "wide")

#한글 폰트 설정(Window 환경에서 "Malgun Gothic" 사용)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#CSV 파일 경로(GitHub에서 불러오기)
CSV_FILE_PATH = "https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/recycling_off.csv"

#색상 팔레트(Plotly에서 사용)
palette = px.colors.qualitative.Set2

#데이터 로딩 함수(Streamlit의 캐싱 기능 활용)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_FILE_PATH, encoding = "UTF8")
        #무한대(inf) 값 처리 > NaN으로 변환 후 삭제
        df.replace([np.inf, -np.inf], np.nan, inplace = True)
        df.dropna(subset=["날짜"], inplace = True)

        #컬럼명 영어로 변경
        df = df.rename(
            columns = {
                "날짜" : "DATE",
                "지역" : "CITY",
                "방문자수" : "VISITORS",
                "연령대" : "age",
                "성별" : "gender",
                "이벤트 종류" : "CAMP",
                "참여자수" : "PART",
                "참여비율" : "P_Ratio"
            }
        )

        #날짜 데이터 타입 변환 및 결측치 제거
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df.dropna(subset = ["DATE"], inplace = True)

        #요일 컬럼 추가(숫자를 한글 요일로 변환)
        df["WEEKDAY"] = df["DATE"].dt.day_of_week
        week_mapping = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        df["WEEKDAY"] = df["WEEKDAY"].map(week_mapping)
        return df

    except Exception as e:
        st.error(f"CSV 로딩 오류: {e}")
        return pd.DataFrame()


#데이터 필터링 함수
@st.cache_data
def filter_data(df, start_date, end_date, selected_day, selected_city):
    df_filtered = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]
    
    #특정 요일만 선택한 경우 필터 적용
    if selected_day and "All" not in selected_day:
        df_filtered = df_filtered[df_filtered["WEEKDAY"].isin(selected_day)]
    
    #특정 지역만 선택한 경우 필터 적용
    if "All_CITIES" not in selected_city:
        df_filtered = df_filtered[df_filtered["CITY"].isin(selected_city)]
    return df_filtered


#데이터 시각화 함수
def barplot(df, x, y):
    if not df.empty:
        df_grouped = df.groupby(x)[y].mean().reset_index()
        fig = px.bar(
            df_grouped,
            x = x,
            y = y,
            color = x,
            color_discrete_sequence = palette,
            title = f"{x}별 평균 {y} 비교",
            text_auto = True,
            hover_data = [y]
        )
        fig.update_layout(legend_title_text=x)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")

#박스플롯 (데이터 분포 분석)
def baxplot(df, x, y):
    if not df.empty:
        fig = px.box(
            df,
            x = x,
            y = y,
            title = f"{x}별 {y} 분포",
            points = "outliers",    #이상치 표시
            color = x,
            color_discrete_sequence = palette,
            hover_data = [x, y]
        )
        fig.update_layout(legend_title_text=x)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")

#선형 차트 (트렌드 분석)
def linechart(df, x, y):
    if not df.empty:
        df_grouped = df.groupby(x)[y].mean().reset_index()
        fig = px.line(
            df_grouped,
            x = x,
            y = y,
            markers = True,
            color_discrete_sequence = palette,
            title = f"{x}별 평균 {y} 비교",
            hover_data = [y]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")

#산점도 (상관관계 분석)
def scatterplot(df, x, y):
    if not df.empty:
        fig = px.scatter(
            df,
            x = x,
            y = y,
            color = x,
            color_discrete_sequence = palette,
            title = f"{x} vs {y} 의 상관관계",
            hover_data = [x, y]
        )
        fig.update_layout(legend_title_text=x)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("조회된 데이터가 없습니다.")

#원형 차트 (비율 비교)
def piechart(df, x, y):
    if not df.empty:
        if x not in df.columns or y not in df.columns:
            st.error(f"⚠️ 컬럼 '{x}' 또는 '{y}'가 존재하지 않습니다.")
            return
        df_grouped = df.groupby(x)[y].sum().reset_index()
        fig = px.pie(
            df_grouped,
            names = x,
            values = y,
            title = f"{x}별 {y} 비율 비교",
            hole = 0.3,
            color = x,
            color_discrete_sequence = palette,
            hover_data = [y]
        )
        fig.update_layout(legend_title_text = x, width = 900, height = 700)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("⚠️ 선택한 조건에 해당하는 데이터가 없습니다.")


#데이터 로딩 및 필터링
with st.spinner("데이터 로딩 중..."):
    time.sleep(1)
df = load_data()
st.success("데이터 로딩 완료!")

if df.empty:
    st.stop()

#날짜 설정
period_q1 = df["DATE"].quantile(0.25)
period_q3 = df["DATE"].quantile(0.75)
start_date = df["DATE"].min()
end_date = df["DATE"].max()

#Streamlit 사이드바 필터링 설정
st.sidebar.header("데이터 조회 옵션 선택")
start_date_input = st.sidebar.date_input(
    "시작날짜", value = period_q1, min_value = start_date, max_value = end_date
)
end_date_input = st.sidebar.date_input(
    "종료날짜", value = period_q3, min_value = start_date, max_value = end_date
)

start_date_input = pd.to_datetime(start_date_input)
end_date_input = pd.to_datetime(end_date_input)

wdays_options = ["All"] + df["WEEKDAY"].unique().tolist()
selected_day_w = st.sidebar.multiselect(
    "요일 선택", options = wdays_options, default = ["All"]
)

city_options = ["All_CITIES"] + df["CITY"].unique().tolist()
selected_city = st.sidebar.multiselect(
    "지역 선택", options = city_options, default = ["All_CITIES"]
)

#필터링 적용
df_select = filter_data(
    df, start_date_input, end_date_input, selected_day_w, selected_city
)
columns_to_display = [
    "DATE",
    "CITY",
    "VISITORS",
    "age",
    "gender",
    "CAMP",
    "PART",
    "P_Ratio",
    "WEEKDAY"
]
filtered_selected_df = df_select[columns_to_display]

#데이터 조회 탭 구성
tab1, tab2, tab3 = st.tabs(["데이터(지표)", "분석", "예측"])

#탭1: 지표
with tab1:
    if st.sidebar.button("데이터 조회"):
        if not filtered_selected_df.empty:
            st.dataframe(filtered_selected_df, use_container_width=True)
            total_visitors = int(filtered_selected_df["VISITORS"].sum())
            total_part = int(filtered_selected_df["PART"].sum())
            part_ratio = (
                (total_part / total_visitors * 100) if total_visitors > 0 else 0
            )
            cities_count = (
                len(selected_city)
                if "All_CITIES" not in selected_city
                else len(filtered_selected_df["CITY"].unique())
            )
            events_count = len(filtered_selected_df["CAMP"].unique())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("방문자수", f"{total_visitors:,}명")
            c2.metric("참여자수", f"{total_part:,}명")
            c3.metric("참여비율", f"{part_ratio:.1f}%")
            c4.metric("참여 도시", cities_count)
            c5.metric("진행 이벤트", f"{events_count}건")

#탭2: 분석
with tab2:
    st.subheader("기본 분석 시각화")
    barplot(filtered_selected_df, x = "CITY", y = "PART")
    col1, col2 = st.columns([1, 1])
    with col1:
        baxplot(filtered_selected_df, x = "WEEKDAY", y = "P_Ratio")
    with col2:
        linechart(filtered_selected_df, x = "DATE", y = "VISITORS")
    scatterplot(filtered_selected_df, x = "VISITORS", y = "PART")
    piechart(filtered_selected_df, x = "CAMP", y = "PART")

#탭3: 예측모델(선형 회귀)
with tab3:
    daily_data = (
        filtered_selected_df.groupby("DATE")
        .agg({"VISITORS": "sum", "PART": "sum"}).reset_index()
    )
    if daily_data.empty:
        st.warning("조회된 데이터가 없습니다.")
    else:
        X = daily_data[["VISITORS"]]
        y = daily_data[["PART"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        daily_data["PREDICTED_PART"] = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        st.write(f"예측 RMSE: {rmse:.2f}")
        fig = px.line(
            daily_data,
            x = "DATE",
            y = ["PART", "PREDICTED_PART"],
            color_discrete_sequence = palette,
            labels = {"value" : "참여자 수"},
            title = "실제 vs 예측 참여자 수",
            markers = True,
        )
        st.plotly_chart(fig, use_container_width = True)
