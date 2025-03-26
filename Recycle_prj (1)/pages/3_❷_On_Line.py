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

#Streamlit 페이지 기본 설정
st.set_page_config(layout = "wide")

#한글 폰트 및 마이너스 기호 정상 표시 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터 파일 경로 (GitHub에서 CSV파일 불러오기)
CSV_FILE_PATH = "https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/recycling_online.csv"

#그래프에서 사용할 색상 팔레트 설정
palette = px.colors.qualitative.Set2

# 데이터 로딩 함수 (캐싱 활용)
@st.cache_data
def load_data():
    try:
        #CSV파일 불러오기 및 결측치 처리
        df = pd.read_csv(CSV_FILE_PATH, encoding = "UTF8").fillna(0)
        df.replace([np.inf, -np.inf], np.nan, inplace = True)
        df.fillna(0, inplace = True)

        #컬럼명 변경(한글 > 영문)
        df = df.rename(
            columns = {
                "날짜": "DATE",
                "디바이스": "Device",
                "유입경로": "Route",
                "키워드": "KeyWord",
                "노출수": "Exposure",
                "유입수": "Inflow",
                "유입률(%)": "In_rate",
                "체류시간(min)": "Stay_min",
                "평균체류시간(min)": "M_Stay_min",
                "페이지뷰": "P_view",
                "평균페이지뷰": "M_P_view",
                "이탈수": "Exit",
                "이탈률(%)": "Exit_R",
                "회원가입": "Join",
                "전환율(가입)": "Join_R",
                "앱 다운": "Down",
                "전환율(앱)": "Down_R",
                "구독": "Scribe",
                "전환율(구독)": "Scr_R",
                "전환수": "Action",
                "전환율(%)": "Act_R"
            }
        )

        #날짜 컬럼을 datetime형식으로 변환
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"])     #변환 실패한 행 제거

        #요일 컬럼 추가
        df["WEEKDAY"] = df["DATE"].dt.day_of_week
        week_mapping = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        df["WEEKDAY"] = df["WEEKDAY"].map(week_mapping)

        return df

    except Exception as e:
        st.error(f"CSV 파일 로딩 중 오류 발생: {e}")
        return pd.DataFrame()


#데이터 필터링 함수
@st.cache_data
def filter_data(df, start_date, end_date, selected_day, selected_dv):
    df_filtered = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    #요일 필터링(전체 선택이 아니면 특정 요일만)
    if selected_day and "All" not in selected_day:
        df_filtered = df_filtered[df_filtered["WEEKDAY"].isin(selected_day)]
    
    #디바이스 필터링(전체 선택이 아니면 특정 디바이스만)
    if "All_DV" not in selected_dv:
        df_filtered = df_filtered[df_filtered["Device"].isin(selected_dv)]
    return df_filtered


#시각화 함수들
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
        fig.update_layout(legend_title_text = x)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("조회된 데이터가 없습니다.")


def baxplot(df, x, y):
    if not df.empty:
        fig = px.box(
            df,
            x = x,
            y = y,
            title = f"{x}별 {y} 분포",
            points = "outliers",
            color = x,
            color_discrete_sequence = palette,
            hover_data = [x, y]
        )
        fig.update_layout(legend_title_text = x)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("조회된 데이터가 없습니다.")


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
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("조회된 데이터가 없습니다.")


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
        fig.update_layout(legend_title_text = x)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("조회된 데이터가 없습니다.")


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


#Streamlit 실행 시 시작 로딩
with st.spinner("자동 데이터 로딩 중..."):
    time.sleep(1)       #로딩 효과를 위한 딜레이
df = load_data()
st.success("데이터 로딩 완료!")

#데이터가 없으면 실행 중단
if df.empty:
    st.stop()

#날짜 범위 설정
period_q1 = df["DATE"].quantile(0.25)   #데이터의 25% 지점
period_q3 = df["DATE"].quantile(0.75)   #데이터의 75% 지점
start_date = df["DATE"].min()
end_date = df["DATE"].max()

#사이드바 필터 설정
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

dv_options = ["All_DV"] + df["Device"].unique().tolist()
selected_dv = st.sidebar.multiselect(
    "디바이스 선택", options = dv_options, default = ["All_DV"]
)

#필터링 적용
df_select = filter_data(
    df, start_date_input, end_date_input, selected_day_w, selected_dv
)
columns_to_display = [
    "DATE",
    "WEEKDAY",
    "Device",
    "Route",
    "KeyWord",
    "Exposure",
    "Inflow",
    "In_rate",
    "Stay_min",
    "M_Stay_min",
    "P_view",
    "M_P_view",
    "Exit",
    "Exit_R",
    "Join",
    "Join_R",
    "Down",
    "Down_R",
    "Scribe",
    "Scr_R",
    "Action",
    "Act_R"
]
filtered_selected_df = df_select[columns_to_display]

#탭 생성
tab1, tab2, tab3 = st.tabs(["데이터(지표)", "분석", "예측"])

#탭1 : 지표
with tab1:
    if st.sidebar.button("데이터 조회"):
        if not filtered_selected_df.empty:
            st.dataframe(filtered_selected_df, use_container_width = True)
            total_exposure = int(filtered_selected_df["Exposure"].sum())
            total_in = int(filtered_selected_df["Inflow"].sum())
            in_ratio = (total_in / total_exposure * 100) if total_exposure > 0 else 0
            total_stay = int(filtered_selected_df["Stay_min"].sum())
            total_exit = int(filtered_selected_df["Exit"].sum())
            exit_ratio = (total_exit / total_in * 100) if total_in > 0 else 0
            total_act = int(filtered_selected_df["Action"].sum())
            act_ratio = (total_act / total_in * 100) if total_in > 0 else 0
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
            c1.metric("노출수", f"{total_exposure:,}")
            c2.metric("유입수", f"{total_in:,}")
            c3.metric("유입비율", f"{in_ratio:.1f}%")
            c4.metric("체류시간", f"{total_stay:,}분")
            c5.metric("이탈수", f"{total_exit:,}")
            c6.metric("이탈률", f"{exit_ratio:.1f}%")
            c7.metric("전환수", f"{total_act:,}")
            c8.metric("전환율", f"{act_ratio:.1f}%")

#탭2 : 분석
with tab2:
    st.subheader("기본 분석 시각화")

    #디바이스별 유입수 비교(막대 그래프)
    barplot(filtered_selected_df, x = "Device", y = "Inflow")
    col1, col2 = st.columns([1, 1])
    #유입 경로별 전환 수 분포(박스 플롯)
    with col1:
        baxplot(filtered_selected_df, x = "Route", y = "Action")
    #날짜별 유입률 변화(선 그래프)
    with col2:
        linechart(filtered_selected_df, x = "DATE", y = "In_rate")

    #유입수와 전환수 상관관계(산점도)
    scatterplot(filtered_selected_df, x = "Inflow", y = "Action")

    #유입 경로별 환수 비율(파이 차트)
    piechart(filtered_selected_df, x = "Route", y = "Action")

#탭3 : 예측
with tab3:
    daily_data = (
        filtered_selected_df.groupby("DATE")
        .agg({"Inflow": "sum", "Action": "sum"}).reset_index()
    )
    if daily_data.empty:
        st.warning("조회된 데이터가 없습니다.")
    else:
        #데이터 분리 및 학습
        X = daily_data[["Inflow"]]
        y = daily_data[["Action"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        #예측 수행
        daily_data["PREDICTED_Action"] = model.predict(X)

        #모델 성능 평가
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        st.write(f"**예측 RMSE: {rmse:.2f}**")

        #실제 VS 예측 그래프
        fig = px.line(
            daily_data,
            x = "DATE",
            y = ["Action", "PREDICTED_Action"],
            labels = {"value" : "전환수"},
            title = "실제 vs 예측 전환수",
            markers = True
        )
        st.plotly_chart(fig, use_container_width = True)
