from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

#Streamlit 설정
st.set_page_config(layout = "wide")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#CSV 파일 경로
CSV_FILE_PATH = "https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/recycling_online.csv"

#색상 팔레트 설정
palette = px.colors.qualitative.Set2

#데이터 로딩 함수 > 성능 최적화 및 반복 실행 방지
@st.cache_data
def load_data():
    try:
        #CSV파일 로드 후 결측값 처리 및 컬럼명 변경
        df = pd.read_csv(CSV_FILE_PATH, encoding = "UTF8").fillna(0)
        df.replace([np.inf, -np.inf], np.nan, inplace = True)
        df.fillna(0, inplace = True)

        #컬럼명을 사용자 친화적인 이름으로 변경
        df = df.rename(
            columns = {
                "날짜" : "DATE",
                "디바이스" : "Device",
                "유입경로" : "Route",
                "키워드" : "KeyWord",
                "노출수" : "Exposure",
                "유입수" : "Inflow",
                "유입률(%)" : "In_rate",
                "체류시간(min)" : "Stay_min",
                "평균체류시간(min)" : "M_Stay_min",
                "페이지뷰" : "P_view",
                "평균페이지뷰" : "M_P_view",
                "이탈수" : "Exit",
                "이탈률(%)" : "Exit_R",
                "회원가입" : "Join",
                "전환율(가입)" : "Join_R",
                "앱 다운" : "Down",
                "전환율(앱)" : "Down_R",
                "구독" : "Scribe",
                "전환율(구독)" : "Scr_R",
                "전환수" : "Action",
                "전환율(%)" : "Act_R",
            }
        )

        #날짜 컬럼을 날짜 형식으로 변환
        df["DATE"] = pd.to_datetime(df["DATE"], errors = "coerce")
        df = df.dropna(subset = ["DATE"])

        #요일 정보 추가
        df["WEEKDAY"] = df["DATE"].dt.day_of_week
        week_mapping = {0 : "월", 1 : "화", 2 : "수", 3 : "목", 4 : "금", 5 : "토", 6 : "일"}
        df["WEEKDAY"] = df["WEEKDAY"].map(week_mapping)

        return df

    except Exception as e:
        #오류 발생 시 메시지 출력
        st.error(f"CSV 파일 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

#필터링 함수 > 성능 최적화 및 반복 실행 방지
@st.cache_data
def filter_data(df, start_date, end_date, selected_day, selected_dv):
    #주어진 기간에 맞는 데이터 필터링
    df_filtered = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]
    #선택된 요일에 맞게 데이터 필터링
    if selected_day and "All" not in selected_day:
        df_filtered = df_filtered[df_filtered["WEEKDAY"].isin(selected_day)]
    #선택된 디바이스에 맞게 필터링
    if "All_DV" not in selected_dv:
        df_filtered = df_filtered[df_filtered["Device"].isin(selected_dv)]
    return df_filtered

#시각화 함수
#막대그래프 : 디바이스별 평균 유입수 비교
def barplot(df, x, y):
    if not df.empty:
        #디바이스별 평균 유입수 계산
        df_grouped = df.groupby(x)[y].mean().reset_index()
        fig = px.bar(
            df_grouped,
            x = x,
            y = y,
            color = x,
            color_discrete_sequence = palette,
            title = "디바이스별 평균 유입수 비교",
            text_auto = True,
            hover_data = [y],
        )
        fig.update_layout(legend_title_text = x)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("조회된 데이터가 없습니다.")

#박스플롯 : 유입경로별 전환 분포
def baxplot(df, x, y):
    if not df.empty:
        fig = px.box(
            df,
            x = x,
            y = y,
            title = "유입채널별 전환 분포",
            points = "outliers",        #이상치 표시
            color = x,
            color_discrete_sequence = palette,
            hover_data = [x, y],
        )
        fig.update_layout(legend_title_text = x)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("조회된 데이터가 없습니다.")

#라인차트 : 날짜별 평균 유입률 비교
def linechart(df, x, y):
    if not df.empty:
        df_grouped = df.groupby(x)[y].mean().reset_index()
        fig = px.line(
            df_grouped,
            x = x,
            y = y,
            markers = True,
            color_discrete_sequence = palette,
            title = "날짜별 평균 유입률 비교",
            hover_data = [y],
        )
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("조회된 데이터가 없습니다.")

#산점도 : 유입수와 전환수 상관관계 분석
def scatterplot(df, x, y):
    if not df.empty:
        fig = px.scatter(
            df,
            x = x,
            y = y,
            color = x,
            color_discrete_sequence = palette,
            title = "유입-전환 상관관계",
            hover_data = [x, y],
        )
        fig.update_layout(legend_title_text = x)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("조회된 데이터가 없습니다.")

#파이차트 : 유입경로별 전환율 비교
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
            title = "유입채널별 전환율 비교",
            hole = 0.3,
            color = x,
            color_discrete_sequence = palette,
            hover_data = [y],
        )
        fig.update_layout(legend_title_text = x, width = 900, height = 700)
        st.plotly_chart(fig, use_container_width = True)
    else:
        st.warning("⚠️ 선택한 조건에 해당하는 데이터가 없습니다.")

#데이터 로딩
with st.spinner("자동 데이터 로딩 중..."):
    time.sleep(1)
df = load_data()
st.success("데이터 로딩 완료!")

#데이터가 비어있으면 중지
if df.empty:
    st.stop()

#날짜 설정
period_q1 = df["DATE"].quantile(0.25)
period_q3 = df["DATE"].quantile(0.75)
start_date = df["DATE"].min()
end_date = df["DATE"].max()

#사이드바에서 필터링 옵션 설정
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

#필터링된 데이터 가져오기
df_select = filter_data(
    df, start_date_input, end_date_input, selected_day_w, selected_dv
)
#선택된 컬럼만 표시
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
    "Act_R",
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
            c4.metric("체류시간(분)", f"{total_stay:,.0f}")
            c5.metric("이탈수", f"{total_exit:,}")
            c6.metric("이탈률", f"{exit_ratio:.1f}%")
            c7.metric("전환수", f"{total_act:,}")
            c8.metric("전환율", f"{act_ratio:.1f}%")

#탭2 : 분석
with tab2:
    st.subheader("기본 분석 시각화")
    barplot(filtered_selected_df, x = "Device", y = "Inflow")
    col1, col2 = st.columns([1, 1])
    with col1:
        baxplot(filtered_selected_df, x = "Route", y = "Action")
    with col2:
        linechart(filtered_selected_df, x = "DATE", y = "In_rate")
    scatterplot(filtered_selected_df, x = "Inflow", y = "Action")
    piechart(filtered_selected_df, x = "Route", y = "Action")

#탭3 : 예측
with tab3:
    daily_data = (
        filtered_selected_df.groupby("DATE").agg({"Inflow" : "sum", "Action" : "sum"}).reset_index()
    )
    if daily_data.empty:
        st.warning("조회된 데이터가 없습니다.")
    else:
        X = daily_data[["Inflow"]]
        y = daily_data[["Action"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        daily_data["PREDICTED_Action"] = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        st.write(f"**예측 RMSE: {rmse:.2f}**")
        st.write(f"모델이 예측한 전환수와 실제 수치 사이의 평균차가 :blue[{rmse:.2f}]입니다.")

        fig = px.line(
            daily_data,
            x = "DATE",
            y = ["Action", "PREDICTED_Action"],
            labels = {"value": "전환수"},
            title = "실제 vs 예측 전환수",
            markers = True,
        )
        st.plotly_chart(fig, use_container_width = True)