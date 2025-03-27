import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mysql.connector

st.set_page_config(page_title="방문자 수 예측", page_icon="🤖")

#한글 폰트 설정 (맑은 고딕 적용)
plt.rc("font", family = "Malgun Gothic")

st.title("🤖 머신러닝 기반 방문자 수 예측")

@st.cache_data
def on_load_data():
    url_on = "https://raw.githubusercontent.com/jjjjunn/YH_project/main/recycling_online.csv"
    df_on = pd.read_csv(url_on, encoding="UTF8").fillna(0)
    df_on.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_on.fillna(0, inplace=True)
    return df_on

@st.cache_data
def off_load_data():
    url_off = "https://raw.githubusercontent.com/jjjjunn/YH_project/main/recycling_off.csv"
    df_off = pd.read_csv(url_off, encoding="UTF8")
    df_off.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_off.dropna(subset=["날짜"], inplace=True)
    return df_off

on_machine, off_machine = st.tabs(["온라인", "오프라인"])
with on_machine:
    # ✅ 데이터 로드
    df_on = on_load_data()
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
        
    #온라인 데이터 복사 및 원-핫 인코딩
    df_ml_on = df_on.copy()
    df_ml_on = pd.get_dummies(df_ml_on, columns = ["디바이스", "유입경로"])        

        

    #체류시간 및 원-핫 인코딩된 디바이스, 유입경로 및 타겟 변수 설정
    features = ["체류시간(min)"] + [col for col in df_ml_on.columns if "디바이스_" in col or "유입경로_" in col]
    target = "전환율(가입)"

    if st.button("온라인 예측"):
        #입력(X), 출력(y) 데이터 정의
        X = df_ml_on[features]
        y = df_ml_on[target]

        #학습 데이터와 테스트 데이터 분할(학습 데이터 : 80%, 테스트 데이터 : 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        #결측값 처리
        y_train.fillna(y_train.median(), inplace = True)

        #랜덤 포레스트 회귀 모델 생성 및 학습
        on_model = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs=-1)
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
            linestyle = "-",
            label="예측 vs 실제",
            errorbar = None
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

with off_machine:
    # ✅ 데이터 로드
    df_off = off_load_data()

    # ✅ 학습 데이터 준비
    df_ml_off = df_off.groupby(["날짜", "지역"])["방문자수"].sum().reset_index()
    df_ml_off["날짜"] = pd.to_datetime(df_ml_off["날짜"])
    df_ml_off["year"] = df_ml_off["날짜"].dt.year
    df_ml_off["month"] = df_ml_off["날짜"].dt.month
    df_ml_off["day"] = df_ml_off["날짜"].dt.day
    df_ml_off["day_of_week"] = df_ml_off["날짜"].dt.weekday

    # ✅ 사용자 선택
    select_region = st.selectbox("지역을 선택하세요.", df_ml_off["지역"].unique())

    # ✅ 데이터 필터링 및 모델 학습
    df_region = df_ml_off[df_ml_off["지역"] == select_region]
    features = ["year", "month", "day", "day_of_week"]
    X = df_region[features]
    y = df_region["방문자수"]

    if st.button("오프라인 예측"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        off_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        off_model.fit(X_train, y_train)

        # ✅ 향후 12개월 예측
        future_dates = pd.date_range(start=df_region["날짜"].max(), periods=12, freq="ME")
        future_df = pd.DataFrame({"year": future_dates.year, "month": future_dates.month, "day": future_dates.day, "day_of_week": future_dates.weekday})
        future_pred = off_model.predict(future_df)
        future_df["예측 방문자 수"] = future_pred
        future_df["날짜"] = future_dates

        # ✅ 예측 결과 시각화
        st.subheader(":chart: 향후 12개월 방문자 수 예측")
        fig_ml_off, ax_ml_off = plt.subplots(figsize=(9, 6))
        ax_ml_off.plot(future_df.index, future_df["예측 방문자 수"], marker="o", linestyle="-", color="red", label="예측 방문자 수")
        ax_ml_off.set_title(f"{select_region} 방문자 수 예측")
        ax_ml_off.set_xlabel("날짜")
        ax_ml_off.set_ylabel("방문자 수")
        ax_ml_off.legend()
        st.pyplot(fig_ml_off)

        #✅예측 방문자 수 데이터프레임 출력
        #날짜 데이터를 "YYYY-MM-01" 형식으로 변환
        future_df["날짜"] = pd.to_datetime(future_df["날짜"]).apply(lambda x: x.replace(day = 1))
        future_df["날짜"] = future_df["날짜"] + pd.DateOffset(months=1)
        
        #방문자 수를 정수로 변환 후 "명" 추가
        future_df["예측 방문자 수"] = future_df["예측 방문자 수"].astype(int).astype(str) + "명"
        st.write(future_df[["날짜","예측 방문자 수"]])
