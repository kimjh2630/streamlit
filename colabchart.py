import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트 설정 (맑은 고딕 적용)
plt.rc("font", family="Malgun Gothic")

# Streamlit 앱 제목
st.title(":bar_chart: 유입 경로 및 지역별 방문자 분석")

# GitHub에서 데이터 불러오기
@st.cache_data
def load_data():
    url_on = "https://raw.githubusercontent.com/kimjh2630/streamlit/main/recycling_online.csv"
    url_off = "https://raw.githubusercontent.com/kimjh2630/streamlit/main/recycling_off.csv"
    df_on = pd.read_csv(url_on)
    df_off = pd.read_csv(url_off)
    return df_on, df_off

# 데이터 로드
st.subheader(":globe_with_meridians: 데이터 불러오는 중...")
df_on, df_off = load_data()
st.success("데이터 로드 완료!")

# 유입 경로 데이터
data = {
    "유입경로": ["직접 유입", "키워드 검색", "블로그", "인스타그램", "유튜브", "배너 광고", "트위터 X", "기타 SNS"],
    "유입수": [55, 80, 6, 9, 93, 88, 0, 62]
}

# 유입 경로별 유입수 파이 차트
st.subheader(":pushpin: 유입 경로별 유입자 수 비율")
fig1, ax1 = plt.subplots(figsize=(11, 6))
sns.barplot(
    x = '유입경로',
    y = '유입수',
    data = df_on,
    palette = "pastel",
    ax = ax1
    )
ax1.set_title("유입 경로별 유입수 비율", fontsize=9)
ax1.set_xlabel = ("유입 경로")
ax1.set_ylabel = ("유입수")
plt.xticks(rotation =45)
st.pyplot(fig1)

# 지역별 방문자수 데이터
st.subheader(":pushpin: 지역별 방문자 수 비율")
region_visitors = df_off.groupby("지역")["방문자수"].sum()

# 지역별 방문자수 파이 차트
fig2, ax2 = plt.subplots(figsize=(11, 6))
sns.barplot(
    x = region_visitors.index,
    y = region_visitors.values,
    palette = "pastel",
    ax = ax2
    )
ax2.set_title("지역별 방문자 수 비율", fontsize=9)
ax2.set_xlabel = ("지역")
ax2.set_ylabel = ("방문자 수")
plt.xticks(rotation = 45)
st.pyplot(fig2)

# 데이터프레임 표시
st.subheader(":bar_chart: 업로드된 데이터 확인")
online, offline = st.tabs(["온라인", "오프라인"])

with online:
    st.write("**온라인 유입 데이터**")
    st.dataframe(df_on.head())

with offline:
    st.write("**오프라인 방문 데이터**")
    st.dataframe(df_off.head())