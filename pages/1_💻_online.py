import streamlit as st
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="온라인 데이터 분석", page_icon="🌐")

#한글 폰트 설정 (맑은 고딕 적용)
plt.rc("font", family = "Malgun Gothic")

st.title("🌐 온라인 데이터 분석")

# ✅ MySQL 데이터 로드 함수
def load_data(query):
    secrets = st.secrets["mysql"]
    conn = mysql.connector.connect(
        host=secrets["host"],
        database=secrets["database"],
        user=secrets["user"],
        password=secrets["password"]
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ 데이터 로드
df_on = load_data("SELECT * FROM ontbl;")
df_on['날짜'] = pd.to_datetime(df_on['날짜'])
df_on['year'] = df_on['날짜'].dt.year
df_on['month'] = df_on['날짜'].dt.month

select_year = st.sidebar.radio(
    "2023년과 2024년 중 선택하세요",
    df_on['year'].unique()
)
select_month = st.sidebar.radio(
    "1월 ~ 12월에서 선택하세요",
    df_on['month'].unique()
)

df_on_date = df_on[(df_on['year'] == select_year) & (df_on['month'] == select_month)]
data = {
    "유입경로": ["직접 유입", "키워드 검색", "블로그", "인스타그램", "유튜브", "배너 광고", "트위터 X", "기타 SNS"],
    "유입수": [55, 80, 6, 9, 93, 88, 0, 62]
}
device_counts = df_on_date["디바이스"].value_counts()

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

st.divider()
st.write("📌 **머신러닝 예측을 원하시면 Prediction 페이지를 이용하세요!**")
