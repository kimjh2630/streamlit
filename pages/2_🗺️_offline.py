import streamlit as st
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="오프라인 데이터 분석", page_icon="🏢")

#한글 폰트 설정 (맑은 고딕 적용)
plt.rc("font", family = "Malgun Gothic")

st.title("🏢 오프라인 데이터 분석")

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
df_off = load_data("SELECT * FROM offtbl;")
df_off['날짜'] = pd.to_datetime(df_off['날짜'])
df_off['year'] = df_off['날짜'].dt.year
df_off['month'] = df_off['날짜'].dt.month

select_year = st.sidebar.radio(
    "2023년과 2024년 중 선택하세요",
    df_off['year'].unique()
)
select_month = st.sidebar.radio(
    "1월 ~ 12월에서 선택하세요",
    df_off['month'].unique()
)

df_off_date = df_off[(df_off['year'] == select_year) & (df_off['month'] == select_month)]

#✅지역별 방문자수 데이터
region_visitors = df_off_date.groupby("지역")["방문자수"].sum()

#✅지역별 참여자수 데이터
region_event = df_off_date.groupby("이벤트 종류")["참여자수"].sum()

#✅성별에 따른 참여 이벤트 데이터
gender_event = df_off_date.groupby(["이벤트 종류", "성별"])["참여자수"].sum().reset_index()

# ✅ 지역별 방문자 수 시각화
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

st.divider()
st.write("📌 **방문자 수 예측을 원하시면 Prediction 페이지를 이용하세요!**")
