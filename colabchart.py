import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

#한글 폰트 설정 (맑은 고딕 적용)
plt.rc("font", family="Malgun Gothic")

#Streamlit 앱 제목
st.title(":bar_chart: 유입 경로 및 지역별 방문자 분석")

#GitHub에서 데이터 불러오기
@st.cache_data
def load_data():
    url_on = "https://raw.githubusercontent.com/jjjjunn/YH_project/main/recycling_online.csv"
    url_off = "https://raw.githubusercontent.com/jjjjunn/YH_project/main/recycling_off.csv"
    df_on = pd.read_csv(url_on)
    df_off = pd.read_csv(url_off)
    return df_on, df_off

#데이터 로드
st.subheader(":globe_with_meridians: 데이터 불러오는 중...")
df_on, df_off = load_data()
st.success("데이터 로드 완료!")

df_on['날짜'] = pd.to_datetime(df_on['날짜'])
df_on['year'] = df_on['날짜'].dt.year
df_on['month'] = df_on['날짜'].dt.month

df_off['날짜'] = pd.to_datetime(df_off['날짜'])
df_off['year'] = df_off['날짜'].dt.year
df_off['month'] = df_off['날짜'].dt.month

st.sidebar.header("사이드바")
select_year = st.sidebar.radio(
    "2023년과 2024년 중 선택하세요",
    df_on['year'].unique()
)
select_month = st.sidebar.radio(
    "1월 ~ 12월에서 선택하세요",
    df_on['month'].unique()
)

df_on_date = df_on[(df_on['year'] == select_year) & (df_on['month'] == select_month)]
df_off_date = df_off[(df_off['year'] == select_year) & (df_off['month'] == select_month)]

#유입 경로 데이터
data = {
    "유입경로": ["직접 유입", "키워드 검색", "블로그", "인스타그램", "유튜브", "배너 광고", "트위터 X", "기타 SNS"],
    "유입수": [55, 80, 6, 9, 93, 88, 0, 62]
}

#PC, 모바일 숫자로 변환
device_counts = df_on_date["디바이스"].value_counts()

#지역별 방문자수 데이터
region_visitors = df_off_date.groupby("지역")["방문자수"].sum()

#지역별 참여자수 데이터
event_count = df_off_date.groupby("이벤트 종류")["참여자수"].sum()

#차트 표시
st.subheader("온/오프라인 차트")
online, offline = st.tabs(["온라인", "오프라인"])
with online:
    # 유입 경로별 유입수 막대그래프
    st.subheader(":chart: 유입 경로별 유입자 수")
    fig1, ax1 = plt.subplots(figsize = (11, 5))
    sns.barplot(
        x = '유입경로',
        y = '유입수',
        data = df_on_date,
        palette = "pastel",
        ax = ax1
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
    plt.xticks(rotation = 45)
    st.pyplot(fig3)

with offline:
    #지역별 방문자수 막대그래프
    st.subheader(":chart: 지역별 방문자 수")
    fig2, ax2 = plt.subplots(figsize = (7, 7))
    plt.pie(
        x = region_visitors,
        labels = region_visitors.index,
        autopct = "%.2f%%",
        explode = [0.03] * len(region_visitors.index),
        wedgeprops = {"width" : 0.95, "edgecolor" : "w", "linewidth" : 1},
        colors = sns.color_palette("pastel")
    )
    ax2.set_title("지역별 방문자 수", fontsize = 9)
    ax2.set_xlabel = ("지역")
    ax2.set_ylabel = ("방문자 수")
    plt.xticks(rotation = 45)
    st.pyplot(fig2)

    #이벤트별 참여자수 파이차트
    st.subheader(":chart: 지역별 참여자 수")
    fig4, ax4 = plt.subplots(figsize = (7, 7))
    plt.pie(
        x = event_count,
        labels = event_count.index,
        autopct = "%.2f%%",
        explode = [0.03] * len(event_count.index),
        wedgeprops = {"width" : 0.95, "edgecolor" : "w", "linewidth" : 1},
        colors = sns.color_palette("pastel")
    )
    plt.xticks(rotation = 45)
    st.pyplot(fig4)

st.divider()

#데이터프레임 표시
st.subheader("사용된 데이터")
d_online, d_offline = st.tabs(["온라인", "오프라인"])

with d_online:
    st.write("**온라인 유입 데이터**")
    st.dataframe(df_on_date.head(15))

with d_offline:
    st.write("**오프라인 방문 데이터**")
    st.dataframe(df_off_date.head(15))