from sqlalchemy import create_engine

import matplotlib.pyplot as plt
import pandas as pd
import pymysql
import seaborn as sns
import streamlit as st

#한글 폰트 설정 (맑은 고딕 적용)
plt.rc("font", family="Malgun Gothic")

#Streamlit 페이지 아이콘 설정
st.set_page_config(page_icon = ":chart:")

#Streamlit 앱 제목
st.title(":bar_chart: 유입 경로 및 지역별 방문자 분석")

#DB 연결 정보
DB_HOST = 'localhost'           #호스트이름                          
DB_NAME = 'prjdb'               #데이터베이스 이름
DB_USER = 'root'                #사용자 계정
DB_PASSWORD = '1234'            #비밀번호
DB_PORT = '3306'                #데이터베이스 접속 포트번호

#SQLAlchemy 엔진 생성 (MySQL연결용)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

#데이터 로드 함수(캐싱을 사용하여 최적화)
@st.cache_data
def load_data():
    query_on = "SELECT * FROM ontbl;"       #온라인 유입 데이터 조회
    query_off = "SELECT * FROM offtbl;"     #오프라인 방문 데이터 조회

    #MySQL - prjdb에서 데이터를 읽어 Dataframe으로 변환
    df_on = pd.read_sql(query_on, engine)
    df_off = pd.read_sql(query_off, engine)

    return df_on, df_off

#데이터 로드
st.subheader(":globe_with_meridians: 데이터 불러오는 중...")
df_on, df_off = load_data()
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

#Streamlit UI 구성
#차트 표시
st.subheader("온/오프라인 차트")
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
    sns.lineplot(
        x = '이벤트 종류',
        y = '참여자수',
        data = gender_event,
        hue = "성별",
        palette = {"남": "#0000ff", "여": "#ff0000"},
        ax = ax6,
        marker = "o",
        linewidth = 1
    )
    ax6.set_title("성별에 따른 참여 이벤트", fontsize=9)
    ax6.set_xlabel = ("이벤트 종류")
    ax6.set_ylabel = ("참여자 수")
    ax6.legend(["남", "여"], loc = "upper right")
    plt.xticks(rotation =45)
    st.pyplot(fig6)

st.divider()

#데이터프레임 표시
st.subheader(":floppy_disk:사용된 데이터")
on_data, off_data = st.tabs(["온라인", "오프라인"])

#온라인 탭
with on_data:
    st.write(":floppy_disk:**온라인 유입 데이터**")
    st.dataframe(df_on_date)

#오프라인 탭
with off_data:
    st.write(":floppy_disk:**오프라인 방문 데이터**")
    st.dataframe(df_off_date)