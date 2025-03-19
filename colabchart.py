import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from io import StringIO

#한글 폰트 설정 (맑은 고딕 적용)
plt.rc("font", family="Malgun Gothic")

#Streamlit 앱 제목
st.title(":bar_chart: 유입 경로 및 지역별 방문자 분석")

#파일 업로드 위젯
st.subheader(":arrows_counterclockwise: CSV 파일 업로드")
file_on = st.file_uploader("온라인 유입 데이터 (recycling_online.csv)", type=["csv"])
file_off = st.file_uploader("오프라인 방문 데이터 (recycling_off.csv)", type=["csv"])

if file_on and file_off:
    #데이터 로드
    df_on = pd.read_csv(file_on)
    df_off = pd.read_csv(file_off)
    
    #유입 경로 데이터
    data = {
        "유입경로": ["직접 유입", "키워드 검색", "블로그", "인스타그램", "유튜브", "배너 광고", "트위터 X", "기타 SNS"],
        "유입수": [55, 80, 6, 9, 93, 88, 0, 62]
    }
    
    #유입 경로별 유입수 파이 차트
    st.subheader(":pushpin: 유입 경로별 유입수 비율")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(data["유입수"], labels=data["유입경로"], autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("pastel"), explode=[0.01] * len(data["유입경로"]), shadow=True)
    ax1.set_title("유입 경로별 유입수 비율", fontsize = 9)
    st.pyplot(fig1)

    #지역별 방문자수 데이터
    st.subheader(":pushpin: 지역별 방문자 수 비율")
    region_visitors = df_off.groupby("지역")["방문자수"].sum()
    
    #지역별 방문자수 파이 차트
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie(region_visitors, labels=region_visitors.index, autopct='%1.1f%%',
            colors=sns.color_palette("pastel"), startangle=140, explode=[0.01] * len(data["유입경로"]), shadow=True)
    ax2.set_title("지역별 방문자 수 비율", fontsize = 9)
    st.pyplot(fig2)

    #데이터프레임 표시
    st.subheader(":bar_chart: 업로드된 데이터 확인")
    st.write("**온라인 유입 데이터**")
    st.dataframe(df_on.head())
    st.write("**오프라인 방문 데이터**")
    st.dataframe(df_off.head())
else:
    st.warning(":open_file_folder: CSV 파일을 업로드하세요!")
