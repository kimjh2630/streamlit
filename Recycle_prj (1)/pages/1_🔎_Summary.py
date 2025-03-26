from matplotlib import rc

import folium
import time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

#메인 페이지 너비 넓게 (가장 처음에 설정해야 함)
st.set_page_config(layout="wide")

with st.spinner("잠시만 기다려 주세요..."):
    time.sleep(1)  #대기 시간 시뮬레이션
st.success("Data Loaded!")

#한글 및 마이너스 깨짐 방지 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#CSV 파일 경로 설정
CSV_FILE_PATH = "https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/"

#온/오프라인 데이터 파일명 설정
on_csv = "recycling_online.csv"
off_csv = "recycling_off.csv"

#CSV데이터 불러오기
on_df = pd.read_csv(CSV_FILE_PATH + on_csv, encoding = "UTF8")
off_df = pd.read_csv(CSV_FILE_PATH + off_csv, encoding = "UTF8")

#지역 리스트 출력
#cities = off_df['지역'].unique().tolist()
#st.write(cities)

#오프라인 전체 데이터
#st.dataframe(off_df, use_container_width=True)

#오프라인 데이터에서 지역별 방문자 수 및 참여자 수 집계
off_data_by_city = (
    off_df.groupby("지역").agg({"방문자수" : "sum", "참여자수" : "sum"}).reset_index()
)
off_data_by_city = off_data_by_city.dropna(subset=["방문자수", "참여자수"])  #NaN 제거


#지역별 참여율 지도 시각화 함수 정의 (ploty 지도)
def map_campain():
    #지역별 위도, 경도 좌표 설정
    coordinates = {
        "인천": (37.4563, 126.7052), "강원": (37.8228, 128.1555), "충북": (36.6351, 127.4915),
        "경기": (37.4138, 127.5183), "울산": (35.5373, 129.3167), "제주": (33.4997, 126.5318),
        "전북": (35.7210, 127.1454), "대전": (36.3504, 127.3845), "대구": (35.8714, 128.6014),
        "서울": (37.5665, 126.9780), "충남": (36.6887, 126.7732), "경남": (35.2345, 128.6880),
        "세종": (36.4805, 127.2898), "경북": (36.1002, 128.6295), "부산": (35.1796, 129.0756),
        "광주": (35.1595, 126.8526), "전남": (34.7802, 126.1322)
    }
    if not off_df.empty:
        #지역별 방문자 수 및 참여자 수 집계
        off_data_by_city = (
            off_df.groupby("지역")
            .agg({"방문자수" : "sum", "참여자수" : "sum"})
            .reset_index()
        )
        #참여율 계산
        off_data_by_city["참여율"] = off_data_by_city.apply(
            lambda row: 
                (row["참여자수"] / row["방문자수"] * 100) if row["방문자수"] > 0 else 0, axis=1
        )

        #지역별 위도 및 경도 추가
        off_data_by_city["위도"] = off_data_by_city["지역"].map(
            lambda x: coordinates.get(x, (None, None))[0]
        )
        off_data_by_city["경도"] = off_data_by_city["지역"].map(
            lambda x: coordinates.get(x, (None, None))[1]
        )

        #유효 데이터 필터링
        valid_data = off_data_by_city.dropna(subset=["위도", "경도"])

        #Plotly 지도 시각화
        fig = px.scatter_geo(
            valid_data,
            lat="위도",
            lon="경도",
            size="참여율",
            color="지역",
            text="지역",
            hover_name="지역",
            size_max=30,
            projection="natural earth",
            title="🗺️ 지역별 참여율 (Plotly 지도)",
        )

        #지도 스타일 설정
        fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        fig.update_layout(
            legend_title_text="지역",
            height=650,
            geo=dict(center={"lat": 36.5, "lon": 127.8}, projection_scale=30),
        )

        #Streamlit에 지도 출력
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("지도에 표시할 데이터가 없습니다.")


#참여율 계산
off_data_by_city["참여율"] = off_data_by_city.apply(
    lambda row: (row["참여자수"] / row["방문자수"] * 100) if row["방문자수"] > 0 else 0, axis=1
)
palette = pc.qualitative.Pastel

#Streamlit탭 생성
tab1, tab2 = st.tabs(["오프라인", "온라인"])

#오프라인 데이터 탭
with tab1:
    st.markdown("**💻지역별 방문자수 데이터**")
    st.dataframe(off_data_by_city, use_container_width=True)  # 오프라인 지역별 데이터
    map_campain()

#온라인 데이터 탭
with tab2:
    st.markdown("**💻온라인 마케팅 데이터**")
    
    #유입경로별 데이터 집계
    on_by_route = (
        on_df.groupby("유입경로").agg(
            {
                "노출수": "sum",
                "유입수": "sum",
                "체류시간(min)": "sum",
                "페이지뷰": "sum",
                "이탈수": "sum",
                "회원가입": "sum",
                "앱 다운": "sum",
                "구독": "sum"
            }
        ).reset_index()
    )

    #Nan값이 포함된 행 제거(유입경로별 데이터)
    on_by_route = on_by_route.dropna(
        subset=[
            "노출수",
            "유입수",
            "체류시간(min)",
            "페이지뷰",
            "이탈수",
            "회원가입",
            "앱 다운",
            "구독",
        ]
    )  #NaN값이 있는 행을 제거하여 데이터 정리

    #"키워드 검색" 유입경로 제외한 데이터 필터링
    on_by_route_ex = on_by_route[on_by_route["유입경로"] != "키워드 검색"]
    st.dataframe(on_by_route, use_container_width = True)

    #유입경로별 유입수 시각화를 위한 Plotly 산점도 생송
    fig = go.Figure()

    #산점도 추가(유입경로별 유입수 시각화)
    fig.add_trace(
        go.Scatter(
            x = on_by_route_ex["유입수"],               #x축 : 유입수
            y = on_by_route_ex["유입경로"],             #y축 : 유입경로
            mode = "markers+text",                      #마커와 텍스트 표시
            name = "유입수 데이터",                     #범례 이름
            text = on_by_route_ex["유입수"],            #데이터 레이블(유입수 표시)
            textposition = "top center",                #텍스트 표시 위치
            marker = dict(color = palette, size = 10)   #마커 스타일 설정
        )
    )

    #그래프 레이아웃 설정
    fig.update_layout(
        title = "유입경로별 유입수 Scatter Plot",   #그래프 제목
        xaxis_title = "유입수",                     #x축 제목
        yaxis_title = "유입경로",                   #y축 제목
        boxmode = "group",                          #그룹화된 박스 플롯
        height = 600,                               #그래프 높이 설정
        showlegend = True                           #범례 표시
    )

    #Streamlit에 그래프 출력
    st.plotly_chart(fig, use_container_width=True)
    
    #구분선 추가
    st.divider()

    #키워드별 전환수 데이터 집계
    act_by_keyword = on_df[on_df["유입경로"] == "키워드 검색"]
    act_by_keyword = (
        act_by_keyword.groupby("키워드").agg(
            {
                "노출수": "sum",
                "유입수": "sum",
                "체류시간(min)": "sum",
                "페이지뷰": "sum",
                "이탈수": "sum",
                "회원가입": "sum",
                "앱 다운": "sum",
                "구독": "sum",
                "전환수": "sum",
            }
        ).reset_index()
    )

    #NaN 제거
    act_by_keyword = act_by_keyword.dropna(
        subset=[
            "노출수",
            "유입수",
            "체류시간(min)",
            "페이지뷰",
            "이탈수",
            "회원가입",
            "앱 다운",
            "구독",
            "전환수",
        ]
    )

    # 바 레이스 차트 생성(키워드별 전환수 시각화)
    fig = go.Figure()

    #각 키워드별 바 차트 추가
    for i, row in act_by_keyword.iterrows():
        fig.add_trace(
            go.Bar(
                x = [row["전환수"]],                        #x축 : 전환수
                y = [row["키워드"]],                        #y축 : 키워드
                name = row["키워드"],                       #범례 설정
                orientation = "h",                          #가로형 바 차트 설정
                marker_color = palette[i % len(palette)]    #마커 색상 설정
            )
        )

    #그래프 레이아웃 설정
    fig.update_layout(
        title = "전환수 바 레이스 차트",      #그래프 제목
        barmode = "stack",                    #바 차트 형태 설정(누적)
        height = 600,                         #그래프 높이 설정
        template = "plotly_white"             #스타일 템플릿 적용
    )

    #Streamlit에 그래프 출력
    st.plotly_chart(fig, use_container_width=True)
