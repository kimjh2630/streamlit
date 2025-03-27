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
import seaborn as sns
import streamlit as st

#https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
#메인 페이지 너비 넓게 (가장 처음에 설정해야 함)
st.set_page_config(layout = "wide")

with st.spinner("잠시만 기다려 주세요..."):
    time.sleep(1)  #대기 시간 시뮬레이션
st.success("Data Loaded!")

#한글 및 마이너스 기호 깨짐 방지 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#CSV 파일 경로 설정
CSV_FILE_PATH = "https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/"

#온/오프라인 CSV파일 설정
off_csv = "recycling_off.csv"
on_csv = "recycling_online.csv"

#CSV파일을 읽어 DataFrame으로 변환
off_df = pd.read_csv(CSV_FILE_PATH + off_csv, encoding = "UTF8")
on_df = pd.read_csv(CSV_FILE_PATH + on_csv, encoding = "UTF8")

#오프라인 데이터에서 지역별 방문자 수와 참여자 수 합산
off_data_by_city = (
    off_df.groupby("지역").agg({"방문자수" : "sum", "참여자수" : "sum"}).reset_index()
)
#NaN값 제거(방문자수, 참여자수가 결측값인 경우 제거)
off_data_by_city = off_data_by_city.dropna(subset = ["방문자수", "참여자수"])  # NaN 제거

#지역별 참여율 계산 및 시각화 함수 정의(ploty 지도)
def map_campain():
    coordinates = {
        "인천": (37.4563, 126.7052), "강원": (37.8228, 128.1555), "충북": (36.6351, 127.4915),
        "경기": (37.4138, 127.5183), "울산": (35.5373, 129.3167), "제주": (33.4997, 126.5318),
        "전북": (35.7210, 127.1454), "대전": (36.3504, 127.3845), "대구": (35.8714, 128.6014),
        "서울": (37.5665, 126.9780), "충남": (36.6887, 126.7732), "경남": (35.2345, 128.6880),
        "세종": (36.4805, 127.2898), "경북": (36.1002, 128.6295), "부산": (35.1796, 129.0756),
        "광주": (35.1595, 126.8526), "전남": (34.7802, 126.1322)
    }
    if not off_df.empty:
        #지역별 방문자수와 참여자수 합산
        off_data_by_city = (
            off_df.groupby("지역")
            .agg({"방문자수" : "sum", "참여자수" : "sum"}).reset_index()
        )
        #참여율 계산 (참여자수/방문자수) * 100
        off_data_by_city["참여율"] = off_data_by_city.apply(
            lambda row: (
                (row["참여자수"] / row["방문자수"] * 100) if row["방문자수"] > 0 else 0
            ),
            axis = 1,
        )
        #위도, 경보 정보 추가
        off_data_by_city["위도"] = off_data_by_city["지역"].map(
            lambda x: coordinates.get(x, (None, None))[0]
        )
        off_data_by_city["경도"] = off_data_by_city["지역"].map(
            lambda x: coordinates.get(x, (None, None))[1]
        )
        #유효한 위도, 경도 데이터만 선택
        valid_data = off_data_by_city.dropna(subset=["위도", "경도"])

        #Plotly지도로 참여율 시각화
        fig = px.scatter_geo(
            valid_data,
            lat = "위도",
            lon = "경도",
            size = "참여율",                            #참여율을 크기로 설정
            color = "지역",                             #지역별 색상 설정
            text = "지역",                              #툴팁에 지역 표시
            hover_name = "지역",                        #마우스 오버 시 지역 이름 표시
            size_max = 30,                              #최대 마커 크기
            projection = "natural earth",               #자연 지구 지도 사용
            title = "🗺️ 지역별 참여율 (Plotly 지도)",  #제목 설정
        )
        #마커에 테두리 추가
        fig.update_traces(marker = dict(line = dict(width = 1, color = "DarkSlateGrey")))
        #지도 설정(중심 좌표와 크기 설정)
        fig.update_layout(
            legend_title_text = "지역",
            height = 650,
            geo = dict(center = {"lat" : 36.5, "lon" : 127.8}, projection_scale = 30),
        )
        #Streamlit에 Plotly차트 표시
        st.plotly_chart(fig, use_container_width = True)
    else:
        #데이터가 없으면 경고 메시지 표시
        st.warning("지도에 표시할 데이터가 없습니다.")

#오프라인 데이터에서 참여율 계산
off_data_by_city["참여율"] = off_data_by_city.apply(
    lambda row: (row["참여자수"] / row["방문자수"] * 100) if row["방문자수"] > 0 else 0, axis = 1
)

#팔레트 색상 결정(Plotly 색상 팔레트)
palette = pc.qualitative.Pastel

#두 개의 탭 생성
tab1, tab2 = st.tabs(["오프라인", "온라인"])
#탭1 : 오프라인
with tab1:
    #지역별 데이터 확장해서 표시
    with st.expander("**💻지역별 데이터**"):
        #오프라인 데이터에서 지역별로 방문자수와 참여자수를 표시
        st.dataframe(off_data_by_city, use_container_width = True)  
    
    #지역별 참여율 파이차트
    fig = px.pie(
        off_data_by_city,
        names = '지역',                     #파이 차트의 레이블로 지역 사용
        values = '참여율',                  #파이 차트의 크기 값으로 참여율 사용
        title = "🎨지역별 참여율 비교",    #제목 설정
        hole = 0.3,                         #도넛 차트 모양으로 설정
        color = '지역',                     #지역별 색상 지정
        color_discrete_sequence = palette,  #팔레트 색상 설정
        hover_data = '참여율',              #마우스 오버 시 참여율 표시
    )
    fig.update_layout(legend_title_text = '지역', width = 900, height = 700)
    st.plotly_chart(fig, use_container_width = True)
    #요일별 데이터 확장해서 표시
    with st.expander("**💻요일별 데이터**"):
        #날짜 컬럼으르 datetime으로 변환
        off_df["날짜"] = pd.to_datetime(off_df["날짜"], errors = "coerce")
        off_df['요일'] = off_df['날짜'].dt.day_of_week
        #요일 컬럼 생성
        week_mapping = {0 : "월", 1 : "화", 2 : "수", 3 : "목", 4 : "금", 5 : "토", 6 : "일"}
        off_df['요일'] = off_df['요일'].map(week_mapping)
        #요일별로 방문자수, 참여자수 합산
        off_df_by_week = (off_df.groupby('요일').agg({"방문자수" : "sum","참여자수" : "sum"}).reset_index())
        #요일별 데이터 표시
        st.dataframe(off_df_by_week, use_container_width = True)
    
    #시각화
    def create_barplot_by_day_off(data, value_col, title):
        fig = px.bar(
            data_frame = data,
            x = value_col,              #x축 값 (방문자수 또는 참여자수)
            y = "요일",                 #y축 값 (요일)
            orientation = "h",          #수평 방향으로 막대 그래프 그리기
            title = f"<b>{title}</b>",  #제목 설정
            color = "요일",             #요일별 색상 설정
            template = "plotly_white"   #흰색 배경 템플릿 설정
        )

        #x축 범위 설정
        min_value = data[value_col].min()   #최소값
        max_value = data[value_col].max()   #최대값
        fig.update_xaxes(range=[min_value * 0.9, max_value * 1.1])  #범위 설정

        #레이아웃 업데이트
        fig.update_layout(
            plot_bgcolor = "rgba(0,0,0,0)",
            xaxis = dict(showgrid = False)
        )

        return fig
        
    #차트 생성
    col1, col2 = st.columns(2)

    with col1:
        #요일별 방문자수 그래프 생성
        fig_weekday_visit = create_barplot_by_day_off(off_df_by_week, '방문자수', '요일별 방문자수 막대그래프')
        st.plotly_chart(fig_weekday_visit, use_container_width = True)
    
    with col2:
        #요일별 참여자수 그래프 생성
        fig_weekday_part = create_barplot_by_day_off(off_df_by_week, '참여자수', '요일별 참여자수 막대그래프')
        st.plotly_chart(fig_weekday_part, use_container_width = True)
        
    st.divider()

    #월별 데이터 확장해서 표시
    with st.expander("**💻월별 데이터**"):
        #날짜를 월 단위로 변환하여 연도와 월 정보 생성
        off_df['연도'] = off_df['날짜'].dt.year
        off_df['월'] = off_df['날짜'].dt.month
        
        #연도와 월로 그룹화하여 방문자 수와 참여자 수 집계
        off_df_by_month = off_df.groupby(['연도', '월']).agg({"방문자수" : "sum", "참여자수" : "sum"}).reset_index()
        
        #데이터 확인
        st.dataframe(off_df_by_month, use_container_width = True)

    #라인차트 생성
    def create_monthly_by_year_line_chart(data, value_col, title):
        fig = px.line(
            data_frame = data,
            x = "월",                       #x축 값(월)
            y = value_col,                  #y축 값(방문자수 또는 참여자수)
            orientation = "v",              #세로 방향으로 라인차트
            title = f"<b>{title}</b>",      #제목 설정
            color = "연도",                 #연도별 색상 설정
            template = "plotly_white"       #흰색 배경 템플릿 설정
        )

        #레이아웃 업데이트
        fig.update_layout(
            plot_bgcolor = "rgba(0,0,0,0)",
            xaxis = dict(showgrid = False)
        )
        return fig
    
    #차트 생성
    col1, col2 = st.columns(2)

    with col1:
        #월별 방문자수 연도별 비교 라인 차트 생성
        fig_month_visit = create_monthly_by_year_line_chart(off_df_by_month, '방문자수', '월별 방문자수 연도별 비교')
        st.plotly_chart(fig_month_visit, use_container_width = True)

    with col2:
        #월별 참여자수 연도별 비교 라인 차트 생성
        fig_month_part = create_monthly_by_year_line_chart(off_df_by_month, '참여자수', '월별 참여자수 연도별 비교')
        st.plotly_chart(fig_month_part, use_container_width = True)

    #지역별 참여율 시각화 지도 함수 호출
    map_campain()

#탭2 : 온라인
with tab2:
    #온라인 마케팅 데이터 확장해서 표시
    with st.expander("**💻온라인 마케팅 데이터**"):
        #유입경로별 데이터 그룹화하여 여러 수치 합산
        on_by_route = (
            on_df.groupby("유입경로")
            .agg(
                {
                    "노출수" : "sum",
                    "유입수" : "sum",
                    "체류시간(min)" : "sum",
                    "페이지뷰" : "sum",
                    "이탈수" : "sum",
                    "회원가입" : "sum",
                    "앱 다운" : "sum",
                    "구독" : "sum",
                }
            ).reset_index()
        )
        #NaN값이 있는 행 제거
        on_by_route = on_by_route.dropna(
            subset = [
                "노출수",
                "유입수",
                "체류시간(min)",
                "페이지뷰",
                "이탈수",
                "회원가입",
                "앱 다운",
                "구독",
            ]
        ) 
        #키워드 검색 제외
        on_by_route_ex = on_by_route[
            on_by_route["유입경로"] != "키워드 검색"
        ]  
        #유입경로별 집게된 데이터 표시
        st.dataframe(on_by_route, use_container_width = True)

    #요일별 데이터 확장해서 표시
    with st.expander("**💻요일별 데이터**"):
            #날짜 컬럼을 datetime으로 변환
            on_df["날짜"] = pd.to_datetime(on_df["날짜"], errors = "coerce")
            #요일 컬럼 생성
            on_df['요일'] = on_df['날짜'].dt.day_of_week
            #요일 숫자를 한글로 변환
            week_mapping = {0 : "월", 1 : "화", 2 : "수", 3 : "목", 4 : "금", 5 : "토", 6 : "일"}
            on_df['요일'] = on_df['요일'].map(week_mapping)
            #요일별로 유입수, 회원가입, 앱 다운, 구독 합산
            on_df_by_week = (on_df.groupby('요일').agg({"유입수" : "sum","회원가입" : "sum", "앱 다운" : "sum", "구독" : "sum"}).reset_index())
            #요일별 데이터 표시
            st.dataframe(on_df_by_week, use_container_width = True)
        
    #시각화
    def create_barplot_by_day(data, value_col, title):
        fig = px.bar(
            data_frame = data,
            x = value_col,              #x값 (유입수, 회원가입 등)
            y = "요일",                 #y축 값(요일)
            orientation = "h",          #수평 방향으로 막대 그래프 표시
            title = f"<b>{title}</b>",  #제목 설정
            color = "요일",             #요일별 색상 설정
            template = "plotly_white"   #흰색 배경 템플릿 설정
        )

        #x축 범위 설정
        min_value = data[value_col].min()   #최소값
        max_value = data[value_col].max()   #최대값
        fig.update_xaxes(range = [min_value * 0.9, max_value * 1.1])  #범위 설정

        #레이아웃 업데이트
        fig.update_layout(
            plot_bgcolor = "rgba(0,0,0,0)",
            xaxis = dict(showgrid = False)
        )

        return fig
        
    #차트 생성
    col1, col2 = st.columns(2)

    with col1:
        #요일별 유입수 막대 그래프 생성
        fig_weekday_in = create_barplot_by_day(on_df_by_week, '유입수', '요일별 유입수 막대그래프')
        st.plotly_chart(fig_weekday_in, use_container_width = True)
    
    with col2:
        #'회원가입', '앱 다운', '구독'의 합을 '전환수'로 계산
        on_df_by_week['전환수'] = on_df_by_week[['회원가입', '앱 다운', '구독']].sum(axis = 1)
        #요일별 전환수 막대그래프 생성
        fig_weekday_act = create_barplot_by_day(on_df_by_week, '전환수', '요일별 전환수 막대그래프')
        st.plotly_chart(fig_weekday_act, use_container_width = True)
        
    st.divider()

    #월별 데이터 확장해서 표시
    with st.expander("**💻월별 데이터**"):
        #날짜를 월 단위로 변환하여 연도와 월 정보 생성
        on_df['연도'] = on_df['날짜'].dt.year
        on_df['월'] = on_df['날짜'].dt.month
        
        #연도와 월로 그룹화하여 방문자 수와 참여자 수 집계
        on_df_by_month = on_df.groupby(['연도', '월']).agg({"유입수" : "sum","회원가입" : "sum", "앱 다운" : "sum", "구독" : "sum"}).reset_index()
        
        #데이터 확인
        st.dataframe(on_df_by_month, use_container_width = True)

    #라인차트 생성
    def create_monthly_by_year_line_chart_on(data, value_col, title):
        fig = px.line(
            data_frame = data,
            x = "월",                   #x축 값 (월)
            y = value_col,              #y축 값 (유입수, 회원가입 등)
            orientation = "v",          #세로 방향으로 라인 차트 그리기
            title = f"<b>{title}</b>",  #제목 설정
            color = "연도",             #연도별 색상 설정
            template = "plotly_white"   #흰색 배경 템플릿 설정
        )

        #레이아웃 업데이트
        fig.update_layout(
            plot_bgcolor = "rgba(0,0,0,0)",
            xaxis = dict(showgrid = False)
        )
        return fig
    
    #차트 생성
    col1, col2 = st.columns(2)

    with col1:
        #월별 유입수에 대한 연도별 비교 라인 차트 생성
        fig_month_in = create_monthly_by_year_line_chart_on(on_df_by_month, '유입수', '월별 유입수 연도별 비교')
        st.plotly_chart(fig_month_in, use_container_width = True)

    with col2:
        #'회원가입', '앱 다운', '구독'의 합을 '전환수'로 계산
        on_df_by_month['전환수'] = on_df_by_month[['회원가입', '앱 다운', '구독']].sum(axis = 1)
        #월별 전환수에 대한 연도별 비교 라인 차트 생성
        fig_month_act = create_monthly_by_year_line_chart_on(on_df_by_month, '전환수', '월별 전환수 연도별 비교')
        st.plotly_chart(fig_month_act, use_container_width = True)

    c1, c2 = st.columns(2)
    with c1:
        #유입수와 유입경로의 관계를 산점도로 시각화
        fig = go.Figure()

        #유입수에 대한 산점도 추가
        fig.add_trace(
            go.Scatter(
            x = on_by_route_ex["유입수"],               #x축 : 유입수
            y = on_by_route_ex["유입경로"],             #y축 : 유입경로
            mode = "markers+text",                      #마커와 텍스트 표시
            name = "유입수 데이터",                     #데이터 이름
            text = on_by_route_ex["유입수"],            #텍스트로 유입수 값 표시
            textposition = "top center",                #텍스트 위치 설정
            marker = dict(color = palette, size = 10),  #마커 색상 및 크기 설정
            )
        )

        #레이아웃 설정
        fig.update_layout(
            title = "유입경로별 유입수 Scatter Plot",
            xaxis_title = "유입수",
            yaxis_title = "유입경로",
            boxmode = "group",  #그룹화된 박스 플롯
            height = 600,
            showlegend = True,
        )

        #결과 출력
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        #전환 수 합산 열 추가
        on_by_route_ex['전환수'] = on_by_route_ex[['회원가입', '앱 다운', '구독']].sum(axis = 1)

        #전환수와 유입경로의 관계를 산점도로 시각화
        fig = go.Figure()

        #전환수에 대한 산점도 추가
        fig.add_trace(
            go.Scatter(
                x = on_by_route_ex["전환수"],               #x축 : 전환수
                y = on_by_route_ex["유입경로"],             #y축 : 유입경로
                mode = "markers+text",                      #마커와 텍스트 표시
                name = "전환수 데이터",                     #데이터 이름
                text = on_by_route_ex["전환수"],            #텍스트로 전환수 값 표시
                textposition = "top center",                #텍스트 위치 설정
                marker = dict(color=palette, size = 10),    #마커 색상 및 크기 설정
            )
        )

        #레이아웃 설정
        fig.update_layout(
            title = "유입경로별 전환수 Scatter Plot",
            xaxis_title = "전환수",
            yaxis_title = "유입경로",
            boxmode = "group",  #그룹화된 박스 플롯
            height = 600,
            showlegend = True,
        )

        #결과 출력
        st.plotly_chart(fig, use_container_width = True)
    #키워드 광고 제외 표시
    st.write(":red[※키워드광고 제외]")
    st.divider()

    #키워드별 전환수
    act_by_keyword = on_df[on_df["유입경로"] == "키워드 검색"]
    act_by_keyword = (
        act_by_keyword.groupby("키워드")
        .agg(
            {
                "노출수" : "sum",
                "유입수" : "sum",
                "체류시간(min)" : "sum",
                "페이지뷰" : "sum",
                "이탈수" : "sum",
                "회원가입" : "sum",
                "앱 다운" : "sum",
                "구독" : "sum",
                "전환수" : "sum",
            }
        ).reset_index()
    )
    act_by_keyword = act_by_keyword.dropna(
        subset = [
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
    )  #NaN 제거

    #수평 막대 차트 생성
    fig = go.Figure()

    for i, row in act_by_keyword.iterrows():
        fig.add_trace(
            go.Bar(
                x = [row["전환수"]],
                y = [row["키워드"]],
                name = row["키워드"],
                orientation = "h",
                marker_color = palette[i % len(palette)],
            )
        )
        
    #x축 범위 설정
    min_value = act_by_keyword["전환수"].min()    #최소값
    max_value = act_by_keyword["전환수"].max()    #최대값
    fig.update_xaxes(range = [min_value * 0.9, max_value * 1.1])  #범위 설정

    fig.update_layout(
        title = "광고 키워드별 전환수 그래프",
        barmode = "stack",
        height = 600,
        template = "plotly_white",
    )

    st.plotly_chart(fig, use_container_width = True)