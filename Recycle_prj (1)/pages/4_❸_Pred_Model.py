from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import folium
import time  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st

#[파스텔톤 Hex Codes]
#파스텔 블루 : #ADD8E6
#파스텔 그린 : #77DD77
#파스텔 퍼플 : #B19CD9
#파스텔 옐로우 : #FFFACD
#파스텔 피치 : #FFDAB9
#파스텔 민트 : #BDFCC9
#파스텔 라벤더 : #E6E6FA
#파스텔 노란색 : #FFF44F
#파스텔 그린 : #B2FBA5

#Streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
#메인 페이지 너비 넓게(가장 처음에 설정해야 함)
st.set_page_config(layout = "wide") 

with st.spinner("잠시만 기다려 주세요..."):
    time.sleep(1)  #대기 시간 시뮬레이션
st.success("Data Loaded!")

#한글 및 마이너스 깨짐 방지(클라우드 배포 시 필수 설정)
plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

#CSV 파일 경로 설정
CSV_FILE_PATH = 'https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/'

#회원 데이터 로드(CSV파일을 pandas로 불러옴)
memeber_df = pd.read_csv(CSV_FILE_PATH + 'members_data.csv')

#온/오프라인 데이터 로드
@st.cache_data
def on_load_data():
    #온라인 데이터 파일 불러오기 및 결측치 처리
    df_on = pd.read_csv(CSV_FILE_PATH + 'recycling_online.csv', encoding = "UTF8").fillna(0)
    df_on.replace([np.inf, -np.inf], np.nan, inplace = True)
    df_on.fillna(0, inplace = True)
    return df_on

@st.cache_data
def off_load_data():
    #오프라인 데이터 파일 불러오기 및 결측치 처리
    df_off = pd.read_csv(CSV_FILE_PATH + 'recycling_off.csv', encoding = "UTF8")
    df_off.replace([np.inf, -np.inf], np.nan, inplace = True)
    df_off.dropna(subset=["날짜"], inplace = True)  #날짜 결측치가 있는 행은 제거
    return df_off

#온/오프라인 데이터 로드
df_on = on_load_data()
df_off = off_load_data()

#회원 데이터 컬럼 이름 변경(영어 > 한국어)
print_df = memeber_df.rename(columns = {
     "age" : "나이",
     "gender" : "성별",
     "marriage" : "혼인여부",
     "city" : "도시",
     "channel" : "가입경로",
     "before_ev" : "참여_전",
     "part_ev" : "참여이벤트",
     "after_ev" : "참여_후"
})

#데이터값 변경(숫자로된 값을 범주형 값으로 매핑)
print_df['성별'] = print_df['성별'].map({0 :'남자', 1 :'여자'})
print_df['혼인여부'] = print_df['혼인여부'].map({0 :'미혼', 1 :'기혼'})
print_df['도시'] = print_df['도시'].map({
    0 : '부산', 1 : '대구', 2 : '인천', 3 : '대전', 
    4 : '울산', 5 : '광주', 6 : '서울', 7 : '경기', 
    8 : '강원', 9 : '충북', 10 : '충남', 11 : '전북', 
    12 : '전남', 13 : '경북', 14 : '경남', 15 : '세종', 16 : '제주'})
print_df['가입경로'] = print_df['가입경로'].map({
    0 : "직접 유입", 1 : "키워드 검색", 2 : "블로그", 3 : "카페", 
    4 : "이메일", 5 : "카카오톡", 6 : "메타", 7 : "인스타그램", 
    8 : "유튜브", 9 : "배너 광고", 10 : "트위터 X", 11 : "기타 SNS"})
print_df['참여_전'] = print_df['참여_전'].map({0:'가입', 1:'미가입'})
print_df['참여이벤트'] = print_df['참여이벤트'].map({
    0 : "워크숍 개최", 1 : "재활용 품목 수집 이벤트", 2 : "재활용 아트 전시",
    3 : "게임 및 퀴즈", 4 : "커뮤니티 청소 활동", 5 : "업사이클링 마켓", 6 : "홍보 부스 운영"})
print_df['참여_후'] = print_df['참여_후'].map({0 : '가입', 1 : '미가입'})

#분석에 필요한 데이터 추출
data = memeber_df[['age', 'city', 'gender', 'marriage', 'after_ev']]

tab1, tab2, tab3, tab4, tab5 = st.tabs(['서비스가입 예측', '추천 캠페인', '추천 채널', '전환율 예측', '방문자수 예측'])

#탭1 : 서비스 가입 예측 모델
with tab1: 
    with st.expander('회원 데이터'):
        st.dataframe(print_df, use_container_width = True)
    col1, col2, col3 = st.columns([4, 3, 3])
    #첫 번째 열 : 사용자가 선택할 연령대를 위한 슬라이더
    with col1:
        st.write("서비스가입 예측 모델입니다. 아래의 조건을 선택해 주세요.")
        ages_1 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45)
        )
        st.write(f"**선택 연령대: :red[{ages_1}]세**")
    #두 번째 열 : 성별 선택을 위한 라디오 버튼
    with col2:
        gender_1 = st.radio(
            "성별을 선택해 주세요.",
            ["남자👨🏻", "여자👩🏻"],
            index = 0
        )
    #세 번째 열 : 혼인여부 선택을 위한 라디오 버튼
    with col3:
        marriage_1 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼🙅🏻‍♀️", "기혼👰🏻🤵🏻"],
            index = 0
        )
    
    #예측 모델 학습 및 평가 함수 > 성능 최적화 및 반복 실행 방지
    @st.cache_data
    def train_model(data):
        numeric_features = ['age']                      #나이 컬럼은 수치형
        categorical_features = ['gender', 'marriage']   #성별과 혼인여부는 범주형

        #ColumnTransformer 설정
        preprocessor = ColumnTransformer(
            transformers = [
                ('num', StandardScaler(), numeric_features),                        #수치형 : 표준화                                     #수치형 - 표준화 
                ('cat', OneHotEncoder(categories = 'auto'), categorical_features)   #범주형 : 원핫인코딩
            ]
        )

        #랜덤 포레스트 모델 정의
        model = Pipeline(steps = [
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state = 42, n_jobs = -1))
        ])

        #데이터 분할
        X = data.drop(columns = ['after_ev'])
        y = data['after_ev']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        #하이퍼파라미터 튜닝을 위한 그리드 서치
        param_grid = {
            'classifier__n_estimators' : [100, 200],    #트리의 개수
            'classifier__max_depth' : [None, 10, 20],   #트리의 깊이
            'classifier__min_samples_split' : [2, 5]    #분할에 필요한 최소 샘플 수
        }

        grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
        grid_search.fit(X_train, y_train)

        return grid_search, X_test, y_test

    #성능 평가 및 지표 출력 함수
    def evaluate_model(grid_search, X_test, y_test):
        y_pred = grid_search.predict(X_test)

        #정확도, 정밀도, 재현율, F1 스코어 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        #성능 지표 출력
        st.write(f"이 모델의 정확도: {accuracy * 100:.1f}%, 정밀도(Precision): {precision * 100:.1f}%, 재현율 (Recall): {recall * 100:.1f}%")
        st.write(f"F1-Score: {f1 * 100:.1f}%")

        return y_pred

    #시각화 함수(혼동 행렬 및 ROC 곡선)
    def plot_metrics(y_test, y_pred, grid_search):
        cm = confusion_matrix(y_test, y_pred)

        y_scores = grid_search.predict_proba(X_test)[:, 1]  #긍정 클래스 확률
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)  #ROC곡선 계산
        roc_auc = auc(fpr, tpr)                             #AUC 계산

        #첫 번째 열에 혼동 행렬 시각화
        col1, col2 = st.columns(2)

        with col1:
            #혼동 행렬 시각화
            cm_df = pd.DataFrame(cm, index = ['가입', '미가입'], columns = ['가입', '미가입'])
            fig = px.imshow(cm_df, 
                            text_auto = True, 
                            color_continuous_scale = 'GnBu', 
                            title = '혼동 행렬')
            fig.update_xaxes(title = '예측 레이블')
            fig.update_yaxes(title = '실제 레이블')
            fig.update_layout(width = 600, height = 600)
            st.plotly_chart(fig)

        with col2:
            #ROC 곡선 시각화
            fig_roc = go.Figure()

            #ROC 곡선 추가
            fig_roc.add_trace(go.Scatter(
                x = fpr,
                y = tpr, 
                mode = 'lines', 
                name = 'ROC curve (area = {:.2f})'.format(roc_auc),
                line = dict(width = 2, color = 'blue')))

            #랜덤 분류기 추가
            fig_roc.add_trace(go.Scatter(
                x = [0, 1], 
                y = [0, 1], 
                mode = 'lines', 
                name = 'Random Classifier',
                line = dict(width = 2, dash = 'dash', color = 'black')))

            #레이아웃 설정
            fig_roc.update_layout(
                title = 'Receiver Operating Characteristic (ROC)',
                xaxis_title = 'False Positive Rate',
                yaxis_title = 'True Positive Rate',
                showlegend = True,
                width = 600,
                height = 600
            )

            #Streamlit에서 ROC 곡선 그래프 표시
            st.plotly_chart(fig_roc)

    #예측 결과 출력 함수
    def pre_result(model, new_data):
        prediction = model.predict(new_data)
        st.write(f"**모델 예측 결과: :rainbow[{'가입' if prediction[0] == 0 else '미가입'}]**") # 0:가입, 1:미가입

    #버튼 클릭에 따른 동작
    if st.button("예측하기"):
        #입력된 값을 새로운 데이터 형식으로 변환
        new_data = pd.DataFrame({
            'age' : [(ages_1[0] + ages_1[1]) / 2],          #나이의 중앙값
            'gender' : [1 if gender_1 == '여자' else 0],    #성별 인코딩 (0 : 남자, 1 : 여자)
            'marriage' : [1 if marriage_1 == '기혼' else 0] #혼인 여부 인코딩 (0 : 미혼, 1 : 기혼)
        })

        #기존 데이터로 모델 학습
        grid_search, X_test, y_test = train_model(data)

        #예측 수행
        pre_result(grid_search.best_estimator_, new_data)

        #성능 평가 및 지표 출력
        y_pred = evaluate_model(grid_search, X_test, y_test)

        #시각화
        plot_metrics(y_test, y_pred, grid_search)

data_2 = memeber_df[['age', 'city', 'gender', 'marriage', 'before_ev', 'part_ev', 'after_ev']]

#참여 이벤트 매핑
event_mapping = {
    0 : "워크숍 개최", 1 : "재활용 품목 수집 이벤트", 2 : "재활용 아트 전시",
    3 : "게임 및 퀴즈", 4 : "커뮤니티 청소 활동", 5 : "업사이클링 마켓", 6 : "홍보 부스 운영"
}

city_mapping = {
    0 : '부산', 1 : '대구', 2 : '인천', 3 : '대전', 
    4 : '울산', 5 : '광주', 6 : '서울', 7 : '경기', 
    8 : '강원', 9 : '충북', 10 : '충남', 11 : '전북', 
    12 : '전남', 13 : '경북', 14 : '경남', 15 : '세종', 16 : '제주'
}

city_options = ["전체지역"] + list(city_mapping.values())

#탭2 : 캠페인 추천 모델
with tab2:
    #회원 데이터 출력
    with st.expander('회원 데이터'):
        st.dataframe(print_df, use_container_width = True)
    col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
    #연령대 선택 슬라이더
    with col1:
        st.write("캠페인 추천 모델입니다. 아래의 조건을 선택해 주세요.")
        ages_2 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45),
            key = 'slider_2'
        )
        st.write(f"**선택 연령대: :red[{ages_2}]세**")
    #도시 선택
    with col2:
        city_2 = st.selectbox(
            "도시를 선택해 주세요.",
            city_options,
            index = 0,
            key = 'selectbox2'
        )
        city_index = city_options.index(city_2)  #선택된 도시의 인덱스 저장
    #성별 선택
    with col3:
        gender_2 = st.radio(
            "성별을 선택해 주세요.",
            ["남자👨🏻", "여자👩🏻"],
            index = 0,
            key = 'radio2_1'
        )
    #혼인 여부 선택
    with col4:
        marriage_2 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼🙅🏻‍♀️", "기혼👰🏻🤵🏻"],
            index = 0,
            key = 'radio2_2'
        )

    #추천 모델 함수 > 성능 최적화 및 반복 실행 방지
    @st.cache_data
    def calculate_enrollment_increase_rate(data):
        #캠페인 별 가입 증가율 계산
        increase_rates = {}
        
        #조건별 캠페인 그룹화 및 계산
        campaign_groups = data.groupby('part_ev')
        
        for campaign, group in campaign_groups:
            #캠페인 전후 가입자 수 계산
            pre_signups = group['after_ev'].sum()       #이벤트 후 가입자 수
            post_signups = group['before_ev'].sum()     #이벤트 전 가입자 수
            
            #가입 증가율 계산 (0으로 나누는 경우 처리)
            if pre_signups > 0:
                increase_rate = (post_signups - pre_signups) / pre_signups
            else:       #가입자 수가 없다면 증가율 1
                increase_rate = 1 if post_signups > 0 else 0  
            
            increase_rates[campaign] = increase_rate

        return increase_rates
    #추천 캠페인 함수
    def recommend_campaign(data, age_range, city_index, gender, marriage):
    #조건에 따라 데이터 필터링
        if city_index == 0:     #'전체 지역' 선택
            filtered_data = data[
                (data['age'].between(age_range[0], age_range[1])) &
                (data['gender'] == (1 if gender == '여자' else 0)) &
                (data['marriage'] == (1 if marriage == '기혼' else 0))
            ]
        else:       #특정 도시 선택
            city_name = list(city_mapping.values())[city_index]  #선택된 도시의 이름을 가져옴
            filtered_data = data[
                (data['age'].between(age_range[0], age_range[1])) &
                (data['city'] == city_name) &
                (data['gender'] == (1 if gender == '여자' else 0)) &
                (data['marriage'] == (1 if marriage == '기혼' else 0))
            ]

        if filtered_data.empty:
            return "해당 조건에 맞는 데이터가 없습니다."
        
        #가입 증가율 계산
        increase_rates = calculate_enrollment_increase_rate(filtered_data)

        #가장 높은 가입 증가율을 가진 캠페인 추천
        best_campaign = max(increase_rates, key=increase_rates.get)
        
        return best_campaign, increase_rates

    #사용자 정보 입력을 통한 추천 이벤트 평가
    if st.button("캠페인 추천 받기"):
        best_campaign, increase_rates = recommend_campaign(data_2, ages_2, city_index, gender_2, marriage_2)
            
        if isinstance(best_campaign, str):
            st.write(best_campaign)
        else:
            st.write(f"**추천 캠페인: :violet[{event_mapping[best_campaign]}] 👈 이 캠페인이 가장 가입을 유도할 수 있습니다!**")
            
            #가입 증가율 결과 출력
            with st.expander("**각 캠페인별 가입 증가율 보기**"):
                for campaign, rate in increase_rates.items():
                    st.write(f"캠페인 {event_mapping[campaign]}의 가입 증가율: {rate:.2%}")
            
            #가입 증가율 결과 출력 및 가로 막대그래프 표시
            campaigns, rates = zip(*increase_rates.items())
            campaigns = [event_mapping[campaign] for campaign in campaigns]     #매핑된 캠페인 이름
            
            #파스텔 톤 색상 리스트 생성
            pastel_colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#77DD77', '#B19CD9', '#FFDAB9' ]
            
            #가로 막대그래프 시각화
            fig_bar = go.Figure()

            #가로 막대 추가
            fig_bar.add_trace(go.Bar(
                y = campaigns,                              #캠페인 이름
                x = rates,                                  #가입 증가율
                orientation = 'h',                          #가로 막대그래프
                marker = dict(color = pastel_colors),       #색상 설정
            ))

            #0 선 추가
            fig_bar.add_shape(
                type = 'line',
                x0 = 0,
                y0 = -0.5,
                x1 = 0,
                y1 = len(campaigns) - 0.5,
                line = dict(color = 'gray', width = 0.8),
            )

            #레이아웃 설정
            fig_bar.update_layout(
                title = '캠페인별 가입 증가율',
                xaxis_title = '가입 증가율',
                height = 600
            )

            #X축 설정
            fig_bar.update_xaxes(
                range = [min(min(rates), 0), max(max(rates), 0)],       #X축 범위 설정
                showgrid = True
            )

            #Y축 설정
            fig_bar.update_yaxes(
                title = '캠페인',
                showgrid = False
            )

            #Streamlit에서 가로 막대그래프 표시
            st.plotly_chart(fig_bar)

data_3 = memeber_df[['age', 'gender', 'marriage', 'channel', 'before_ev']]

#가입 시 유입경로 매핑
register_channel = {
    0:"직접 유입", 1:"키워드 검색", 2:"블로그", 3:"카페",
    4:"이메일", 5:"카카오톡", 6:"메타", 7:"인스타그램",
    8:"유튜브", 9:"배너 광고", 10:"트위터 X", 11:"기타 SNS"
}

#탭3 : 마케팅 채널 추천 모델
with tab3: 
    with st.expander('회원 데이터'):
        #회원 데이터 출력
        st.dataframe(print_df, use_container_width = True)
    col1, col2, col3 = st.columns([6, 2, 2])
    #연령대 선택 슬라이더
    with col1:
        st.write("마케팅 채널 추천 모델입니다. 아래의 조건을 선택해 주세요")
        ages_3 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45),
            key = 'slider_3'
        )
        st.write(f"**선택 연령대: :red[{ages_3}]세**")
    #성별 선택
    with col2:
        gender_3 = st.radio(
            "성별을 선택해 주세요.",
            ["남자👨🏻", "여자👩🏻"],
            index = 0,
            key = 'radio3_1'
        )
    #혼인여부 선택
    with col3:
        marriage_3 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼🙅🏻‍♀️", "기혼👰🏻🤵🏻"],
            index = 0,
            key = 'radio3_2'
        )

    #추천 모델 함수
    @st.cache_data
    def calculate_channel_conversion_rate(data):
        #마케팅 채널별 가입률 계산
        channel_stats = data.groupby('channel').agg(
        total_members = ('before_ev', 'count'),                     #전체 유입자 수
        total_signups = ('before_ev', lambda x: (x == 0).sum())     #가입자 수 (before_ev가 0인 경우)
        )
        
        #가입률 계산 : 가입자의 수 / 전체 유입자의 수
        channel_stats['conversion_rate'] = channel_stats['total_signups'] / channel_stats['total_members']
        channel_stats.reset_index(inplace = True)
        return channel_stats[['channel', 'conversion_rate']]

    def recommend_channel(data, age_range, gender, marriage):
        #조건에 맞는 가장 추천 마케팅 채널 3개를 반환
        filtered_data = data[
            (data['age'].between(age_range[0], age_range[1])) &
            (data['gender'] == (1 if gender == '여자' else 0)) &
            (data['marriage'] == (1 if marriage == '기혼' else 0))
        ]

        channel_rates = calculate_channel_conversion_rate(filtered_data)
        
        #"직접 유입" 채널 제외
        channel_rates = channel_rates[channel_rates['channel'] != 0]
        top_channels = channel_rates.nlargest(3, 'conversion_rate')
        
        return top_channels

    def display_channel_rates(channel_rates):
        #마케팅 채널 가입률 수치 표시
        with st.expander("**각 마케팅 채널별 가입률 보기**"):
            for _, row in channel_rates.iterrows():
                channel_name = register_channel[row['channel']]
                st.write(f"{channel_name}: {row['conversion_rate']:.2%}")

    def plot_channel_rates(channel_rates):
        #마케팅 채널 가입률 시각화(막대 그래프)
        fig_bar = go.Figure()

        #파스텔 톤 색상 리스트 생성
        pastel_colors = ['#FFDAB9', '#BDFCC9', '#E6E6FA']
        #막대 그래프 추가
        fig_bar.add_trace(go.Bar(
            y = channel_rates['channel'].apply(lambda x: register_channel[x]),
            x = channel_rates['conversion_rate'],   #가입률
            orientation = 'h',                      #가로 막대 그래프
            marker = dict(color = pastel_colors),   #색상 설정
        ))

        #선추가
        fig_bar.add_shape(
            type = 'line',
            x0 = 0,
            y0 = -0.5,
            x1 = 0,
            y1 = len(channel_rates) - 0.5,      #Y축 개수
            line=dict(color = 'gray', width = 0.8),
        )

        #레이아웃 설정
        fig_bar.update_layout(
            title = '마케팅 채널별 가입률',
            xaxis_title = '가입률',
            height = 600
        )

        #X축 설정
        fig_bar.update_xaxes(
            range = [min(min(channel_rates['conversion_rate']), 0), max(max(channel_rates['conversion_rate']), 0)],
            showgrid = True
        )

        #y축 설정
        fig_bar.update_yaxes(
            title = '마케팅 채널',
            showgrid = False)
        
        #표시
        st.plotly_chart(fig_bar)

    #사용자 정보 입력을 통한 추천 이벤트 평가
    if st.button("효과적인 마케팅 채널 추천받기"):
        #추천 모델 훈련
        top_channels = recommend_channel(data_3, ages_3, gender_3, marriage_3)

        #추천된 마케팅 채널이 있으면 출력
        if not top_channels.empty:
            st.write(f"**추천 마케팅 채널:** :violet[{', '.join(top_channels['channel'].apply(lambda x: register_channel[x]))}] 👈 이 채널들이 가장 효과적입니다!")
            display_channel_rates(top_channels)     #각 채널의 가입률 표시
            plot_channel_rates(top_channels)        #채널 가입률 시각화
        else:       #조건에 맞는 채널이 없을 경우 메시지 출력
            st.write("해당 조건에 맞는 마케팅 채널이 없습니다.")

#탭4 : 전환율 예측
with tab4:  
    #데이터 로드
    with st.expander('온라인 데이터'):
        st.dataframe(df_on, use_container_width = True)
    select_all_device = st.checkbox("디바이스 전체 선택")
    device_options = df_on["디바이스"].unique().tolist()    #디바이스 목록
    select_all_path = st.checkbox("유입경로 전체 선택")
    path_options = df_on["유입경로"].unique().tolist()      #유입경로 목록

    #디바이스 선택
    if select_all_device:
        select_device = st.multiselect("디바이스", device_options, default = device_options)        
    else:
        select_device = st.multiselect("디바이스", device_options)
    #유입경로 선택
    if select_all_path:
        select_path = st.multiselect("유입경로", path_options, default = path_options)
    else:
        select_path = st.multiselect("유입경로", path_options)
    #체류시간 입력(슬라이더)
    time_input = st.slider("체류 시간(분)", min_value = 0, max_value = 100, value = 0, step = 5)
        
    #온라인 데이터 복사 및 원-핫 인코딩
    df_ml_on = df_on.copy()
    df_ml_on = pd.get_dummies(df_ml_on, columns = ["디바이스", "유입경로"])        

    #특성과 목표 변수 설정
    features = ["체류시간(min)"] + [col for col in df_ml_on.columns if "디바이스_" in col or "유입경로_" in col]
    target = "전환율(가입)"

    #전환율 예측 버튼 클릭 시 동작
    if st.button("온라인 전환율 예측"):
        #입력(X), 출력(y) 데이터 정의
        X = df_ml_on[features]      #특성 변수
        y = df_ml_on[target]        #목표 변수

        #데이터 분할(학습 데이터 : 80%, 테스트 데이터 : 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        #결측값 처리
        y_train.fillna(y_train.median(), inplace = True)

        #랜덤 포레스트 회귀 모델 생성 및 학습
        on_model = RandomForestRegressor(n_estimators = 50, max_depth = 10, random_state = 42, n_jobs = -1)
        on_model.fit(X_train, y_train)

        #테스트 데이터 예측
        y_pred = on_model.predict(X_test)

        #예측 결과 시각화(실제 전환율 VS 예측 전환율 비교)
        fig_ml_on = go.Figure()

        #실제 값과 예측 값 비교를 위한 산점도 추가
        fig_ml_on.add_trace(go.Scatter(
            x = y_test,                 #실제 값
            y = y_pred,                 #예측 값
            mode = 'markers+lines',     #마커와 선을 동시에 표시
            marker = dict(symbol = 'circle', size = 8, color = 'blue', line = dict(width = 2)),
            line = dict(shape = 'linear'),
            name = '예측 vs 실제'       #레전드에 표시될 이름
        ))

        #레이아웃 설정
        fig_ml_on.update_layout(
            title = '✅전환율 예측 결과 비교',
            xaxis_title = '실제 전환율',
            yaxis_title = '예측 전환율',
            height = 600,
            xaxis = dict(showgrid = True),      #X축 그리드 표시
            yaxis = dict(showgrid = True),      #Y축 그리드 표시
        )

        #Streamlit에서 시각화 표시
        st.plotly_chart(fig_ml_on)
    
        #사용자가 입력한 값을 기반으로 전환율 예측
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
        st.subheader(f"예상 전환율 : {predicted_conversion:.2f}%")

#탭5 : 방문자 수 예측
with tab5:  
    #오프라인 데이터 출력
    with st.expander('오프라인 데이터'):
        st.dataframe(df_off, use_container_width=True)
    #지역 선택 옵션(전체 지역 + 매핑된 지역 목록)
    city_options = ["전체지역"] + list(city_mapping.values())

    #학습 데이터 준비(날짜, 지역별 방문자 수 합산)
    df_ml_off = df_off.groupby(["날짜", "지역"])["방문자수"].sum().reset_index()
    df_ml_off["날짜"] = pd.to_datetime(df_ml_off["날짜"])       #날짜 컬럼을 datetime 형식으로 변환
    df_ml_off["year"] = df_ml_off["날짜"].dt.year               #연도 컬럼 생성
    df_ml_off["month"] = df_ml_off["날짜"].dt.month             #월 컬럼 생성
    df_ml_off["day"] = df_ml_off["날짜"].dt.day                 #일자 컬럼 생성
    df_ml_off["day_of_week"] = df_ml_off["날짜"].dt.weekday     #요일 컬럼 생성

    #지역 선택 박스
    select_region = st.selectbox("지역을 선택하세요.", city_options)

    #선택된 지역에 따른 데이터 필터링
    if select_region == "전체지역":     #전체 지역 데이터를 사용
        df_region = df_ml_off       
    else:                               #특정 지역 데이터 사용
        df_region = df_ml_off[df_ml_off["지역"] == select_region]  

    #학습에 사용할 특성 설정
    features = ["year", "month", "day", "day_of_week"]
    X = df_region[features]     #특성 변수
    y = df_region["방문자수"]   #목표 변수

    #방문자 수 예측 버튼 클릭 시 동작
    if st.button("오프라인 방문자 수 예측"):        #향후 12개월간의 방문자 수 예측
        #데이터 분할(학습 데이터 : 80%, 테스트 데이터 : 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        #랜덤 포레스트 회귀 모델 생성 및 학습
        off_model = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs = -1)
        off_model.fit(X_train, y_train)

        #최대 날짜의 다음 달부터 12개월 간의 날짜 생성
        max_date = df_region["날짜"].max()
        start_date = (max_date + pd.DateOffset(months = 1)).replace(day = 1)        #다음 달의 첫날
        future_dates = pd.date_range(start = start_date, periods = 365, freq = "D")
        future_df = pd.DataFrame({
            "year" : future_dates.year,
            "month" : future_dates.month,
            "day" : future_dates.day,
            "day_of_week" : future_dates.weekday
        })
        
        #방문자 수 예측
        future_pred = off_model.predict(future_df)      #예측값 계산
        future_df["예측 방문자 수"] = future_pred       #예측 방문자 수 컬럼 추가

        #"년-월" 형식의 칼럼 만들기
        future_df["년월"] = future_df["year"].astype(str) + "-" + future_df["month"].astype(str).str.zfill(2)  # 월을 두 자리로 표시

        #월 별로 집계한 방문자 수
        future_summary = future_df.groupby("년월", as_index=False)["예측 방문자 수"].sum()

        #예측 방문자 수 형식 변경
        future_summary["예측 방문자 수"] = future_summary["예측 방문자 수"].astype(int).astype(str) + "명"

        st.subheader(f":chart: 향후 12개월 동안 {select_region}의 방문자 수 예측")

        #방문자 수 예측 시각화
        fig_ml_off = go.Figure()

        #예측 방문자 수 선 그래프 추가
        fig_ml_off.add_trace(go.Scatter(
            x = future_summary["년월"],
            y = future_summary["예측 방문자 수"].str.extract('(\d+)')[0].astype(int),  #숫자만 추출하여 y값으로 사용
            mode = 'markers+lines',         #마커와 선을 동시에 표시
            marker = dict(symbol = 'circle', size = 8, color = 'red'),
            line = dict(shape = 'linear'),
            name = '예측 방문자 수'         #범례에 표시될 이름
        ))

        #레이아웃 설정
        fig_ml_off.update_layout(
            title = f"{select_region}의 방문자 수 예측",
            xaxis_title = '년-월',
            yaxis_title = '방문자 수',
            height = 600,
            xaxis = dict(showgrid = False),     #x축 그리드 표시 안 함
            yaxis = dict(showgrid = True),      #y축 그리드 표시
        )   

        #Streamlit에서 시각화와 데이터프레임 표시
        col1, col2 = st.columns(2)

        #예측 결과 그래프 표시
        with col1:
            st.plotly_chart(fig_ml_off)
        #예측 방문자 수 데이터프레임 표시
        with col2:
            st.dataframe(future_summary[["년월", "예측 방문자 수"]], height = 550)        