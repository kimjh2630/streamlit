from matplotlib import rc
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import folium
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#Streamlit 페이지 기본 설정 (메인 페이지 너비 넓게)
st.set_page_config(layout = "wide") 

with st.spinner("잠시만 기다려 주세요..."):
    time.sleep(1)  #대기 시간 시뮬레이션
st.success("Data Loaded!")

#한글 및 마이너스 깨짐
plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

#GitHub에서 데이터를 불러오기 위한 기본 경로 설정
CSV_FILE_PATH = 'https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/'

#회원 데이터 불러오기
memeber_df = pd.read_csv(CSV_FILE_PATH + 'members_data.csv')

#온라인 재활용 데이터 로드 함수(Streamlit 캐싱 적용)
@st.cache_data
def on_load_data():
    url_on = "https://raw.githubusercontent.com/jjjjunn/YH_project/main/recycling_online.csv"
    df_on = pd.read_csv(url_on, encoding = "UTF8").fillna(0)    #결측치(NaN)을 0으로 채움
    df_on.replace([np.inf, -np.inf], np.nan, inplace = True)    #무한대 값을 NaN으로 변환
    df_on.fillna(0, inplace = True)                             #다시 NaN 값을 0으로 변환
    return df_on                                                #전처리된 데이터 반환

#오프라인 재활용 데이터 로드 함수(Streamlit 캐싱 적용)
@st.cache_data
def off_load_data():
    url_off = "https://raw.githubusercontent.com/jjjjunn/YH_project/main/recycling_off.csv"
    df_off = pd.read_csv(url_off, encoding="UTF8")
    df_off.replace([np.inf, -np.inf], np.nan, inplace=True)     #무한대 값을 NaN으로 변환
    df_off.dropna(subset=["날짜"], inplace=True)                #다시 NaN 값을 0으로 변환
    return df_off                                               #전처리된 데이터 반환

# Streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

#회원 데이터 컬럼명을 한글로 변경(데이터프레임 가독성 걔선)
print_df = memeber_df.rename(columns = {
     "age": "나이",
     "gender": "성별",
     "marriage": "혼인여부",
     "city": "도시",
     "channel": "가입경로",
     "before_ev": "참여_전",
     "part_ev": "참여이벤트",
     "after_ev": "참여_후"
})

#범주형 데이터 값 변경(숫자 > 의미 있는 문자열)
print_df['성별'] = print_df['성별'].map({0:'남자', 1:'여자'})
print_df['혼인여부'] = print_df['혼인여부'].map({0:'미혼', 1:'기혼'})
#도시 코드 > 도시 이름 매핑
print_df['도시'] = print_df['도시'].map({0:'부산', 1:'대구', 2:'인천', 3:'대전', 4:'울산', 5:'광주', 6:'서울', 
    7:'경기', 8:'강원', 9:'충북', 10:'충남', 11:'전북', 12:'전남', 13:'경북', 14:'경남', 15:'세종', 16:'제주'})
#가입 경로 코드 > 의미 있는 문자열 매핑
print_df['가입경로'] = print_df['가입경로'].map({0:"직접 유입", 1:"키워드 검색", 2:"블로그", 3:"카페", 4:"이메일", 
        5:"카카오톡", 6:"메타", 7:"인스타그램", 8:"유튜브", 9:"배너 광고", 10:"트위터 X", 11:"기타 SNS"})
#참여 전 상태 코드 > 문자열 변환
print_df['참여_전'] = print_df['참여_전'].map({0:'가입', 1:'미가입'})
#참여 이벤트 코드 > 의미 있는 이벤트명 매핑
print_df['참여이벤트'] = print_df['참여이벤트'].map({0:"워크숍 개최", 1:"재활용 품목 수집 이벤트", 2:"재활용 아트 전시",
          3:"게임 및 퀴즈", 4:"커뮤니티 청소 활동", 5:"업사이클링 마켓", 6:"홍보 부스 운영"})

#참여 후 상태 코드 > 문자열 변환
print_df['참여_후'] = print_df['참여_후'].map({0:'가입', 1:'미가입'})

#온라인 & 오프라인 데이터 로드
df_on = on_load_data()
df_off = off_load_data()

#회원 데이터 중 일부 컬럼 선택
data = memeber_df[['age', 'gender', 'marriage', 'after_ev']]

#Streamlit UI 구성
with st.expander('회원 데이터'):
    st.dataframe(print_df, use_container_width = True)      #변환된 회원 데이터 표시
with st.expander('온라인 데이터'):
    st.dataframe(df_on, use_container_width = True)         #온라인 재활용 데이터 표시
with st.expander('오프라인 데이터'):
    st.dataframe(df_off, use_container_width = True)        #오프라인 재활용 데이터 표시

#Streamlit 탭을 이용한 다중 페이지 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs(['서비스가입 예측', '추천 캠페인', '추천 채널', '전환율 예측', '방문자 수 예측'])

#탭1 : 서비스 가입 예측 모델
with tab1: 
    first_column, second_column, thrid_columns = st.columns([6, 2, 2])
    
    #첫 번째 열 : 연령대 선택
    with first_column:
        st.write("서비스가입 예측 모델입니다. 아래의 조건을 선택해 주세요.")
        #슬라이더로 연령대 선택 (기본값 : 35 ~ 45세)
        ages_1 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45)
        )
        st.write(f"**선택 연령대: :red[{ages_1}]세**")
    
    #두 번째 열 : 성별 선택
    with second_column:
        gender_1 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=1     #기본값 : 여자
        )
    
    #세 번째 열 : 혼인여부 선택
    with thrid_columns:
        marriage_1 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=1     #기본값 : 기혼
        )
    
    #예측 모델 학습 및 평가 함수
    def service_predict(data):
        #데이터 전처리 및 파이프라인 설정
        numeric_features = ['age']                      #수치형 피처
        categorical_features = ['gender', 'marriage']   #범주형 피처

        preprocessor = ColumnTransformer(
            #수치형, 범주형 데이터 변환기 설정
            transformers = [    
                ('num', StandardScaler(), numeric_features),    #수치형 : StandardScaler
                ('cat', OneHotEncoder(categories='auto'),           categorical_features)                           #범주형 : OneHotEncoder
            ]
        )

        #랜덤 포레스트 모델
        #파이프라인 구성
        model = Pipeline(steps = [
            #전처리기
            ('preprocessor', preprocessor),
            #분류기 : 랜덤 포레스트
            ('classifier', RandomForestClassifier(random_state = 42, n_jobs = -1))      #n_jobs = -1로 모든 코어 사용
        ])

        #데이터 분할(X : 특성, y : 타겟)
        X = data.drop(columns = ['after_ev'])     #'after_ev'를 제외한 특성만 사용
        y = data['after_ev']                      #타겟 변수('after_ev'가 가입/미가입 여부)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)       #학습/테스트 데이터 분할

        #하이퍼파라미터 튜닝을 위한 그리드 서치
        param_grid = {
            'classifier__n_estimators': [100, 200],     #n_estimators : 트리 개수
            'classifier__max_depth': [None, 10, 20],    #max_depth : 트리 깊이 
            'classifier__min_samples_split': [2, 5]     #min_samples_split : 노드 분할 기준
        }

        #그리드 서치 실행
        grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
        grid_search.fit(X_train, y_train)       #학습 수행

        #최적의 모델로 예측 수행
        y_pred = grid_search.predict(X_test)

        #성능 평가(정확도 출력)
        accuracy = accuracy_score(y_test, y_pred)   #정확도 계산
        st.write(f"이 모델의 테스트 정확도는 {accuracy * 100:.1f}% 입니다.")

        #최적 모델과 특성 중요도 반환
        return grid_search.best_estimator_, grid_search.best_estimator_.named_steps['classifier'].feature_importances_

    #사용자가 입력한 값을 새로운 데이터로 변환하여 예측 수행
    def pre_result(model, new_data):
        prediction = model.predict(new_data)    #예측 수행
        st.write(f"**모델 예측 결과: :rainbow[{'가입' if prediction[0] == 0 else '미가입'}]**")

    #특성 중요도 시각화
    def plot_feature_importance(importances, feature_names):
        indices = np.argsort(importances)[::-1]     #중요도 내림차순 정렬
        plt.figure(figsize = (2, 1))                #그래프 크기 설정
        plt.title("특성 중요도")                    #제목
        plt.barh(range(len(importances)), importances[indices], align = "center")       #수평 바 그래프
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])        # 특성 이름 출력
        plt.xlabel("중요도")        #x축 레이블
        st.pyplot(plt)              #Streamlit에서 시각화 출력

    #예측하기 버튼 클릭에 따른 동작
    if st.button("예측하기"):
        #기존 데이터로 모델 학습
        model, feature_importances = service_predict(data)

        #입력된 값을 새로운 데이터 형식으로 변환
        new_data = pd.DataFrame({
            'age': [(ages_1[0] + ages_1[1]) / 2],           #나이의 중앙값
            'gender': [1 if gender_1 == '여자' else 0],     #성별을 1(여자), 0(남자)로 변환
            'marriage': [1 if marriage_1 == '기혼' else 0]  #혼인 여부를 1(기혼), 0(미혼)으로 변환
        })

        #예측 수행 및 결과 출력
        pre_result(model, new_data)

        #특성 중요도 시각화
        feature_names = ['age'] + list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())       #특성 이름 리스트
        plot_feature_importance(feature_importances, feature_names)                                                         #중요도 시각화

#두 번째 데이터셋(참여 이벤트 및 후속 가입 여부 포함)
data_2 = memeber_df[['age', 'gender', 'marriage', 'part_ev', 'after_ev']]   #추가된 'part_ev' (참여 이벤트)

#참여 이벤트를 위한 매핑정의
event_mapping = {
    0: "워크숍 개최",
    1: "재활용 품목 수집 이벤트",
    2: "재활용 아트 전시",
    3: "게임 및 퀴즈",
    4: "커뮤니티 청소 활동",
    5: "업사이클링 마켓",
    6: "홍보 부스 운영"
}

with tab2: # 캠페인 추천 모델
    first_column, second_column, thrid_columns = st.columns([6, 2, 2])
    with first_column:
        st.write("캠페인 추천 모델입니다. 아래의 조건을 선택해 주세요.")
        ages_2 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45),
            key='slider_2'
        )
        st.write(f"**선택 연령대: :red[{ages_2}]세**")
    
    with second_column:
        gender_2 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=1,
            key='radio2_1'
        )
    
    with thrid_columns:
        marriage_2 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=1,
            key='radio2_2'
        )

    # 추천 모델 함수
    def recommend_event(data_2):
        # X, y 설정
        X = data_2[['age', 'gender', 'marriage', 'part_ev']]
        y = data_2['after_ev']

        # 더미 변수 생성하여 참여 이벤트 인코딩
        X = pd.get_dummies(X, columns=['part_ev'], drop_first=True)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

        # 렌덤 포레스트 모델 정의 및 학습
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        return model, X_train.columns  # 모델과 피쳐 이름을 반환

    # 사용자 정보 입력을 통한 추천 이벤트 평가
    if st.button("효과적인 이벤트 추천받기"):
        # 추천 모델 훈련
        model, feature_names = recommend_event(data_2)

        event_results = {}

        # 각 이벤트에 대한 추천 가능성 평가
        for event in range(7):  # part_ev가 0에서 6까지의 숫자이므로, 7개의 이벤트에 대해 반복
            # 새로운 사용자 정보 세팅
            new_user_data = pd.DataFrame({
                'age': [(ages_2[0] + ages_2[1]) / 2],  # 연령대의 중앙값
                'gender': [1 if gender_2 == '여자' else 0],  # 성별 인코딩
                'marriage': [1 if marriage_2 == '기혼' else 0],  # 혼인 여부 인코딩
                'part_ev': [event]  # 번호로 매핑된 이벤트
            })

            # 더미 변수 생성
            new_user_data = pd.get_dummies(new_user_data, columns=['part_ev'], drop_first=True)

            # 피쳐 정렬
            new_user_data = new_user_data.reindex(columns=feature_names, fill_value=0)

            # 예측 수행
            prediction = model.predict(new_user_data)
            event_results[event] = prediction[0]  # 가입 여부 저장 (0: 가입, 1: 미가입)

        # 가입(0) 가능성이 높은 이벤트 중 가장 높은 것
        possible_events = {event: result for event, result in event_results.items() if result == 0} 

        if possible_events:
            best_event = max(possible_events, key=possible_events.get)
            st.write(f"**추천 이벤트: :violet[{event_mapping[best_event]}] 👈 이벤트가 가장 효과적입니다!**")
        else:
            st.write("추천 이벤트: 가입 확률이 높지 않으므로 다른 캠페인을 고려해보세요.")

data_3 = memeber_df[['age', 'gender', 'marriage', 'channel', 'after_ev']]

# 가입 시 유입경로 매핑
register_channel = {
    0:"직접 유입",
    1:"키워드 검색",
    2:"블로그",
    3:"카페",
    4:"이메일",
    5:"카카오톡",
    6:"메타",
    7:"인스타그램",
    8:"유튜브", 
    9:"배너 광고", 
    10:"트위터 X", 
    11:"기타 SNS"
}

with tab3: # 마케팅 채널 추천 모델
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.write("마케팅 채널 추천 모델입니다. 아래의 조건을 선택해 주세요")
        ages_3 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45),
            key='slider_3'
        )
        st.write(f"**선택 연령대: :red[{ages_3}]세**")
    
    with col2:
        gender_3 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=1,
            key='radio3_1'
        )
    
    with col3:
        marriage_3 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=1,
            key='radio3_2'
        )

        # 추천 모델 함수
    def recommend_channel(data_3):
        # X, y 설정
        X = data_3[['age', 'gender', 'marriage', 'channel']]
        y = data_3['after_ev']

        # 더미 변수 생성하여 유입 채널 인코딩
        X = pd.get_dummies(X, columns=['channel'], drop_first=True)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

        # 렌덤 포레스트 모델 정의 및 학습
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        return model, X_train.columns  # 모델과 피쳐 이름을 반환

    # 사용자 정보 입력을 통한 추천 이벤트 평가
    if st.button("효과적인 마케팅 채널 추천받기"):
        # 추천 모델 훈련
        model, feature_names = recommend_channel(data_3)

        channel_results = {}

        # 각 이벤트에 대한 추천 가능성 평가
        for channel in range(12):  # part_ev가 0에서 12까지의 숫자이므로, 12개의 이벤트에 대해 반복
            if channel in (0,1): # 직접 유입과 키워드 검색 채널 제외
                continue

            # 새로운 사용자 정보 세팅
            new_user_data = pd.DataFrame({
                'age': [(ages_2[0] + ages_2[1]) / 2],  # 연령대의 중앙값
                'gender': [1 if gender_2 == '여자' else 0],  # 성별 인코딩
                'marriage': [1 if marriage_2 == '기혼' else 0],  # 혼인 여부 인코딩
                'channel': [channel]  # 번호로 매핑된 채널
            })

            # 더미 변수 생성
            new_user_data = pd.get_dummies(new_user_data, columns=['channel'], drop_first=True)

            # 피쳐 정렬
            new_user_data = new_user_data.reindex(columns=feature_names, fill_value=0)

            # 예측 수행
            prediction = model.predict(new_user_data)
            channel_results[channel] = prediction[0]  # 가입 여부 저장 (0: 가입, 1: 미가입)

        # 가입(0) 가능성이 높은 채널 중 가장 높은 것 3개
        possible_channels = {channel: result for channel, result in channel_results.items() if result == 0} 

        if possible_channels:
            best_channels = sorted(possible_channels.keys(), key=lambda x: possible_channels[x])[:3]  # 가장 좋은 3개 채널
            recommended_channels = [register_channel[ch] for ch in best_channels]
            st.write(f"**추천 마케팅 채널:** :violet[{', '.join(recommended_channels)}] 👈 이 채널들이 가장 효과적입니다!")
        else:
            st.write("추천 마케팅 채널: 가입 확률이 높지 않으므로 다른 채널을 고려해보세요.")

with tab4:  #전환율 예측
    #사용자 입력을 위한 체크박스 설정
    select_all_device = st.checkbox("디바이스 전체 선택")   #전체 디바이스 선택 여부
    device_options = df_on["디바이스"].unique().tolist()    #디바이스 목록
    select_all_path = st.checkbox("유입경로 전체 선택")     #전체 유입경로 선택 여부
    path_options = df_on["유입경로"].unique().tolist()      #유입경로 목록

    #사용자가 전체 디바이스 또는 유입경로를 선택하면 모든 옵션을 선택하도록 설정
    if select_all_device:
        select_device = st.multiselect("디바이스", device_options, default = device_options)        
    else:
        select_device = st.multiselect("디바이스", device_options)

    if select_all_path:
        select_path = st.multiselect("유입경로", path_options, default = path_options)
    else:
        select_path = st.multiselect("유입경로", path_options)

    #사용자가 체류 시간을 슬라이더로 선택
    time_input = st.slider("체류 시간(분)", min_value = 0, max_value = 100, value = 0, step = 5)
        
    #온라인 데이터 복사 및 원-핫 인코딩
    df_ml_on = df_on.copy()
    df_ml_on = pd.get_dummies(df_ml_on, columns = ["디바이스", "유입경로"])     #디바이스와 유입경로에 대한 원-핫 인코딩

        

    #체류시간 및 원-핫 인코딩된 디바이스, 유입경로 및 타겟 변수 설정
    features = ["체류시간(min)"] + [col for col in df_ml_on.columns if "디바이스_" in col or "유입경로_" in col]
    target = "전환율(가입)"

    if st.button("온라인 전환율 예측"):
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

        #예측 결과 시각화(실제 전환율 VS 예측 전환율 비교)
        fig_ml_on, ax_ml_on = plt.subplots(figsize = (9, 6))
        sns.lineplot(
            x = y_test,         #실제 값
            y = y_pred,         #예측 값
            marker = "o",
            ax = ax_ml_on,
            linestyle = "-",
            label="예측 vs 실제"        
        )
        ax_ml_on.grid(visible = True, linestyle = "-", linewidth = 0.5)
        ax_ml_on.set_title("전환율 예측 결과 비교")
        ax_ml_on.set_xlabel("실제 전환율")
        ax_ml_on.set_ylabel("예측 전환율")
        ax_ml_on.legend()
        st.pyplot(fig_ml_on)
    
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

with tab5:  #방문자 수 예측
    #오프라인 데이터 준비 : 날짜와 지역별 방문자 수 합산
    df_ml_off = df_off.groupby(["날짜", "지역"])["방문자수"].sum().reset_index()
    df_ml_off["날짜"] = pd.to_datetime(df_ml_off["날짜"])       #날짜 형식으로 전환
    df_ml_off["year"] = df_ml_off["날짜"].dt.year               #연도 정보 추출
    df_ml_off["month"] = df_ml_off["날짜"].dt.month             #월 정보 추출
    df_ml_off["day"] = df_ml_off["날짜"].dt.day                 #일 정보 추출
    df_ml_off["day_of_week"] = df_ml_off["날짜"].dt.weekday     #요일 정보 추출

    #지역 선택 옵션 추가
    select_region = st.selectbox("지역을 선택하세요.", df_ml_off["지역"].unique())

    #선택된 지역에 해당하는 데이터 필터링
    df_region = df_ml_off[df_ml_off["지역"] == select_region]
    features = ["year", "month", "day", "day_of_week"]
    X = df_region[features]
    y = df_region["방문자수"]

    if st.button("오프라인 방문자 수 예측"):    #향후 12개월간의 방문자 수 예측
        #학습 데이터와 테스트 데이터 분할(학습 데이터 : 80%, 테스트 데이터 : 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        #랜덤 포레스트 회귀 모델 생성 및 학습
        off_model = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs = -1)
        off_model.fit(X_train, y_train)

        #향후 12개월의 날짜 생성
        future_dates = pd.date_range(start = df_region["날짜"].max(), periods = 12, freq = "ME")
        future_df = pd.DataFrame({"year" : future_dates.year, "month": future_dates.month, "day": future_dates.day, "day_of_week": future_dates.weekday})

        #예측 수행
        future_pred = off_model.predict(future_df)
        future_df["예측 방문자 수"] = future_pred
        future_df["날짜"] = future_dates

        #예측 결과 시각화(향후 방문자 수 예측)
        st.subheader(f":chart: {select_region}의 방문자 수 예측")
        fig_ml_off, ax_ml_off = plt.subplots(figsize = (9, 6))
        ax_ml_off.plot(future_df.index, future_df["예측 방문자 수"], marker = "o", linestyle = "-", color = "red", label = "예측 방문자 수")
        ax_ml_off.set_title(f"{select_region}의 방문자 수 예측")
        ax_ml_off.set_xlabel("날짜")
        ax_ml_off.set_ylabel("방문자 수")
        ax_ml_off.legend()
        st.pyplot(fig_ml_off)

        #날짜 포맷 수정 후 예측 방문자 수 출력
        future_df["날짜"] = pd.to_datetime(future_df["날짜"]).apply(lambda x: x.replace(day = 1))
        future_df["날짜"] = future_df["날짜"] + pd.DateOffset(months = 1)
        future_df["예측 방문자 수"] = future_df["예측 방문자 수"].astype(int).astype(str) + "명"

        #향후 12개월의 예측 결과 출력
        st.subheader(":chart: 향후 12개월의 방문자 수 예측")
        st.write(future_df[["날짜", "예측 방문자 수"]])