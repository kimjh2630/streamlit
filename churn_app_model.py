#pip install xgboost joblib

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib  # 모델 저장 및 로드용

# 한글 폰트 설정 (Windows 기준, MacOS는 'AppleGothic' 사용 가능)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

st.title('📞 통신 고객 이탈 예측 앱')

st.info('이 앱은 5개의 주요 피처를 사용해 고객 이탈 여부를 예측합니다!')

# 데이터 로드
with st.expander('데이터'):
  st.write('**원본 데이터**')
  df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
  df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
  selected_features = {'tenure': '고객 유지 기간', 'MonthlyCharges': '월 요금', 'Contract': '계약 유형', 
                       'InternetService': '인터넷 서비스', 'PaymentMethod': '결제 방법', 'Churn': '이탈 여부'}
  df = df[list(selected_features.keys())].rename(columns=selected_features)
  df

  st.write('**X (입력 데이터)**')
  X_raw = df.drop('이탈 여부', axis=1)
  X_raw

  st.write('**y (타겟 데이터)**')
  y_raw = df['이탈 여부']
  y_raw

# 데이터 시각화
with st.expander('데이터 시각화'):
  st.write('**산점도: 고객 유지 기간 vs 월 요금**')
  st.scatter_chart(data=df, x='고객 유지 기간', y='월 요금', color='이탈 여부')

  st.write('**이탈 여부별 고객 유지 기간 분포**')
  fig, ax = plt.subplots()
  sns.histplot(data=df, x='고객 유지 기간', hue='이탈 여부', multiple='stack', ax=ax)
  ax.set_xlabel('고객 유지 기간 (개월)')
  ax.set_ylabel('고객 수')
  st.pyplot(fig)

  st.write('**모든 피처 간 상관관계 히트맵**')
  encode = ['계약 유형', '인터넷 서비스', '결제 방법']
  df_encoded = pd.get_dummies(df, columns=encode)
  df_encoded['이탈 여부'] = df_encoded['이탈 여부'].map({'No': 0, 'Yes': 1})
  corr = df_encoded.corr()
  fig, ax = plt.subplots(figsize=(12, 10))
  sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
  ax.set_title('모든 피처 간 상관관계 히트맵')
  plt.tight_layout()
  st.pyplot(fig)

# 입력 특성 (사이드바)
with st.sidebar:
  st.header('입력 피처')
  고객_유지_기간 = st.slider('고객 유지 기간 (개월)', 0, 72, 24)
  월_요금 = st.slider('월 요금 ($)', 18.0, 120.0, 70.0)
  계약_유형 = st.selectbox('계약 유형', ('Month-to-month', 'One year', 'Two year'), 
                          format_func=lambda x: {'Month-to-month': '월 단위', 'One year': '1년', 'Two year': '2년'}[x])
  인터넷_서비스 = st.selectbox('인터넷 서비스', ('DSL', 'Fiber optic', 'No'), 
                              format_func=lambda x: {'DSL': 'DSL', 'Fiber optic': '광통신', 'No': '없음'}[x])
  결제_방법 = st.selectbox('결제 방법', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'), 
                          format_func=lambda x: {'Electronic check': 'E-check', 'Mailed check': 'Mailed-check', 
                                                'Bank transfer (automatic)': '은행 이체 (자동)', 
                                                'Credit card (automatic)': '신용카드 (자동)'}[x])

  data = {
      '고객 유지 기간': 고객_유지_기간,
      '월 요금': 월_요금,
      '계약 유형': 계약_유형,
      '인터넷 서비스': 인터넷_서비스,
      '결제 방법': 결제_방법
  }
  input_df = pd.DataFrame(data, index=[0])
  input_customers = pd.concat([input_df, X_raw], axis=0)

with st.expander('입력 피처 확인'):
  st.write('**입력 고객 데이터**')
  input_df
  st.write('**결합된 고객 데이터**')
  input_customers

# 데이터 준비
encode = ['계약 유형', '인터넷 서비스', '결제 방법']
df_customers = pd.get_dummies(input_customers, columns=encode)

X = df_customers[1:]  # 학습 데이터
input_row = df_customers[:1]  # 예측할 입력 데이터

# y 인코딩
target_mapper = {'No': 0, 'Yes': 1}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('데이터 전처리'):
  st.write('**인코딩된 X (입력 고객)**')
  input_row
  st.write('**인코딩된 y**')
  y

# 모델 학습 및 저장
model_path = 'xgb_churn_model.pkl'
try:
  # 기존 모델 로드 시도
  clf = joblib.load(model_path)
  st.write("저장된 모델을 로드했습니다.")
except FileNotFoundError:
  # 모델이 없으면 새로 학습 후 저장
  clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
  clf.fit(X, y)
  joblib.dump(clf, model_path)
  st.write("새로운 모델을 학습하고 저장했습니다.")

# 예측 수행
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# 예측 확률을 소숫점 2자리로 포맷팅
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['아니오', '예'])
df_prediction_proba = df_prediction_proba.round(2)  # 소숫점 2자리로 반올림

# 예측 결과 표시
st.subheader('예측된 이탈 확률')
st.dataframe(df_prediction_proba,
             column_config={
               '아니오': st.column_config.ProgressColumn(
                 '아니오 (유지)',
                 format='%.2f',  # 소숫점 2자리 표시
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               '예': st.column_config.ProgressColumn(
                 '예 (이탈)',
                 format='%.2f',  # 소숫점 2자리 표시
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)

churn_status = np.array(['아니오 (유지)', '예 (이탈)'])
st.success(str(churn_status[prediction][0]))