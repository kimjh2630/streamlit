from datetime import datetime as dt

import datetime
import pandas as pd
import streamlit as st

#1. 버튼 위젯
#버튼 클릭
my_button = st.button("버튼을 클릭하세요.")
print(my_button)
#st.write(my_button)

if my_button:
    st.write(":blue[버튼]이 눌렸습니다. :sparkles:")

#링크 버튼
st.link_button("네이버", "https://www.naver.com/")

#파일 업로드 버튼
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding = "utf-8")
    st.write(df)
    st.write(f"Uploaded file name : {uploaded_file.name}")
    st.write(f"File size : {uploaded_file.size} bytes")

uploaded_file = st.file_uploader("Choose a TEXT file")
if uploaded_file:
    st.write(f"Uploaded file name : {uploaded_file.name}")
    file_content = uploaded_file.read().decode("utf8")
    st.write(file_content)

#파일 다운로드 버튼
#샘플 데이터 생성
dataframe = pd.DataFrame({
    'first column' : [1, 2, 3, 4],
    'second column' : [10, 20, 30, 40],
})

#다운로드 버튼 연결
st.download_button(
    label = "CSV로 다운로드",
    data = dataframe.to_csv(),
    file_name = "sample.csv",
    mime = "text.csv"
)

#text 파일 다운로드
data = """이것은 다운로드할 데이터입니다.
여러줄의 예시 텍스트를 작성하여 다운로드 기능을 테스트합니다.
이 데이터는 streamlit에서 다운로드 버튼을 누르면 저장됩니다.
예제 데이터를 다운로드하여 기능을 확인해보세요.
"""

download_button = st.download_button(
    "클릭하면 다운로드가 됩니다.",
    data = data,
    file_name = "결과.txt",
    mime = "text/plain"
)

#3. 선택형 위젯
#3-1 체크 박스
agree = st.checkbox("동의 하시겠습니까?")

if agree:
    st.write("동의 해주셔서 감사합니다. :100:")

#3-2 라디오 버튼
mbti = st.radio(
    "당신의 MBTI는 무엇입니까?",
    ('ISTJ', 'ENTP', '선택지 없음'),
    index = 2
)

if mbti == "ISTJ":
    st.write("당신은 :blue[현실주의자] 이시네요.")
elif mbti == "ENTP":
    st.write("당신은 :green[활동가] 이시네요.")
else:
    st.write("당신에 대해 :red[알고 싶어요.]:grey_exclamation:")

#3-3 셀렉트 박스
mbti = st.selectbox(
    "당신의 MBTI는 무엇입니까?",
    ('ISTJ', 'ENTP', '선택지 없음'),
    index = 2
)

if mbti == "ISTJ":
    st.write("당신은 :blue[현실주의자] 이시네요.")
elif mbti == "ENTP":
    st.write("당신은 :green[활동가] 이시네요.")
else:
    st.write("당신에 대해 :red[알고 싶어요.]:grey_exclamation:")

#3-4 멀티 셀렉트 박스 : 리스트 반환
options = st.multiselect(
    "당신이 좋아하는 과일은 무엇인가요?",
    ["망고", "오렌지", "사과", "바나나"],
    ["바나나", "망고"]
)

st.write(f"당신의 선택은 : :red[{options}]입니다.")

#3-5 슬라이더
values = st.slider(
    "범위의 값을 다음과 같이 지정할 수 있어요. :sparkles:",
    0.0, 100.0, (25.0, 75.0)
)

st.write("선택 범위 : ", values)

start_time = st.slider(
    "약속을 언제 잡는 것이 좋을까요?",
    min_value = dt(2025, 3, 1, 0, 0),
    max_value = dt(2025, 3, 30, 23, 0),
    value = dt(2025, 3, 25, 12, 0),
    step = datetime.timedelta(hours = 1),
    format = "MM/DD/YY - HH:mm"
)

st.write("선택한 약속 시간 : ", start_time)

#4. 입력형 위젯
#텍스트 입력
text_input = st.text_input(
    label = "가고 싶은 여행지가 있나요?",
    placeholder = "여행지를 입력해 주세요"
)

st.write(f"당신이 선택한 여행지 : :violet[{text_input}]")

if text_input:
    st.write("입력 내용 : " + text_input)

#숫자 입력
number = st.number_input(
    label = "나이를 입력해 주세요.",
    min_value = 10,
    max_value = 100,
    value = 30,
    step = 5
)

st.write("당신이 입력한 나이는 : ", number)