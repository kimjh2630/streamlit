import datetime
import streamlit as st
import random

def generate_lotto():
    lotto = set()
    while len(lotto) < 6:
        number = random.randint(1, 45)
        lotto.add(number)
    lotto = list(lotto)
    lotto.sort()
    return lotto

#제목 설정
st.title(":sparkles: 로또 진짜 생성기 :sparkles:")

my_button = st.button("로또를 생성해주세요.")

if my_button:
    lotto_num = set()       #중복체크를 위한 SET 생성
    count = 0               #생성된 로또번호세트

    for i in range(1, 6):   #5회 출력
        lotto_numbers = tuple(sorted(random.sample(range(1, 46), 6)))
        #중복 체크
        if lotto_numbers not in lotto_num:      #중복된 번호가 아니라면 추가
            lotto_num.add(lotto_numbers)        #생성된 로또 번호 세트를 SET에 추가
            count += 1                          #생성된 개수 증가

            #홀수 번째는 파란색, 짝수 번째는 초록색으로 출력
            if i % 2 ==0:
                st.subheader(f"{i}. 행운의 번호 : :green[{lotto_numbers}]")
            else:
                st.subheader(f"{i}. 행운의 번호 : :blue[{lotto_numbers}]")
    st.write(f'생성된 시각 : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

button = st.button("로또를 생성해주세요!")

if button:
    for i in range(1, 6):
        st.subheader(f'{i}. 행운의 번호: :green[{generate_lotto()}]')
    st.write(f"생성된 시각: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")


