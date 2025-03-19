import streamlit as st

#session_state 사용 전
#counter = 0
#button = st.button('Click')

#if button:
#    counter += 1

#st.write(f'버튼을 {counter}번 클릭하였습니다.')

#session_state 사용 후
if "counter" not in st.session_state:
    st.session_state.counter = 0

button = st.button('Click')

if button:
    st.session_state.counter += 1

st.write(f'버튼을 {st.session_state.counter}번 클릭하였습니다.')