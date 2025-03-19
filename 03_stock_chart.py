import datetime
import FinanceDataReader as fdr
import pandas as pd
import streamlit as st

st.title("종목 차트 검색")

with st.sidebar:
    code = st.text_input(
        '종목코드',
        value = '',
        placeholder = '종목코드를 입력해 주세요.'
    )

    date = st.date_input(
        '조회 시작일을 선택해 주세요.',
        datetime.datetime(2022, 1, 1)
    )
    
chart, data = st.tabs(["차트", "데이터"])

with data:
    #한국거래소 상장종목 전체 가져오기
    df = fdr.StockListing("KRX")
    st.dataframe(df)

with chart:
    if code and date:
        df = fdr.DataReader(code, date)
        #날짜를 기준으로 종가를 가져옴
        data = df.sort_index(ascending = True).loc[:, 'Close']
        st.line_chart(data)
    
with st.expander('칼럼 설명'):
        st.markdown('''
        - Open: 시가
        - High: 고가
        - Low: 저가
        - Close: 종가
        - Volumn: 거래량
        - Adj Close: 수정 종가
        ''')