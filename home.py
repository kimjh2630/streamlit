import streamlit as st
import mysql.connector
import pandas as pd

st.set_page_config(page_title="Recycling Dashboard", page_icon=":chart:", layout="wide")

st.title(":bar_chart: 유입 경로 및 지역별 방문자 분석")

# ✅ MySQL 데이터 로드 함수
def load_data(query):
    secrets = st.secrets["mysql"]
    conn = mysql.connector.connect(
        host=secrets["host"],
        database=secrets["database"],
        user=secrets["user"],
        password=secrets["password"]
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ 데이터 로드
df_on = load_data("SELECT * FROM ontbl;")
df_off = load_data("SELECT * FROM offtbl;")

st.success("데이터 로드 완료!")

# ✅ 데이터 샘플 표시
st.subheader(":floppy_disk: 샘플 데이터")
on_data, off_data = st.tabs(["온라인 데이터", "오프라인 데이터"])

with on_data:
    st.dataframe(df_on)

with off_data:
    st.dataframe(df_off)

st.divider()
st.write("📌 좌측 사이드바를 통해 **온라인 / 오프라인 데이터 분석 및 머신러닝 예측** 페이지로 이동하세요!")
