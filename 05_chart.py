import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import streamlit as st
import seaborn as sns
from matplotlib import font_manager, rc

#방법1
#한글 깨짐 해결
#path = "c:/Windows/Fonts/malgun.ttf"

#if platform.system() == 'Windows':
#    font_name = font_manager.FontProperties(fname=path).get_name()
#    rc('font', family=font_name)
#    plt.rcParams['axes.unicode_minus'] = False
#else:
#    print("한글 폰트 불러오기 실패")

#방법2
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

#DataFrame 생성
data = pd.DataFrame({
    '이름': ['영식', '철수', '영희'],
    '나이': [22, 31, 25],
    '몸무게': [75.5, 80.2, 55.1]
})

st.dataframe(data, use_container_width = True)

fig, ax = plt.subplots()
ax.bar(data['이름'], data['나이'])
st.pyplot(fig)

barplot = sns.barplot(x = '이름', y = '나이', data = data, ax = ax, palette = 'Set2')
fig = barplot.get_figure()
st.pyplot(fig)