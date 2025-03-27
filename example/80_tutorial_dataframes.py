import numpy as np
import pandas as pd
import streamlit as st
from faker import Faker  #가짜 데이터 생성을 위한 Faker 라이브러리

#데이터 캐싱을 위한 데코레이터
@st.cache_data
def get_profile_dataset(number_of_items: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    가짜 사용자 프로필 데이터를 생성하는 함수
    :param number_of_items: 생성할 프로필 수
    :param seed: 랜덤 시드 값 (재현 가능성 보장)
    :return: pandas DataFrame 형태의 사용자 프로필 데이터
    """
    new_data = []
    fake = Faker()
    np.random.seed(seed)    #NumPy 랜덤 시드 설정
    Faker.seed(seed)        #Faker 랜덤 시드 설정

    for i in range(number_of_items):
        profile = fake.profile()
        new_data.append(
            {
                "name": profile["name"],                        #사용자 이름
                "daily_activity": np.random.rand(25),           #최근 25일 동안의 활동량 (0~1 사이 랜덤 값)
                "activity": np.random.randint(2, 90, size=12),  #최근 1년간의 월별 활동량 (2~90 사이 랜덤 값)
            }
        )

    profile_df = pd.DataFrame(new_data)
    return profile_df

#데이터프레임 컬럼 설정
column_configuration = {
    "name": st.column_config.TextColumn(
        "Name", help="The name of the user", max_chars=100, width="medium"
    ),
    "activity": st.column_config.LineChartColumn(
        "Activity (1 year)",
        help="The user's activity over the last 1 year",
        width="large",
        y_min=0,
        y_max=100,
    ),
    "daily_activity": st.column_config.BarChartColumn(
        "Activity (daily)",
        help="The user's activity in the last 25 days",
        width="medium",
        y_min=0,
        y_max=1,
    ),
}

#두 개의 탭 생성 : Select members (멤버 선택) / Compare selected (선택된 멤버 비교)
select, compare = st.tabs(["Select members", "Compare selected"])

with select:
    st.header("All members")        #전체 멤버 목록 헤더

    df = get_profile_dataset()      #가짜 프로필 데이터 가져오기

    #데이터프레임을 Streamlit UI에 표시 (선택 가능)
    event = st.dataframe(
        df,
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",              #선택 시 페이지 새로고침
        selection_mode="multi-row",     #여러 행 선택 가능
    )

    st.header("Selected members")       #선택된 멤버 목록 헤더
    people = event.selection.rows       #사용자가 선택한 행의 인덱스 리스트
    filtered_df = df.iloc[people]       #선택된 행만 필터링하여 새로운 데이터프레임 생성
    st.dataframe(
        filtered_df,
        column_config=column_configuration,
        use_container_width=True,
    )

with compare:
    #선택된 멤버들의 연간 활동 데이터 저장할 딕셔너리
    activity_df = {}
    for person in people:
        activity_df[df.iloc[person]["name"]] = df.iloc[person]["activity"]
    activity_df = pd.DataFrame(activity_df)  #데이터프레임 변환

    #선택된 멤버들의 일일 활동 데이터 저장할 딕셔너리
    daily_activity_df = {}
    for person in people:
        daily_activity_df[df.iloc[person]["name"]] = df.iloc[person]["daily_activity"]
    daily_activity_df = pd.DataFrame(daily_activity_df)  # 데이터프레임 변환

    #선택된 멤버가 있으면 활동 비교 그래프 출력
    if len(people) > 0:
        st.header("Daily activity comparison")
        st.bar_chart(daily_activity_df)         #최근 25일 동안의 활동량 비교 (Bar Chart)
        st.header("Yearly activity comparison")
        st.line_chart(activity_df)              #최근 1년 동안의 활동량 비교 (Line Chart)
    else:
        st.markdown("No members selected.")     #선택된 멤버가 없을 경우 메시지 출력
