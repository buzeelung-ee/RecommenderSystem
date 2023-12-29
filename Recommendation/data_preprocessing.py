import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import re


def load_data():
    # 17년도 ~ 20년도 데이터
    temp1 = pd.read_csv('C:/Users/rlawj/sample_DB/Jeju/Recommendation_data/market_17.csv', encoding='cp949')
    temp2 = pd.read_csv('C:/Users/rlawj/sample_DB/Jeju/Recommendation_data/market_18.csv', encoding='cp949')
    temp3 = pd.read_csv('C:/Users/rlawj/sample_DB/Jeju/Recommendation_data/market_19.csv', encoding='cp949')
    temp4 = pd.read_csv('C:/Users/rlawj/sample_DB/Jeju/Recommendation_data/market_20.csv', encoding='cp949')

    result = pd.concat([temp1, temp2, temp3, temp4]).reset_index(drop=True)

    return result


def convert_business_type(df: pd.DataFrame):
    df = df.drop(['년월', '데이터기준일자', '이용금액', '성별'], axis=1)
    for idx, i in enumerate(df['업종명']):
        business_food_find = re.findall('음식점업', i)
        if len(business_food_find) > 0:
            df['업종명'][idx] = '음식점'

    return df


def get_group(df: pd.DataFrame, business_type: str):
    df = df.groupby('업종명').get_group(business_type)
    df.reset_index(drop=True)

    location_info = df['읍면동명'].unique()
    temp = []
    for dong in location_info:
        location_df = df.groupby('읍면동명').get_group(dong)
        city = location_df['시도명'].to_numpy()[0]

        city_dong = f'{city} {dong}'
        location_np = location_df.drop(['시도명', '읍면동명', '업종명'], axis=1).to_numpy()
        location_sum = np.sum(location_np, axis=0).tolist()
        location_sum.append(len(location_np))
        location_sum.insert(0, city_dong)

        temp.append(location_sum)

    df = pd.DataFrame(data=temp, columns=['지역', '이용자수', '거주인구', '근무인구', '방문인구', '유사업종수'])

    return df


def get_threshold(df: pd.DataFrame, columns, many):
    thresholds = []

    for idx, column in enumerate(columns):
        if many[idx]:
            thresholds.append(df[column].quantile(q=0.75))
        else:
            thresholds.append(df[column].quantile(q=0.25))

    return thresholds
