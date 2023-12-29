import pandas as pd
import numpy as np
import argparse
from numpy import dot
from numpy.linalg import norm
from Recommendation.data_preprocessing import load_data, convert_business_type, get_group, get_threshold


def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))


def cos_sim_based(df: pd.DataFrame, conditional_list: list, thresholds: list):
    feature_df = df[conditional_list]
    cos_sim_list = []
    for i in feature_df.to_numpy():
        cosine_sim = cos_sim(i, np.array(thresholds))
        cos_sim_list.append(cosine_sim)

    item = df['지역']
    df = pd.DataFrame(cos_sim_list, columns=['유사도'])

    df = pd.concat([item, df], axis=1)

    return df


def recommender_sys(business_type, conditional_list, many_list):
    df = load_data()
    df = convert_business_type(df)
    df = get_group(df, business_type)
    thresholds = get_threshold(df, conditional_list, many_list)
    df = cos_sim_based(df, conditional_list, thresholds)
    df = df.sort_values(by=['유사도'], axis=0, ascending=False)

    result = df['지역'].to_numpy()[:5]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--business_type', type=str, default='음식점', help='추천받고자 하는 업종')
    parser.add_argument('--conditional_list', type=list, default=['유사업종수', '방문인구', '근무인구'])
    parser.add_argument('--many_list', type=list, default=[True, True, False])
    args = parser.parse_args()

    result = recommender_sys(args.business_type, args.conditional_list, args.many_list)
    print(result)
