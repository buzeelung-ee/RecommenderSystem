import os

temp = os.system('python recommender.py --business_type="음식점" --conditional_list="유사업종수 방문인구 근무인구" --many_list="True True False"')
print(temp)
