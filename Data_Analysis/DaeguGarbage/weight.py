import os
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler

FILE_PATH = './data/가중치/생활인구 가중치 추출 데이터셋(불법투기및cctv 최종수정본).csv'
SAVE_FILE = './data/전처리_완료/'

live_weights_dict = {
    "주거 환경 지표": 0.53909,
    "환경 관리 지표": 0.46091,
    "거주 주택 유형": 0.33333,
    "거주민 특성": 0.66667,
    "CCTV 개수": 0.16857,
    "쓰레기 무단 투기 건수": 0.83143,
    "내국인": 0.66667,
    "외국인": 0.33333
}

df = pd.read_csv(FILE_PATH, index_col=0)
print('정제전')
print(tabulate(df,headers='keys',tablefmt='github'))

# 값들 스케일링 진행
scaler = MinMaxScaler()
cols = ['내국인 수','외국인 수','주택유형(아파트제외)합계','cctv개수','쓰레기 불법 투기 수']
scaled_df = pd.DataFrame(scaler.fit_transform(df[cols]), columns=[ c  for c in cols])
print('스케일링 데이터프레임')
print(tabulate(scaled_df,headers='keys',tablefmt='github'))

# 행정동에 붙이기 
sihangdong_df = df[['군구','행정동']]
final_scaled_df = pd.concat([sihangdong_df,scaled_df],axis=1)
print('최종 데이터 프레임')
print(tabulate(final_scaled_df,headers='keys',tablefmt='github'))

# 주거환경지표 도출
final_scaled_df['거주민유형']= final_scaled_df['내국인 수']*live_weights_dict['내국인'] +  final_scaled_df['외국인 수'] * live_weights_dict['외국인']
final_scaled_df['주거환경지표'] = final_scaled_df['거주민유형']*live_weights_dict['거주민 특성'] + final_scaled_df['주택유형(아파트제외)합계']*live_weights_dict['거주 주택 유형']

# 환경관리지표 도출
final_scaled_df['환경관리지표'] = final_scaled_df['cctv개수']*live_weights_dict['CCTV 개수'] + final_scaled_df['쓰레기 불법 투기 수']*live_weights_dict['쓰레기 무단 투기 건수']

# 쓰레기입지후보지표 도출
final_scaled_df['총 수요지표'] = final_scaled_df['주거환경지표'] +final_scaled_df['환경관리지표']

# 총 수요지표순으로 정렬
final_scaled_df = final_scaled_df.sort_values(by='총 수요지표',ascending=False).reset_index(drop=True)
print('가중치 적용 후 총 수요지표로 정렬한 데이터프레임')
print(tabulate(final_scaled_df,headers='keys',tablefmt='github'))

# 군구,행정동.총수요지표만 추출
save_final_scaled_df = final_scaled_df[['군구','행정동','총 수요지표']]
print('읍면동별 총 수요지표만 있는 데이터프레임')
print(tabulate(save_final_scaled_df,headers='keys',tablefmt='github'))

# 생활인구 가중치 저장
save_final_scaled_df.to_csv(f'{SAVE_FILE}생활인구_읍면동별_수요가중치_산출완료.csv')