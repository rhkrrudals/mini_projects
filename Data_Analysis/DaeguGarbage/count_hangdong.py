import pandas as pd
import os 
import re 
import matplotlib.pyplot as plt
from tabulate import tabulate

def sepHangdpng(df,file):
    '''
        테이블 내 기본키 확인 후 변경 요함
        현 코드 내 id에 해당되는 것
    '''
    print('원본 데이터 테이블/n')
    print(tabulate(df.head(),headers='keys',tablefmt='github'))
    # 행정동별 데이터 분류 함수
    if '시군구명' in df.columns: 
        print(file)
        new_df = df.groupby(by=['시군구명','행정동']).count().reset_index()[['시군구명','행정동','id']]
        new_df = new_df.rename(columns={'id':'합계'})
        new_df.to_csv(f'./data/전처리_완료/{file}_합계추가.csv')
        #print(new_df['군구'].value_counts())
        print(tabulate(new_df.head(),headers='keys',tablefmt='github'))
    else : 
        
        print(file)
        new_df = df.groupby(by=['군구','행정동']).count().reset_index()[['군구','행정동','id']]
        new_df = new_df.rename(columns={'id':'합계'})
        new_df.to_csv(f'./data/전처리_완료/{file}_합계추가.csv')
        print(tabulate(new_df.head(),headers='keys',tablefmt='github'))
        

if __name__ == '__main__':
    DIR_PATH = './data/전처리용/'
    file_list = os.listdir(DIR_PATH)

    for file in file_list: 
        filename = os.path.join(DIR_PATH,file)
        file, ext = os.path.splitext(file)
        
        # 데이터 전처리
        if (ext == '.csv') :
            print(file)
            df = pd.read_csv(filename, index_col=0, encoding='utf-8')
            drop_cols = df.columns[df.columns.str.startswith('Un')]
            df = df.drop(columns=drop_cols)
            sepHangdpng(df,file)
            print('전처리 완료')