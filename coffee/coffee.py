import pandas as pd
from tabulate import tabulate

import re
import os 
import time

import matplotlib.pyplot as plt
import koreanize_matplotlib

def makeDF(FILE_PATH):
    '''
    데이터셋 데이터프레임으로 만들고 정보확인
    '''
    coffee_df = pd.read_excel(FILE_PATH)
    print(tabulate(coffee_df,headers='keys',tablefmt='github'))
    print('데이터 프레임 정보')
    print(coffee_df.info())
    print('데이터 프레임 중복값 확인')
    print(coffee_df.duplicated().sum())
    
    return coffee_df

def makeDir(FILE_PATH):
    if os.path.exists(FILE_PATH):
        print('폴더 이미 존재')
    else: 
        print(f'{FILE_PATH} 존재하지 않아서 하나 생성하겠습니다.')
        time.sleep(2)
        os.makedirs(FILE_PATH)
        print('파일 생성 완료')
        
def checkLastNum(df, column, name):
    cnt = 0
    for idx in df[column]:
        if idx == name: cnt += 1
    print(f'{column}-{name}의 마지막 인덱스 번호 {cnt}')
    return cnt

def cleningDF(df):
    '''
    데이터 정제 함수
    '''
    df['항목'] = df['항목'].ffill()
    print('항목 채운 뒤 df'+'\n')
    print(tabulate(df,headers='keys',tablefmt='github'))
    
    last_index = checkLastNum(df,column='항목',name='생커피원두 (M/T)')
    df.loc[:last_index,'항목'] = '원두생산량'
    print(df.loc[:77,'항목'])
    print(f'원두생산량으로 이름 전환 완료')
    
    df = df.replace('-',0)
    print('데이터 타입 확인')
    print(df.dtypes)
    print('"-"을 0으로 채우기 완료')
    
    return df, last_index

def sumCoffeebean(df, last_index):
    '''
    커피원두 생산량 관련 표 구한 후 총 생산량 구하기 (2017~2022)
    '''
    coffee_bean_df = df.loc[:last_index]
    # print('정제 전')
    # print(tabulate(coffee_bean_df.head(30),headers='keys',tablefmt='github'))
    coffee_bean_df = coffee_bean_df.drop(columns='항목', index=0)
    coffee_bean_sum = coffee_bean_df.loc[:,'2012':].sum(axis=1)
    coffee_bean_df['합계'] = coffee_bean_sum
    
    print('합계 확인')
    print(tabulate(coffee_bean_df.head(30),headers='keys',tablefmt='github'))
    
    return coffee_bean_df

def checkTop3City(df):
    '''
    생산량 많은 상위 3개 국가 구하기 (2017 ~ 2021)
    '''
    coffee_bean_df = df.sort_values('합계', ascending=False).reset_index(drop=True)
    print(tabulate(coffee_bean_df.head(10), headers='keys', tablefmt='github'))
    top5_coffee_bean_df = coffee_bean_df.iloc[:3,:]
    top5_coffee_bean_df = top5_coffee_bean_df.drop(columns=['합계','2012','2013','2014','2015','2016']).reset_index(drop=True)
    
    print('커피 생산량 상위 3개 국가')
    print(tabulate(top5_coffee_bean_df,headers='keys',tablefmt='github'))
    
    return top5_coffee_bean_df

def drawGraph(df, IMG_PATH):
    
    makeDir(IMG_PATH)
    years = df.columns[1:]
    
    countries = df['국가별']
    production_data = df.loc[:,'2017':'2021']
    
    plt.figure(figsize=(12,8))
    for idx, country in enumerate(countries):
        plt.plot(years,production_data.iloc[idx], label=country, marker='o')
        
    plt.title('커피 원두 생산량 (2017-2021)',fontsize=15)
    plt.xlabel('연도')
    plt.ylabel('생산량(톤)')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMG_PATH+'커피원두생산량')
    plt.show()
    
if __name__ == '__main__':
    FILE_PATH = './coffee/coffee.xlsx'
    IMG_PATH = './coffee/image/'
    
    coffee_df = makeDF(FILE_PATH)
    coffee_df, last_index = cleningDF(coffee_df)
    coffee_df = sumCoffeebean(coffee_df, last_index)
    top5_coffee_bean_df = checkTop3City(coffee_df)
    
    drawGraph(top5_coffee_bean_df,IMG_PATH)
    
       