import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from tabulate import tabulate

import os
import time

def makeDir(FILE_PATH):
    if os.path.exists(FILE_PATH):
        print('폴더 이미 존재')
    else: 
        print(f'{FILE_PATH} 존재하지 않아서 하나 생성하겠습니다.')
        time.sleep(2)
        os.makedirs(FILE_PATH, exist_ok=True)
        print('파일 생성 완료')
        
def makeDF(FILE_PATH,countries,debug_mode =True):
    '''
    데이터 파일 불러오기 및 데이터 정제
    '''
    
    vietnam_df = None
    brazil_df = None
    columbia_df = None
    
    files = os.listdir(FILE_PATH)
    for file in files:
        name, ext = os.path.splitext(file)
        if (ext == '.xlsx') & (name != 'coffee'):
            df = pd.read_excel(os.path.join(FILE_PATH,file), parse_dates=['시간(UTC)'])
            if debug_mode:
                print('원래 데이터')
                print(tabulate(df,headers='keys',tablefmt='github'))
                print('데이터 정보')
                print(df.info())
            
            df = df.rename(columns={'시간(UTC)':'시간'})
            df['년도'] = df['시간'].dt.year
            df['월'] = df['시간'].dt.month
            df = df.drop(index=0)
            df = df.drop(columns='시간')
            df = df[['년도', '월', '월평균 일최고기온', '월평균 일최저기온', '최대 일강수량', '월중 최대풍속']]
            df = df.fillna(0).replace(-99.9,0)

            if countries[0] in name: vietnam_df = df
            elif countries[1] in name: brazil_df = df
            elif countries[2] in name: columbia_df = df
 
            
    return [vietnam_df, brazil_df, columbia_df]

def drawGraph(IMG_PATH, temp_sr, country):
    '''
    기후 변화 그래프 그리기
    '''
    makeDir(IMG_PATH)
    x_label = ['2017','2018','2019','2020','2021']
    label = ['연평균 일최고기온', '연평균 일최저기온', '최대 일강수량', '연중 최대풍속']
    
    plt.figure(figsize=(12,8))
    plt.title(f'{country}의 기후 변화')
    plt.plot(x_label,temp_sr, marker='o', label=label)
    plt.grid(True)
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(IMG_PATH+f'{country}기후변화')
    plt.show()
            
def getYearMean(df, years, IMG_PATH, country):
    '''
    연별 기후 평균 구하고 시각화
    '''
    temp_sr = []

    for year in years:
        year_df = df[df['년도']==year]
        temp_mean = round(year_df.iloc[:,2:].mean(axis=0),2)
        year_df = year_df.copy()
        year_df.loc['평균'] = temp_mean
        year_df = year_df.drop(columns='월')
        year_df = year_df.fillna(year)
        year_sr = year_df.loc['평균','월평균 일최고기온':]
        temp_sr.append(year_sr)
        
    # print(temp_sr)
    # print()
    
    drawGraph(IMG_PATH, temp_sr, country)
    
    return temp_sr



if __name__ == '__main__':
    FILE_PATH = './coffee/'
    IMG_PATH = './coffee/image/'
    years = [2017,2018,2019,2020,2021]
    countries = ['베트남','브라질','콜롬비아']
    
    country_list = makeDF(FILE_PATH,countries)
    
    for idx, country in enumerate(country_list):
        if country is not None: getYearMean(country, years, IMG_PATH, countries[idx])
    

    