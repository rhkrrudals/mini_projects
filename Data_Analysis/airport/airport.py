import pandas as pd
import numpy as np
from tabulate import tabulate

import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

def makeDir(FILE_PATH):
    if os.path.exists(FILE_PATH):
        print('폴더 이미 존재')
    else: 
        print(f'{FILE_PATH} 존재하지 않아서 하나 생성하겠습니다.')
        time.sleep(2)
        os.makedirs(FILE_PATH)
        print('파일 생성 완료')
        
        
def getFile(DIR_PATH):
    file_list = os.listdir(DIR_PATH)
    file_list = sorted(file_list)
    
    allDF = []
    
    for file in file_list:
        _, ext = os.path.splitext(file)
        if (ext =='.xlsx') & (file != '항공사매출.xlsx'):
            df = pd.read_excel(DIR_PATH+file, engine='openpyxl', header=2)
            # 합계 행 삭제
            df = df.drop(index=[0,1,4])
            allDF.append(df)
        elif ext == '.csv':
            airline_df = pd.read_csv(DIR_PATH+file)
            # print('항공사별 화물기 표')
            # print(tabulate(airline_df, headers='keys',tablefmt='github'))

    return allDF, airline_df


def showInfo(df):
    print(tabulate(df, headers='keys', tablefmt='github',showindex=True))
    print('\n')
    print('DataFrame 정보')
    print(df.info())
    print('\n')
    print('중복값 확인')
    print(df.duplicated().sum())
    

def makeDF(allDF,years):
    '''
    데이터 정제하기
    '''
    new_airport = ['플라이강원(FGW)','에어프레미아(APZ)', '에어로케이(EOK)']
    airportDF = pd.concat(allDF, keys=years)
    
    if any(col == '구분' for col in airportDF.columns):
        airportDF = airportDF.rename(columns={'구분':'규모','구분.1':'항공사'})
        airportDF = airportDF.fillna(0)
        airportDF = airportDF.query('항공사 not in @new_airport')
        # aiportDF = aiportDF[~airportDF['항공사'].isin(new_airport)]
        airportDF = airportDF.reset_index()
        airportDF = airportDF.drop(columns='level_1')
        airportDF = airportDF.rename(columns={'level_0':'연도'})
        
        big_aiportDF = airportDF[airportDF['규모'] == '대형 국적사']
        small_aiportDF = airportDF[airportDF['규모'] != '대형 국적사']
        START = True
        while START:
            answer = input('대형 국적사 정보(b), 저비용 항공사 정보(s), 총 합계(a)')
            if answer == 'b' or answer == 'B':
                print('대형 국적사')
                showInfo(big_aiportDF)
                START = False
                return big_aiportDF
            elif answer == 's' or answer =='S':
                print('저비용 항공사')
                showInfo(small_aiportDF)
                START = False
                return small_aiportDF
            elif answer == 'a' or answer == 'A': 
                print('항공사업 전체')
                showInfo(airportDF)
                START = False
                return airportDF
            else: print('다시 입력해 주세요')
            
    showInfo(airportDF)
    
    return airportDF


def getCorr(airportDF, years):
    '''
    연도별로 상관관계 각 지표들간 상관관계 도출
    '''
    allAirportCorr = []
    for year in years:
        airportDF_loc = airportDF[airportDF['연도'] == year]
        aiportDF_corr = airportDF_loc.corr(numeric_only=True)
        aiportDF_corr = aiportDF_corr.loc['공급(석)':'화물(톤)','매출액(억원)':'영업이익률(%)']
        allAirportCorr.append(aiportDF_corr)
        
    return allAirportCorr

def checkCorr(allAirportCorr, columns):
    
    corrDF = {}
    
    for col in columns:
        aiport_corr = allAirportCorr.xs(col,level=1)
        corrDF[col] = aiport_corr
        print(f'{col} 상관관계')
        print(tabulate(aiport_corr, headers='keys', tablefmt='github'))
        
    return corrDF

def drawGraph(corrDF,FILE_PATH):
    
    makeDir(FILE_PATH)
    
    for idx, key in enumerate(corrDF):
        plt.plot(corrDF[key].index, corrDF[key].values)
        plt.title(key)
        plt.legend(corrDF[key].columns)
        plt.xlabel('년도')
        plt.ylabel('수치')
        plt.savefig(FILE_PATH+f'{key}상관관계')
        plt.show()
        

def drawAiportGraph(aiportDF,years,columns,IMG_PATH):
    makeDir(IMG_PATH)
    for col in columns:
        fig, axes = plt.subplots(nrows=1, ncols=len(years), figsize=(30,6), sharey=True)
        for i, year in enumerate(years):
            year_airportDF = aiportDF[aiportDF['연도']==year]
            
            color = sns.color_palette('pastel6', len(year_airportDF['항공사']))
            bars = axes[i].bar(year_airportDF['항공사'], year_airportDF[col], width=0.4, color=color)
            
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x()+bar.get_width()/2, height, height, ha='center', va='bottom', fontsize=7)
                
            axes[i].set_title(f'{year}년', fontsize=14)
            
            if i == 0 : axes[i].set_ylabel(col,fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            
        fig.suptitle(f'항공사별 {col} 연도별 비교', fontsize=15)
        fig.supxlabel('항공사', fontsize=15)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(IMG_PATH+f'항공사별 {col} 연도별 비교')
        plt.show()     
        
def drawAirlineGraph(airline_df, IMG_PATH):
    years = ariline_df.columns[1:]
    airports = airline_df['항공사'].values
    # colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0']
    
    for year in years:
        vals = airline_df.loc[:,year].values
        plt.figure(figsize=(5,5))
        plt.pie(vals,labels=airports, autopct='%.1f%%',colors=plt.cm.Set2.colors)
        plt.title(f'{year}년도 항공사별 화물기 비율')
        plt.tight_layout()
        plt.savefig(IMG_PATH+f'{year}년 항공사별 화물기 비율')
        plt.show()
        

if __name__ == '__main__':
    DIR_PATH = './airport/'
    IMG_PATH = './airport/image/'

    columns = ['공급(석)','운항(편)','여객(명)','화물(톤)']
    years = ['2019','2020','2021','2022','2023']

    # 전체 수치 데이터프레임 만들기
    allDF, ariline_df = getFile(DIR_PATH)
    airportDF= makeDF(allDF,years)
    drawAiportGraph(airportDF,years,columns,IMG_PATH)
    
    # 주요 지표들간 상관관계 데이터프레임 만들기
    allAirportCorr = getCorr(airportDF,years)
    allAirportCorr = makeDF(allAirportCorr,years)
    
    # 각 주요지표들 상관관계 그래프 뽑기 
    corrDF = checkCorr(allAirportCorr,columns)
    drawGraph(corrDF,IMG_PATH)
    
    # 항공사별 화물기 비율 그래프 뽑기
    drawAirlineGraph(ariline_df,IMG_PATH)
