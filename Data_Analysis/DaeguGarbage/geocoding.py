# kakao geocoding 필요 모듈 로딩
import geokakao as gk
from tabulate import tabulate
import pandas as pd
import datetime
import requests
import sys
import json
import os

# kakaogeocoding 함수 정의 - 경도 위도 버전 
# api 불러오기 
def json_request(url='',encoding='utf-8', success=None,
                 error=lambda e: print('%s : %s' % (e,datetime.now()), file=sys.stderr)):
    headers = {'Authorization':'KakaoAK {}'.format(APP_KEY)}
    resp = requests.get(url, headers=headers)
    return resp.text

# json 파일 불러와서 필요한 정보 추출
# longitude = 경도, latitude = 위도 
def reverse_geocode(longitude, latitude):
    url = '%s?x=%s&y=%s' % (URL_GEO, longitude, latitude)
    try:
        json_req = json_request(url=url)
        json_data = json.loads(json_req)
        json_doc = json_data.get('documents')[0]
        json_name = json_doc.get('address_name')
        json_gungu_name = json_doc.get('region_2depth_name')
        json_dong_name = json_doc.get('region_3depth_name')
        json_dong_b_name = json_doc.get('region_4depth_name')
        json_code = json_doc.get('code')
    except:
        json_name = 'NaN'
        json_gungu_name = 'NaN'
        json_dong_name = 'NaN'
        json_dong_b_name = 'NaN'
        json_code = 'NaN'
    return json_name, json_gungu_name, json_dong_name,json_dong_b_name,json_code

# 행정동, 지역코드 추가 
# x = 경도, y= 위도
# 컬럼 이름 경도일시 x로 변경 요함
# row.경도 -> row.x
# 그 반대시 x를 경도 변경 요함 
# row.x -> row.경도
def add_hangdong(df):
    for row in df.itertuples():
        try:
            x = float(row.경도)
            y = float(row.위도)
            # print(x,y)
            _, json_gungu_name, json_dong_name,json_dong_b_name,json_code = reverse_geocode(x, y)
            df.loc[row.Index,'군구'] = json_gungu_name
            df.loc[row.Index,'법정동'] = json_dong_b_name
            df.loc[row.Index,'행정동'] = json_dong_name
            df.loc[row.Index,'지역코드'] = json_code
        except Exception as e:
            print(f"Error at row {row.Index}: {e}")
            
    return df

# address 주소 넣어서 정보 가져오기 
def reverse_geocode_address(address):
    url = URL_ADD + address
    #print(url)
    try:
        json_req = json_request(url=url)
        json_data = json.loads(json_req)
        json_doc = json_data.get('documents')[0]
        json_add_dict = json_doc.get('address')
        json_name = json_add_dict['address_name']
        json_gungu_name = json_add_dict['region_2depth_name']
        json_dong_name = json_add_dict['region_3depth_name']
        json_dong_h_name = json_add_dict['region_3depth_h_name']
        json_code = json_add_dict['h_code']
    except:
        json_name = 'NaN'
        json_gungu_name = 'NaN'
        json_dong_name = 'NaN'
        json_dong_h_name = 'NaN'
        json_code = 'NaN'

    return json_name, json_gungu_name, json_dong_name, json_dong_h_name,json_code


def add_hangdong_v2(df):
    for row in df.itertuples():
        try : 
            address = str(row.새주소)
            _, json_gungu_name, json_dong_name, json_dong_h_name,json_code= reverse_geocode_address(address)
            df.loc[row.Index,'군구'] = json_gungu_name
            df.loc[row.Index,'동단위'] = json_dong_name
            df.loc[row.Index,'행정동'] = json_dong_h_name
            df.loc[row.Index,'지역코드'] = json_code
        except Exception as e:
            print(f'Error at row {row.Index}: {e}')
    return df


if __name__ == '__main__':
    
    # API 정보
    APP_KEY = 'dc11a5937f7be72f1695b69f2495fe71'
    # 경도 위도 정보를 검색해서 가져오는 URL -> 사용 권장
    URL_GEO = 'https://dapi.kakao.com/v2/local/geo/coord2regioncode.json'
    # 주소 정보를 검색해서 가져오는 URL 
    URL_ADD = 'https://dapi.kakao.com/v2/local/search/address.json?query=' 
    
    DIR_PATH = './data/전처리용/'
    file_list = os.listdir(DIR_PATH)

    for file in file_list: 
        print(file)
        name,ext = os.path.splitext(file)
        FILE_PATH = os.path.join(DIR_PATH,file)
        if (ext == '.csv'):
            df = pd.read_csv(FILE_PATH, encoding='utf-8')
            #df = df.rename(columns={'소재지지번주소':'주소'})
            df['새주소'] = df['새주소'].str.rstrip()
            df['새주소'] = df['새주소'].str.lstrip()
            print(tabulate(df.head(),headers='keys',tablefmt='github'))
            new_df = add_hangdong_v2(df)
            print(new_df.info())
            print(new_df.isna().sum())
            new_df.to_csv(f'./data/전처리_완료/{name}_행정동추가_주소버전.csv')
            print(f'{file} 행정동 추가 파일 저장완료')
            
        elif (ext == '.xlsx'):
            df = pd.read_excel(FILE_PATH, engine='openpyxl')
            #df = df.rename(columns={'소재지지번주소':'주소'})
            print(tabulate(df.head(),headers='keys',tablefmt='github'))
            new_df = add_hangdong_v2(df)
            print(new_df.info())
            print(new_df.isna().sum())
            new_df.to_csv(f'./data/전처리_완료/{name}_행정동추가.csv')
            print(f'{file} 행정동 추가 파일 저장완료')