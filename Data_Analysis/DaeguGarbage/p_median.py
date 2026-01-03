import os
import time
from pathlib import Path

import pandas as pd
from tabulate import tabulate 
import folium
import geopandas as gpd


import pulp
from math import sqrt
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist

def living_population_p_median(demand_df, cand_df, p_num, WEIGHT=False ):
    
    """
        생활인구 기반 p-Median 문제 풀이 및 시각화
        gungu : 군구명 (예: '수성구')
        p_num : 설치 가능한 후보지 개수
        m_num : 반경(m) 단위 (폴더 구조 맞추기용)
    """
    # 거리 행렬 계산
    print('거리 행렬 계산 시작')
    d_coords = demand_df[['x', 'y']].to_numpy()
    c_coords = cand_df[['x', 'y']].to_numpy()
    dist_matrix_np = cdist(d_coords, c_coords, metric='euclidean')
    
    # dict 형태로 변환
    # 전부 고려하는 방식이어서 i 하나에 j를 모두 고려하는 형식으로 dict이 생성됨
    dist_matrix = {
        (demand_df.iloc[i].id, cand_df.iloc[j].id): dist_matrix_np[i, j]
        for i in range(len(demand_df))
        for j in range(len(cand_df))
    }
    print('거리 행렬 계산 완료')
    
    # 문제 정의
    p = p_num
    prob = pulp.LpProblem(f"p_Median_p_{p}", pulp.LpMinimize)

    # 변수 정의
    # x (후보지) : 바이너리 형태, 0 혹은 1의 값을 가짐
    # y (수요지, 후보지) : 바이너리 형태이며, 0 혹은 1의 값을 가짐. 
    #                  수요가 충족된 후보지를 의미
    x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in cand_df['id']}
    y = {(d.id, c.id): pulp.LpVariable(f"y_{d.id}_{c.id}", cat="Binary")
         for _, d in demand_df.iterrows()
         for _, c in cand_df.iterrows()}
    print('변수 정의 완료!')

    if WEIGHT: 
        # 가중치 있는 목적함수: 총 거리 최소화 (가중치 × 거리 × 할당 여부)
        # prob += pulp.lpSum(
        #     demand_df.loc[demand_df['id'] == d, 'w'].values[0] * dist_matrix[(d, c)] * y[(d, c)]
        #     for d in demand_df['id'] for c in cand_df['id']
        # )
        # id : w(가중치) 매핑 딕셔너리 생성
        weight_dict = dict(zip(demand_df['id'], demand_df['w']))

        # 가중치 있는 총 거리 최소화 목적함수
        prob += pulp.lpSum(
            weight_dict[d] * dist_matrix[(d, c)] * y[(d, c)]
            for d in demand_df['id'] for c in cand_df['id']
        )
    else: 
        # 가중치 없는 목적함수 : 총 거리 최소화 (거리 x 할당여부 )
        prob += pulp.lpSum(dist_matrix[(d, c)] * y[(d, c)] for d in demand_df['id'] for c in cand_df['id'])
    print('목적함수 정의 완료!')

    # 제약조건 ①: 각 수요지는 반드시 하나의 후보지에 할당 -> 수요지를 전부 고려하는 방향
    for d in demand_df['id']:
        prob += pulp.lpSum(y[(d, c)] for c in cand_df['id']) == 1
    print('제약조건1 추가 완료!')

    # 제약조건 ②: 선택되지 않은 후보지에는 할당 불가
    for d in demand_df['id']:
        for c in cand_df['id']:
            prob += y[(d, c)] <= x[c]
    print('제약조건2 정의 완료!')

    # 제약조건 ③: 쓰레기통 설치 가능한 후보지 수 = p 
    prob += pulp.lpSum(x[c] for c in cand_df['id']) == p
    print('제약조건3 정의 완료!\n')

    # 최적화 수행
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    print('최적화 수행 완료!')

    # 결과 추출
    selected_cands = [c for c in cand_df['id'] if pulp.value(x[c]) >= 0.5] # 선택된 후보지들
    # 후보지로부터 수요지까지의 거리 합 평균이 가장 작은 수요지들
    # 후보지부터 수요지까지 거리 합이 최소화 되는 모든 위치 
    allocation = {
        d: min(
            cand_df['id'],
            key=lambda c: dist_matrix[(d, c)] if pulp.value(y[(d, c)]) >= 0.5 else float("inf")
        )
        for d in demand_df['id']
    }
    print('결과 추출 완료!')
    return demand_df, cand_df, selected_cands, allocation

def makeGeodf(demand_df,cand_df, selected_cands, allocation):
    '''
        GeoDataFrame으로 변환
        demand_df : 수요지점 데이터프레임
        cand_df : 후보지점 데이터프레임
        selected_cand : 선정된 후보지
        allocation : 평균합이 최소값에 드는 수요지들
    '''
    demand_gdf = gpd.GeoDataFrame(
        demand_df, geometry=gpd.points_from_xy(demand_df.x, demand_df.y), crs="EPSG:5186"
    )
    cand_gdf = gpd.GeoDataFrame(
        cand_df, geometry=gpd.points_from_xy(cand_df.x, cand_df.y), crs="EPSG:5186"
    )

    cand_gdf["selected"] = cand_gdf["id"].apply(lambda i: 1 if i in selected_cands else 0)
    demand_gdf["assigned"] = demand_gdf["id"].apply(lambda i: allocation[i] if i in allocation else None)
    
    print('GeoDataFrame으로 변환 완료!')
    
    return demand_gdf,cand_gdf

def drawMap(demand_gdf,cand_gdf,gungu,p_num,meter,SAVE_PATH):
    
    demand_gdf4326 = demand_gdf.to_crs(epsg=4326)
    cand_gdf4326 = cand_gdf.to_crs(epsg=4326)

    gdf4326_list = [demand_gdf4326, cand_gdf4326]

    center = [demand_gdf4326.geometry.y.mean(), demand_gdf4326.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=12)

    # 후보지 (선정된 것만)
    for _, row in cand_gdf4326[cand_gdf4326["selected"] == 1].iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"선정된 후보지 {row['id']}",
            icon=folium.Icon(color="red", icon="ok-sign")
        ).add_to(m)

    # 수요지 및 할당 관계 표시
    for _, row in demand_gdf4326.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color="darkred",
            fill=True,
            popup=f"수요지 {row['id']} → 후보지 {row['assigned']}"
        ).add_to(m)

        # 연결선
        assigned_id = row["assigned"]
        if assigned_id is not None:
            cand_row = cand_gdf4326[cand_gdf4326["id"] == assigned_id].iloc[0]
            folium.PolyLine(
                locations=[[row.geometry.y, row.geometry.x],
                            [cand_row.geometry.y, cand_row.geometry.x]],
                color="red",
                weight=2,
                opacity=0.7
            ).add_to(m)


    m.save(f"{SAVE_PATH}/{gungu}/p-median_result_{gungu}_{p_num}개_{meter}_map.html")
    print(f"[{gungu}] 지도 저장 완료")
    
    return gdf4326_list


def getWgs(gdf4326_list):
    result = []
    for i, df in enumerate(gdf4326_list):
        df['WGS4326_x']=df.geometry.x
        df['WGS4326_y']=df.geometry.y
        if i == 0 : cols = ['id', '군구', '행정동', 'x', 'y', 'WGS4326_x','WGS4326_y','assigned']
        else: cols = ['id', 'x', 'y', 'WGS4326_x','WGS4326_y','selected']
        df = df[cols]
        result.append(df)

    return result

def saveResult(gdf4326_list, gungu, p_num, meter,ca, DIR_PATH):

    """
        정제된 gdf4326_list(demand_gdf4326, cand_gdf4326)를 각각 csv 파일로 저장
    """
    filenames = ['수요충족', '선택된후보지']
    filenames = filenames = [f'{DIR_PATH}/결과값/{gungu}/{gungu}_{p_num}개_{meter}_{name}_{ca}.csv' for name in filenames]

    for df, path in zip(gdf4326_list, filenames):
        df.to_csv(path, index=False)
        print(f"저장 완료: {path}")
         

if __name__ == '__main__':
    
    SAVE_PATH = Path('./data/P_Median/시각화')
    DIR_PATH = Path('./data/P_Median')
    DEMAND_DIR_PATH = DIR_PATH / '수요지점'
    CAND_DIR_PATH = DIR_PATH / '입지후보지'
    
    # 시군구 목록
    gungu_list = ['동구']
    # 쓰레기통 개수 리스트 (5배수)
    p_num_list = [ 10, 15, 20, 25 ]
    # 사용할 커버리지 반경 폴더
    meter = '150m'
    
    # 사용할 가중치 파일
    category = '쓰레기및cctv_수요지점_가중치추가'
    
    for p_num in p_num_list:
        print(f'[ 쓰레기통 개수: {p_num}개 ]\n')
        # 쓰레기통 개수
        # p_num = 10
                       
        for gungu in gungu_list:
            print(f'{gungu} p-median 문제 풀기 시작')
            start_time = time.time() # 시작 시간 기록
            
            # 후보지점 파일 읽기
            for cand_file in (CAND_DIR_PATH/meter).glob("*.csv"):
                if cand_file.name.startswith(gungu):
                    cand_df = pd.read_csv(cand_file,index_col=0)
                    print(cand_file.name,': 입지후보지 데이터 프레임')
                    print(tabulate(cand_df.head(), headers='keys',tablefmt='github'))
            
            # 수요지점 파일 읽기   
            for demand_file in (DEMAND_DIR_PATH/gungu).glob('*.csv'):
                if (category in demand_file.name):
                    print(f'{demand_file.name}: 수요지점 데이터프레임')
                    demand_df = pd.read_csv(demand_file,index_col=0)
                    print(tabulate(demand_df.head(), headers='keys',tablefmt='github'))
            
            #p-median 문제 계산 및 시각화
            demand_df, cand_df, selected_cands, allocation = living_population_p_median(demand_df, cand_df, p_num, WEIGHT=False)
            demand_gdf,cand_gdf = makeGeodf(demand_df,cand_df,selected_cands,allocation)
            gdf4326_list = drawMap(demand_gdf,cand_gdf,gungu,p_num,meter,SAVE_PATH)
            
            # 결과값들 저장하기
            gdf4326_list = getWgs(gdf4326_list)
            saveResult(gdf4326_list,gungu,p_num,meter,ca='가중치없음',DIR_PATH=DIR_PATH)
            
            # 종료 시간 및 경과 시간 계산
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60) # 180초 // 60 = 0(초)
            seconds = int(elapsed_time % 60) # 180초 % 60 = 3(분)
            print(f" {gungu} p-median 문제 풀이 완료 — 소요 시간: {minutes}분 {seconds}초")
            print('-'*100)
            



