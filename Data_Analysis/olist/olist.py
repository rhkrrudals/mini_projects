import os
import pymysql

import pandas as pd
from tabulate import tabulate
import seaborn as sns 
import matplotlib.pyplot as plt
import koreanize_matplotlib

def connection(host='127.0.0.1',user='root', password='1234',db='User'):
    conn = pymysql.connect(host=host, 
                           user=user, 
                           password=password,
                           db=db, 
                           charset='utf8')
    return conn

def getQuery(connection,query):
    conn = connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
        
def makeDir(IMG_PATH):
    if os.path.exists(IMG_PATH): print(f'폴더가 존재합니다.')
    else:
        print('폴더가 존재하지 않아 생성합니다.')
        os.makedirs(IMG_PATH)
        print('폴더 생성 완료!')

def makeDF(rows,columns):
    df = pd.DataFrame(rows,columns=columns)
    print("[데이터 프레임 변환 완료]")
    print(tabulate(df,headers='keys', tablefmt='github'))
    return df

def clensingDF(df,col, start_idx, end_idx):
    df.loc[start_idx:end_idx,'origin']= 'unknown'
    final_df = df.groupby(by='origin').sum().reset_index().sort_values(by=col,ascending=False).reset_index(drop=True)
    print('\n[" " -> "unkown"으로 정제 완료]')
    print(tabulate(final_df,headers='keys',tablefmt='github'))
    return final_df
    
def getDF(connection, columns, query, col='CVR(%) by orgin', start_idx=0, end_idx=0):
    rows = getQuery(connection,query)
    df = makeDF(rows,columns)
    final_df = clensingDF(df,col=col,start_idx=start_idx,end_idx=end_idx)
    return final_df

def drawCvrGraph(df, IMG_PATH):
    makeDir(IMG_PATH)
    
    def twoGraph(y2,y3):
            plt.title(f"{x} 퍼널 분석")
            ax.barh(f'cd_{x}', y2, label='Closed Deals', color='pink', height=0.5, align='center')  # 가운데 배치
            ax.text(y2+0.3,1,str(y2),fontweight='regular')
            ax.barh(f'mql_{x}', y3, label='MQL Count', color='skyblue', height=0.5, align='center')  # 왼쪽 배치
            ax.text(y3+0.3,2,str(y3),fontweight='regular')
   
    for i in range(len(df)):
        fig, ax = plt.subplots(figsize=(13, 2)) 
        x = df['origin'][i]
        y2 = df['closed_deals_cnt'][i]
        y3 = df['mql_cnt'][i]
            
        if 'seller_cnt' in df.columns:
            y1 = df['seller_cnt'][i]
            ax.barh(f'seller_{x}', y1, label='Active Seller', color='gold', height=0.5, align='center')  # 왼쪽 배치
            ax.text(y1+0.3,0,str(y1),fontweight='regular')
            twoGraph(y2,y3)
            plt.savefig(f'{IMG_PATH}{x} 잠재고객_유입고객_활동고객_전환율')
            
        else:
            twoGraph(y2,y3)
            plt.savefig(f'{IMG_PATH}{x} 잠재고객_유입고객_전환율')
        ax.legend()
        plt.show()
        
def drawCvrRankGraph(df,col,IMG_PATH):
    makeDir(IMG_PATH)
   
    plt.figure(figsize=(18,5))
    plt.title(f'{col} 순위')
    colors = sns.color_palette('summer',len(df['origin']))
    bars = plt.bar(height=df[col].values,x=df['origin'].values,color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x()+ bar.get_width()/2, height, 
                f'{height:.2f}%', ha='center',va='bottom',
                fontsize=10, fontweight='bold')
    plt.savefig(f'{IMG_PATH}origin_{col}_순위')
    plt.show()
    

if __name__ == '__main__':
    IMG_PATH = './olist/img/'
    
    # 채널별 잠재고객 -> 유입고객 전환율
    query ='''
        select omqld.origin, mql_cnt.mql_origin_cnt ,count(omqld.origin) as closed_deals_origin_cnt,
		count(omqld.origin)/mql_cnt.mql_origin_cnt * 100 as `Success rate by origin(%)`
        from olist_marketing_qualified_leads_dataset omqld 
            left outer join olist_closed_deals_dataset ocdd  on omqld.mql_id = ocdd.mql_id
            left outer join olist_sellers_dataset osd on ocdd.seller_id = osd.seller_id
            left outer join (select origin, count(origin) as mql_origin_cnt from olist_marketing_qualified_leads_dataset omqld group by origin) as mql_cnt
            on omqld.origin = mql_cnt.origin
        where ocdd.mql_id is not null 
        group by omqld.origin, mql_cnt.mql_origin_cnt
        order by `Success rate by origin(%)` DESC ;
    '''
    
    columns=['origin','mql_cnt','closed_deals_cnt','CVR(%) by orgin']
    col = 'CVR(%) by orgin'
    #df = getDF(connection,columns,query)
    #drawCvrGraph(df,IMG_PATH)
    #drawCvrRankGraph(df,col,IMG_PATH)
    
    # 채널별 잠재고객 -> 유입고객 -> 활동고객 전환율
    query1 = '''
    select active_seller_dataset.origin,  cvr_dataset.mql_origin_cnt ,
	cvr_dataset.closed_deals_origin_cnt, cvr_dataset.`Success rate by origin(%)` ,
	active_seller_dataset.seller_cnt, active_seller_dataset.seller_cnt /cvr_dataset.closed_deals_origin_cnt *100 `Seller rate by origin(%)`,
	active_seller_dataset.seller_cnt /cvr_dataset.mql_origin_cnt  * 100 `Activation Rate (%)`
		from (select omqld.origin, mql_cnt.mql_origin_cnt ,count(omqld.origin) as closed_deals_origin_cnt,
				count(omqld.origin)/mql_cnt.mql_origin_cnt * 100 as `Success rate by origin(%)`
				from olist_marketing_qualified_leads_dataset omqld 
					left outer join olist_closed_deals_dataset ocdd  on omqld.mql_id = ocdd.mql_id
					left outer join olist_sellers_dataset osd on ocdd.seller_id = osd.seller_id
					left outer join (select origin, count(origin) as mql_origin_cnt 
									from olist_marketing_qualified_leads_dataset omqld group by origin) as mql_cnt
					on omqld.origin = mql_cnt.origin
					where ocdd.mql_id is not null 
					group by omqld.origin, mql_cnt.mql_origin_cnt
					order by `Success rate by origin(%)` DESC ) as cvr_dataset
		left join (select omqld.origin, count(*) seller_cnt 
					from (select ooid.seller_id , count(*) as item_cnt from olist_order_items_dataset ooid 
						left outer join olist_sellers_dataset osd 
						on ooid.seller_id = osd.seller_id
						group by ooid.seller_id) as seller_datset
					left join olist_closed_deals_dataset ocdd on seller_datset.seller_id = ocdd.seller_id
					right join olist_marketing_qualified_leads_dataset omqld on ocdd.mql_id = omqld.mql_id
					where ocdd.seller_id is not null
					group by omqld.origin) as active_seller_dataset
		on cvr_dataset.origin = active_seller_dataset.origin
		where active_seller_dataset.origin is not null
		order by `Seller rate by origin(%)` DESC;
    '''
    columns = ['origin','mql_cnt','closed_deals_cnt','signup CVR(%)','seller_cnt','active seller CVR(%)','active rate(%)']
    df = getDF(connection,columns,query1,'active rate(%)',9,10)
    drawCvrGraph(df,IMG_PATH)
    #drawCvrRankGraph(df,'active rate(%)',IMG_PATH)
    
    