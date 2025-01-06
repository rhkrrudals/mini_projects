# ------------------------------------------------------------------------------------------------
# SQL 미니 프로젝트 
# 5조 : 유류비 상승 --> 물가상승에 미치는 영향
# -------------------------------------------------------------------------------------------------
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 데이터프레임 뽑기
conn = pymysql.connect(host='172.20.54.245',user='member1',
                       password='1234', db = 'fuel',charset='utf8')

cur = conn.cursor(pymysql.cursors.DictCursor)
cur.execute('select * from ppi_file')
rows = cur.fetchall()

ppi_df = pd.DataFrame(rows)
print(ppi_df)

# 그래프 그리기 - ppi 그래프

print(ppi_df['년도'])
year = ppi_df['년도']
print(ppi_df.iloc[:,2:])
val = ppi_df.iloc[:,2:]
print(ppi_df.columns[2:])
label = ppi_df.columns[2:]

plt.title('생산자 물가 지수')
plt.plot(year, val, 'o-', label = label)
plt.xlabel('년도')
plt.xticks([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
plt.ylabel('지수')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

cur.close()
conn.close()

# 데이터프레임 뽑기
conn = pymysql.connect(host='172.20.54.245',user='member1',
                       password='1234', db = 'fuel',charset='utf8')

cur = conn.cursor(pymysql.cursors.DictCursor)
cur.execute('select * from fuel_file')
rows = cur.fetchall()

fuel_df = pd.DataFrame(rows)
print(ppi_df)

# 그래프 그리기 - 유류비 그래프

print(fuel_df['년도'])
year = fuel_df['년도']
print(fuel_df.iloc[:,1:])
val = fuel_df.iloc[:,1:]
print(fuel_df.columns[1:])
label = fuel_df.columns[1:]

plt.title('연도별 유류비')
plt.plot(year, val, 'o-', label = label)
plt.xlabel('년도')
plt.ylabel('지수')
plt.xticks([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1))
plt.grid(True)
plt.show()

cur.close()
conn.close()



