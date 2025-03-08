from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import pandas as pd
from tabulate import tabulate

DIR_PATH = './airport/'
service = Service(executable_path= DIR_PATH+'chromedriver')
driver =  webdriver.Chrome(service=service)
url = 'https://www.airportal.go.kr/knowledge/indicator/airline/airline2.jsp'
driver.get(url)

time.sleep(3)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

table = soup.find('table')
#print(table.prettify())
dataframes = []

if table:
    raw_headers = [th.get_text(strip=True) for th in table.find_all('th')]
    headers = list(dict.fromkeys(filter(lambda x:x and x not in ['구분'],raw_headers)))
    #print(headers)
    
    rows = []
    for tr in table.find_all('div', class_ = 'GMBodyMid'):
        cells = [td.get_text(strip=True) for td in tr.find_all('td') if td.get_text(strip=True) != '항공기 보유대수(대)']
        for i in range(0,len(cells),12):
            row = cells[i:i+12]
            rows.append(row[1:])
            
    df = pd.DataFrame(rows, columns=headers)
    df = df.replace('','0')
    df.iloc[:,1:] = df.iloc[:,1:].astype(int)
    print(tabulate(df.head(), headers='keys', tablefmt='github'))
    df.to_csv(DIR_PATH+'airplane_data.csv', index=False, encoding='utf-8')
    print('데이터가 저장되었습니다')
    
else: print('테이블을 찾을 수 없습니다.')

driver.quit()


