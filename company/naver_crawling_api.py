import urllib.request
import re
import datetime
import json
import pandas as pd
import csv
import requests
from bs4 import BeautifulSoup
import collections

if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

def get_request_url(url):
    client_id = 'your_api_id'
    client_secret = 'yout_api_password'

    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", client_id)
    req.add_header("X-Naver-Client-Secret", client_secret)

    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            # print(f"[{datetime.datetime.now()}] Url Reqeust Success")
            return response.read().decode('utf-8')
        
    except Exception as e:
        print(e)
        print(f"Error for URL: {url}")


def get_naver_search(node, search_text, start, display):
    base = "https://openapi.naver.com/v1/search"
    node = f"/{node}.json"
    query_string = f"{urllib.parse.quote(search_text)}"

    # f"?query={query_string}&start={start}&display={display}"
    parameters = ("?query={}&start={}&display={}".
                  format(query_string, start, display))

    url = base + node + parameters
    response = get_request_url(url)

    if response is None:
        return None
    else:
        return json.loads(response)  # json 문자열을 Python 객체로 변환


def get_post_data(post, json_result_list, cnt):
    title = post['title']
    description = post['description']
    
    '''
     strptime()
     - %a: abbreviated weekday name     
     - %b" abbreviated month name
    '''
    
    print(f"[{cnt}]", end=" ")
    print(title, end=": ")
    print(description)

    # ['제목', '내용']
    json_result_list.append([title, description])

def main():
    node = 'blog'  # 크롤링 대상
    search_text = '인공지능 면접후기'
    cnt = 0
    json_result_list = []
    
    while cnt < 900:
        json_response = get_naver_search(node, search_text, 1, 100)
        while (json_response is not None) and (json_response['display'] != 0):
            
            for post in json_response['items']:
                cnt += 1
                get_post_data(post, json_result_list, cnt)
            start = json_response['start'] + json_response['display']
            json_response = get_naver_search(node, search_text, start, 100)
            
            if cnt == 900: break

    # csv 파일로 저장
    # ['제목', '개요']
    DIR_PATH = './company/'
    columns = ['title', 'description']
    result_df = pd.DataFrame(json_result_list, columns=columns)
    result_df.to_csv(DIR_PATH + f'{search_text}.csv', index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()

