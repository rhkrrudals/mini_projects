import re
import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from PIL import Image
from konlpy.tag import Okt
from wordcloud import WordCloud
from collections import Counter

def getFile(DIR_PATH):
    df_list = []
    filename_list = []
    file_list = os.listdir(DIR_PATH)
    for file in file_list:
        filename,ext = os.path.splitext(file)
        if ext == '.csv': 
            filename_list.append(filename)
            df = pd.read_csv(DIR_PATH+file)
            df_list.append(df)
    return df_list, filename_list

def replace(df, col):
    df[col] = df[col].str.replace(pat=r'<b>|</b>',repl=r'',regex=True)
    return df

def getNoun(df):
    okt = Okt()
    nouns_list = []
    for list in df.values:
        for i in range(2):
            nouns = okt.nouns(list[i])
            for noun in nouns:
                if len(noun)> 1: nouns_list.append(noun)
    return nouns_list

def removeDict(noun_list, rmword_list):
    return [word for word in noun_list if word not in rmword_list]

def getCommon(nouns_list):
    counts = Counter(nouns_list)
    tags = counts.most_common(100)
    tags_dict = dict(tags)
    return tags, tags_dict

def getImg(tag_dict,path):
    img_mask = np.array(Image.open('./company/square.png'))
    wc = WordCloud(font_path=path,width=400,height=400,
                   background_color='white', max_font_size=200,
                   repeat=True, colormap='Set2', mask=img_mask)
    cloud = wc.generate_from_frequencies(dict(tag_dict))
    return cloud

def drawCloud(cloud,title,DIR_PATH):
    plt.figure(figsize=(10,8))
    plt.axis('off')
    plt.imshow(cloud)
    plt.title(title,fontsize=15)
    plt.savefig(DIR_PATH + title)
    plt.show()

def main():
    rmword_dict = {
                   '머신러닝엔지니어 면접후기':['면접','후기','머신','러닝','엔지니어','준비','지원',
                                   '합격','학원','취업','국비','대해','내용','관련','오늘','통해',
                                   '질문','채용','티스','실제','대한','위해'],
                   
                   '개발자 면접후기':['합격','개발자', '면접','정말','어디','려고','아무',
                           '다도','후기','이상','계열','대비','여기','날씨','준비',
                           '지원','학원','취업','과정','국비','채용','오늘','대해',
                           '내용','질문','관련','통해','채용','진행','실제','대한','위해'],
                   
                   '인공지능 면접후기' : ['면접','후기','인공','지능','준비','합격','취업','학원','질문','국비','대해','내용',
                         '관련','오늘','통해','질문','채용','대한','실제','위해'],
                   
                   '데이터엔지니어 면접후기' : ['합격','면접','후기','머신러닝','엔지니어','준비','지원',
                             '학원','취업','국비','대해','내용','관련','오늘','통해','질문',
                             '채용','실제','대한','위해','티스']        
                   }
    
    if platform.system() == 'Windows':
        path = r'c:\Windows\Fonts\malgun.ttf'
    elif platform.system() == 'Darwin':
        path = r'/System/Library/Fonts/AppleGothic'
    else: 
        font = r'/usr/share/fonts/trutype/name/NanumMyeongjo.ttf'
        
    DIR_PATH = './company/' 
    IMG_PATH ='./company/image/'
    if os.path.exists(IMG_PATH): print('폴더 존재')
    else: 
        print('파일 존재하지 않아서 하나 생성합니다.')
        os.makedirs(IMG_PATH)  
    df_list, filename_list = getFile(DIR_PATH)
    for idx, df in enumerate(df_list):
        cols = df.columns
        for col in cols:    
            df = replace(df,col)
        noun_list = getNoun(df)
        noun_list= removeDict(noun_list,rmword_dict[filename_list[idx]])
        tags, tags_dict = getCommon(noun_list)
        cloud = getImg(tags_dict,path)
        drawCloud(cloud,title=filename_list[idx],DIR_PATH=IMG_PATH)
        print(f'{filename_list[idx]} 정제 및 시각화 완료')
        
if __name__ == '__main__':
    main()
