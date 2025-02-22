import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import koreanize_matplotlib 
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

from metrics import roc_curve_plot
from metrics import roc_curve_plot
from train import *

def makeDir(FILE_PATH,name):
    if os.path.exists(FILE_PATH+name): print('폴더 이미 존재')
    else:
        print(f'{name} 폴더 존재하지 않아 새로 생성하겠습니다.')
        os.makedirs(FILE_PATH+name)
        print(f'{name}파일 생성 완료')
        
def checkDescribe(df, ds_cols, name):
    print(f"[ {name}의 기초 통계량 ]")
    # Seriese --> Dataframe
    print(tabulate(df.loc[:,ds_cols].describe(), headers='keys', tablefmt='github'))
    print()
    
def getFile(FILE_PATH,ds_cols):
    file_list = os.listdir(FILE_PATH)
    for file in file_list:
        filename, ext = os.path.splitext(file)
        if ext == '.csv':
            df = pd.read_csv(FILE_PATH+file)
            df = df[(df['thal'] != 0) & (df['ca'] !=4)]
            target_df = df[df['target']==1]
            non_target_df = df[df['target'] == 0]
            
            print(f'[ {file} 정보 ]')
            print(df.info())
            print()
            print(f'[ {file} 테이블 확인 ]')
            print(tabulate(df.head(),headers='keys',tablefmt='github'))
            print()
            while True:
                df_choice = input('기초 통계량을 확인하고 싶은 데이터프레임을 선택하시오.(a or A: 전체, n or N: 비질환군, y or Y: 심장병 질환군 )')
                if df_choice == 'a' or df_choice =='A':
                    checkDescribe(df,ds_cols,'전체 데이터')
                    break
                elif df_choice == 'n' or df_choice =='N': 
                    checkDescribe(non_target_df,ds_cols,'비질환군')
                    break
                elif df_choice == 'y' or df_choice =='Y':
                    checkDescribe(target_df,ds_cols,'질환군')
                    break
                else: 
                    print('다시 입력하세요.') 
                    continue
                
            return df, non_target_df, target_df

def drawGraph(cols, dfs, color, label, name, FILE_PATH):
    makeDir(FILE_PATH,name)
    for col in cols:
        if len(dfs) > 1:
            plt.title(f'{col}분포')
            for i, df in enumerate(dfs): 
                val_idx = df[col].value_counts().index
                plt.hist(df[col].values, color=color[i], alpha=0.6, edgecolor='grey', linewidth=1.5, histtype='stepfilled', label=label[i])
            if len(val_idx) <=10: plt.xticks(val_idx)
            plt.legend()
            plt.savefig(f'{FILE_PATH}{name}/비질환군 질환군 {col} 비교분포')
            plt.show()
        else:
            df = dfs[0]
            val_idx = df[col].value_counts().index
            plt.title(f'전체 {col}분포')
            plt.hist(df[col].values,edgecolor='grey',linewidth=1.5 ,color=color)
            if len(val_idx) <=10: plt.xticks(val_idx)
            plt.savefig(f'{FILE_PATH}{name}/전체 {col}분포')
            plt.show()

def compareGraph(df,cols, title, range, FILE_PATH):
    plt.title(title)
    plt.hist(df[cols[0]][df[cols[1]].values < range], alpha=0.4,  histtype='stepfilled',color='darkseagreen', edgecolor='grey', label=f'{range}미만')
    plt.hist(df[cols[0]][df[cols[1]].values >= range], alpha=0.4,  histtype='stepfilled',color='lightpink', edgecolor='grey',label=f'{range}이상')
    plt.legend()
    plt.savefig(f'{FILE_PATH}img/{title}')
    plt.show()
    
def drawScatter(df, x_col, y_col,  FILE_PATH, x_lines=None, y_lines=None, colors=None, rect=False):
    
    if df is non_target_df: df_name = '비질환군 데이터'
    elif df is target_df: df_name = '심장병 질환군 데이터'
    else: df_name = '전체 데이터'
    
    x = df[x_col]
    y = df[y_col]
    colors = ['mediumvioletred' if val>=70 else 'palevioletred' 
             if val>=60 else 'plum' 
             if val>=50 else 'pink'
             if val>=40 else 'khaki' 
             for val in x]
    
    fig, ax = plt.subplots() 
    plt.title(f'{df_name}의 {x.name}와 {y.name}의 상관관계')
    ax.scatter(x.values, y.values,color=colors)
    if rect:
        for i in range(len(x_lines)-1):
            rect=patches.Rectangle(
                (x_lines[i],y_lines[i]),
                x_lines[i+1] - x_lines[i],
                y.max() - y_lines[i],
                linewidth=2, facecolor='yellow', alpha=0.2
            )
            ax.add_patch(rect)
    plt.xlabel(f'{x.name}'); plt.ylabel(f'{y.name}')
    plt.grid(axis='x')
    plt.savefig(f'{FILE_PATH}img/{df_name}의 {x.name}과 {y.name}의 상관관계')
    plt.show()
            
def main(df, classifiers):
    featureDF = df.drop(columns='target')
    targetSR = df['target']
    
    # train, test
    X_train, X_test, y_train, y_test = train_test_split(featureDF,targetSR,test_size=0.2,random_state=156)
    print(f'X_train : {X_train.shape}, y_train : {y_train.shape}')
    print(f'x_test : {X_test.shape}, y_test : {y_test.shape}\n')  
    
    # train, val
    X_tr, X_val, y_tr, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=132)
    print(f'X_tr: {X_tr.shape}, y_tr: {y_tr.shape}')
    print(f'X_val: {X_val.shape}, y_val: {y_val.shape}')
    
    classifierModelTrain(classifiers,featureDF,targetSR,X_train,y_train,X_test,y_test,FILE_PATH)
    XGBoostModelTrain(xgb_wrapper,X_tr,y_tr,X_val,y_val,X_test,y_test,FILE_PATH)     

if __name__ == '__main__':
    FILE_PATH = './heart_disease/'
    ds_cols = ['age','trestbps','chol','thalach','oldpeak']
    df, non_target_df, target_df = getFile(FILE_PATH,ds_cols)
    
    cols = df.columns
    dfs = [non_target_df, target_df]
    labels = ['비질환군','심장병 질환군']
    colors = ['skyblue','red']
    #drawGraph(cols, dfs, colors ,labels, 'img', FILE_PATH)
    #drawGraph(cols, [df], 'pink' ,labels, 'img', FILE_PATH)
    
    compare_cols = ['age','trestbps']
    title = '비질환군의 혈압 수치에 따른 나이 분포\n(140mmHg기준)'
    #compareGraph(non_target_df,compare_cols,title,140,FILE_PATH)
    
    x_col = 'age' ; y_col='trestbps'
    x_lines = [30,40,50,60,80]
    y_lines = [127.5,136,144.5,153,161.5]
    # drawScatter(target_df,x_col,y_col,FILE_PATH,x_lines,y_lines,rect=True)
    
    # models 
    lr_clf = LogisticRegression(solver='liblinear')
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    dt_clf = DecisionTreeClassifier(random_state=11, min_samples_leaf=2)
    rf_clf = RandomForestClassifier(random_state=0, max_depth=6)
    gb_clf = GradientBoostingClassifier(random_state=0)
    vo_clf = VotingClassifier(estimators=[('KNN',knn_clf),('DT',dt_clf)],voting ='soft')
    xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.05, eval_metric=['logloss','auc'], early_stopping_rounds=50, max_depth=3)
    
    classifiers = [lr_clf,knn_clf,dt_clf,rf_clf, gb_clf, vo_clf]
    main(df, classifiers)
    
    