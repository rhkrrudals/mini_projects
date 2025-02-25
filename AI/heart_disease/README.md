# 🫀 심장병 환자 분석 및 예측
## 개요
> 1988년 미국, 미국의 롱비치 V, 헝가리, 스위스에서 수집된 병원 데이터로 심장병 환자 분석 결과 및 분류 모델을 통한 심장병 환자 특징 예측

## 목적
> 심장병 환자를 분석하고 심장병 환자 특징을 예측하는 모델 구축하고자 함

## 분석결과
[자세한 분석 결과](https://palm-moon-278.notion.site/1a598ff52a01805d8b7ccc38dd3ab004?pvs=4)

## 모델 학습 결과
> 재현율(recall)에서 0.95로 GradientBoostingClassifier보다 0.01 낮은 성능을 보였지만 그 외 평가지표에서 우수한 성능을 보여준
> RandomForestClassifier를 최종 예측 모델로 선정


|Model|Recall|Accuracy|F1score(macro avg)|AUC|
|---|---|---|---|---|
|LogisticRegression|0.87|0.86|0.84|0.9262|
|KNeighborsClassifier|0.82|0.85|0.85|0.9529|
|VotingClassifier|0.95|0.96|0.96|0.9948|
|RandomForestClassifier|0.95|**0.97**|**0.97**|**0.9968**|
|GradientBoostingClassifier|**0.96**|0.96|0.96|0.9946|
|XGBClassifier|0.92|0.92|0.92|0.9844|

### 데이터셋
|데이터 이름|내용|데이터 형식|
|---|---|---|
|heart|환자 기본 정보 및 심장질환 관련 컬럼들|csv|
### 데이터 출처
[심장병 데이터 출처](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data)

### 요구사항
* python == 3.8.20
* pandas == 2.0.3
* numpy == 1.24.3
* scikit-learn == 1.3.2
* tabulate == 0.9.0
* matplotlib == 3.7.5
* seaborn == 0.13.2
* koreanize-matplotlib == 0.1.1
* joblib == 1.4.2
* xgboost == 2.1.1
* py-xgboost == 2.1.1
