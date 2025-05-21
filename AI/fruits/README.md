# 🫀 과일 예측하기
> Inception_V3 모델을 활용해 과일 이미지 예측 알고리즘 구축

## 목적
> Inception_V3 모델을 활용해 과일 이미지 예측하고자 함

## 분석결과
[자세한 분석 결과](https://palm-moon-278.notion.site/1fa98ff52a0180f48818f7de46090c16?pvs=4)

## 모델 학습 결과
> 모든 데이터셋(훈련, 검증, 테스트)에서 90% 이상의 높은 정확도와 약 10% 이하의 낮은 손실값을 보여, 모델의 안정적인 성능을 확인할 수 있었음


||F1score|Loss|
|---|---|---|
|Train|0.951|0.14|
|Val|0.951|0.069|
|Test|0.951|0.062|

### 데이터셋
|데이터 이름|내용|데이터 형식|
|---|---|---|
|apple|사과 이미지 1000개|jpg|
|banana|바나나 이미지 1000개|jpg|
|orange|오렌지 이미지 1000개|jpg|
|strawberry|딸기 이미지 1000개|jpg|
### 데이터 출처
[과일 이미지 데이터 출처](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)

### 요구사항
* python == 3.8.20
* pandas == 2.0.3
* numpy == 1.24.3
* scikit-learn == 1.3.2
* matplotlib == 3.7.5
* koreanize-matplotlib == 0.1.1
