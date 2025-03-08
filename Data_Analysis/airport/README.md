# ✈️국내 항공사 매출 분석
## 개요
> 코로나 시작부터 종료까지 국내 항공사의 매출 분석 결과

## 목적
> 코로나(2019~2023)가 국내 항공사의 매출에 미친 요인 분석 

## 분석결과
> 항공사가 규모(대형 국적사, 저비용 항공사)에 따라 코로나로 인한 손실 규모가 달랐음

[자세한 분석 결과](https://palm-moon-278.notion.site/16598ff52a01803db7f2f3553c116da6?pvs=4)

### 데이터셋
|데이터 이름|내용|데이터 형식|
|---|---|---|
|항공사_2019|2019년 운항(편) 여객(명) 화물(톤)|xlsx|
|항공사_2020|2020년 운항(편) 여객(명) 화물(톤)|xlsx|
|항공사_2021|2021년 운항(편) 여객(명) 화물(톤)|xlsx|
|항공사_2022|2022년 운항(편) 여객(명) 화물(톤)|xlsx|
|항공사_2023|2023년 운항(편) 여객(명) 화물(톤)|xlsx|
|airplane_data|airplane_data	2014년 ~ 2023년 연도별 항공사 화물기 갯수 |csv|
### 데이터 출처
[항공사별 매출액 데이터 출처](https://www.airportal.go.kr/knowledge/indicator/airline/airline1.jsp)


[항공사 운항(편) 여객(명) 화물(톤) 데이터 출처](https://www.airport.co.kr/www/cms/frFlightStatsCon/airLineStats.do?MENU_ID=1250#none)


[연도별 항공사 항공기 보유대수](https://www.airportal.go.kr/knowledge/indicator/airline/airline2.jsp)

### 요구사항
* python == 3.8.20
* pandas == 2.0.3
* tabulate == 0.9.0
* scikit-learn == 1.3.2
* openpyxl == 3.1.5
* matplotlib == 3.7.5
* koreanize-matplotlib == 0.1.1
* selenium == 4.25.0
* beautifulsoup4 == 4.12.3
* seaborn == 0.13.2
