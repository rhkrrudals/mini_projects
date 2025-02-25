# ☕️ 기후 변화와 커피 원두 생산 관계 분석
## 개요
> 기후변화가 주요 커피 원두 생산국(브라질, 베트남, 콜롬비아)의 생산량에 미친 영향을 분석한 결과

## 목적
> 지구 온난화가 커피 원두 재배지 상위 3개 국가(브라질, 베트남, 콜롬비아)에 미친 영향을 분석하고자 함

## 분석결과
> 베트남을 제외한 나머지 국가(브라질, 콜롬비아)에서는 **강수량 변화에 따라서 생산량이 좌우**되는 것을 확인할 수 있었음

[자세한 분석 결과](https://palm-moon-278.notion.site/17198ff52a018077a17ff452b3a4905c?pvs=4)

### 데이터셋
|데이터 이름|내용|데이터 형식|
|---|---|---|
|베트남 기온|2017~2021 베트남 월평균 일 최고기온, 일 죄저기온, 최대 일 강수량, 월중 최대풍속|xlsx|
|브라질 기온|2017~2021 브라질 월평균 일 최고기온, 일 죄저기온, 최대 일 강수량, 월중 최대풍속|xlsx|
|콜롬비아 기온|2017~2021 콜롬비아 월평균 일 최고기온, 일 죄저기온, 최대 일 강수량, 월중 최대풍속|xlsx|
|coffee|2012~2021 연도별 커피 원두 생산량|xlsx|
### 데이터 출처
[생커피원두 생산 데이터 출처](https://kosis.kr/statHtml/statHtml.do?sso=ok&returnurl=https%3A%2F%2Fkosis.kr%3A443%2FstatHtml%2FstatHtml.do%3Flist_id%3DR_SUB_UTITLE_G%26obj_var_id%3D%26seqNo%3D%26tblId%3DDT_2KAA408%26vw_cd%3DMT_RTITLE%26orgId%3D101%26path%3D%252FstatisticsList%252FstatisticsListIndex.do%26conn_path%3DMT_RTITLE%26itm_id%3D%26lang_mode%3Dko%26scrId%3D%26)


[3개국 기후 데이터 출처](https://data.kma.go.kr/data/ogd/selectGtsRltmList.do?pgmNo=658)

### 요구사항
* python == 3.8.20
* pandas == 2.0.3
* tabulate == 0.9.0
* scikit-learn == 1.3.2
* openpyxl == 3.1.5
* matplotlib == 3.7.5
* koreanize-matplotlib == 0.1.1
