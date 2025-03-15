# Olist 판매자 여정 분석
## 개요
> Brazilian E-Commerce Olist의 판매자 여정 분석 결과

## 목적
> Brazilian E-Commerce Olist의 판매자 여정 분석을 통한 효과적인 고객 유치 방법 도출

## 분석결과
> 잠재고객의 회원고객 전환율이 낮아, 랜딩페이지 및 회원가입 페이지 UX/UI 구조 개선이 필요해 보였음

[자세한 분석 결과](https://palm-moon-278.notion.site/Olist-1b498ff52a018030aa7fff75ef837f3c?pvs=4)

### 데이터셋
|데이터 이름|내용|데이터 형식|
|---|---|---|
|olist_customers_dataset|고객 정보 (판매자 입장에서)|csv|
|olist_geolocation_dataset|지리 정보|csv|
|olist_order_payments_dataset|결제 수단 정보|csv|
|olist_order_reviews_dataset|제품 리뷰 |csv|
|olist_orders_dataset|주문 배송 정보|csv|
|olist_products_dataset|판매하는 제품 정보 |csv|
|product_category_name_translation|제품 영어 이름 (브라질어 → 영어) |csv|
|olist_order_items_dataset|주문 제품 정보 |csv|
|olist_sellers_dataset|판매자 정보 |csv|
|olist_marketing_qualified_leads_dataset|마케팅 통해서 Olist 랜딩 페이지 방문한 잠재 고객 정보 |csv|
|olist_closed_deals_dataset|잠재 고객에서 Olist 회원으로 전환한 고객 정보 |csv|
### 데이터 출처
[Olist 데이터 출처](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### 요구사항
* python == 3.8.20
* pandas == 2.0.3
* tabulate == 0.9.0
* matplotlib == 3.7.5
* koreanize-matplotlib == 0.1.1
* seaborn == 0.13.2
