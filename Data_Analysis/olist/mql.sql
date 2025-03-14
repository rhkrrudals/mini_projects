/*
 * Olist 랜딩 페이지 전환율 Funnel 분석
 */
Use User;

SELECT * from `User`.olist_marketing_qualified_leads_dataset omqld;

/*
 * Check tables 
 */
SELECT count(mql_id) from olist_marketing_qualified_leads_dataset omqld ;
select count(mql_id) from olist_closed_deals_dataset ocdd ;
select count(seller_id) from olist_closed_deals_dataset ocdd ;
select count(seller_id) from olist_sellers_dataset osd ;

/*
 * Olist Landing Page MQl's conversion rate is 10.52%, It's too low a rate.. --> WHY? 
 */
select count(omqld.mql_id) as mql_cnt, count(ocdd.mql_id) as closed_deal_cnt, count(ocdd.mql_id)/count(omqld.mql_id) * 100 as `CVR(%)`
from olist_marketing_qualified_leads_dataset omqld 
	left outer join olist_closed_deals_dataset ocdd  on omqld.mql_id = ocdd.mql_id
	left outer join olist_sellers_dataset osd on ocdd.seller_id = osd.seller_id;

/*
 * Check the omqld origin 
 * (origin is marketing channel)
 */
select omqld.origin, mql_cnt.mql_origin_cnt ,count(omqld.origin) as closed_deals_origin_cnt,
		count(omqld.origin)/mql_cnt.mql_origin_cnt * 100 as `Success rate by origin(%)`
from olist_marketing_qualified_leads_dataset omqld 
	left outer join olist_closed_deals_dataset ocdd  on omqld.mql_id = ocdd.mql_id
	left outer join olist_sellers_dataset osd on ocdd.seller_id = osd.seller_id
	left outer join (select origin, count(origin) as mql_origin_cnt from olist_marketing_qualified_leads_dataset omqld group by origin) as mql_cnt
	on omqld.origin = mql_cnt.origin
where ocdd.mql_id is not null 
group by omqld.origin, mql_cnt.mql_origin_cnt
order by `Success rate by origin(%)` DESC ;

/*
 * 
 */
select * from olist_orders_dataset ood;

