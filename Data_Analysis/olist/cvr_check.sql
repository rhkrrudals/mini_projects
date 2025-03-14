USE User;

select customer_id, customer_unique_id from olist_customers_dataset;
select count(*) from olist_order_items_dataset ooid;
select count(*) from olist_sellers_dataset osd ;

/*
 * 유입 고객 중 활동하는 고객
 */

select omqld.origin, ocdd.seller_id,count(*) seller_cnt from (select ooid.seller_id , count(*) as item_cnt from olist_order_items_dataset ooid 
	left outer join olist_sellers_dataset osd 
	on ooid.seller_id = osd.seller_id
group by ooid.seller_id) as seller_datset
left join olist_closed_deals_dataset ocdd on seller_datset.seller_id = ocdd.seller_id
right join olist_marketing_qualified_leads_dataset omqld on ocdd.mql_id = omqld.mql_id
where ocdd.seller_id is not null
group by ocdd.seller_id, omqld.origin;

/*
 * 잠재고객 -> 유입고객 -> 활동고객 전환율 
 */

select active_seller_dataset.origin,  cvr_dataset.mql_origin_cnt ,
	cvr_dataset.closed_deals_origin_cnt, cvr_dataset.`Success rate by origin(%)` ,
	active_seller_dataset.seller_cnt, active_seller_dataset.seller_cnt /cvr_dataset.closed_deals_origin_cnt *100 `Seller rate by origin(%)`,
	active_seller_dataset.seller_cnt /cvr_dataset.mql_origin_cnt  * 100 `Activation Rate (%)`
		from (select omqld.origin, mql_cnt.mql_origin_cnt ,count(omqld.origin) as closed_deals_origin_cnt,
				count(omqld.origin)/mql_cnt.mql_origin_cnt * 100 as `Success rate by origin(%)`
				from olist_marketing_qualified_leads_dataset omqld 
					left outer join olist_closed_deals_dataset ocdd  on omqld.mql_id = ocdd.mql_id
					left outer join olist_sellers_dataset osd on ocdd.seller_id = osd.seller_id
					left outer join (select origin, count(origin) as mql_origin_cnt 
									from olist_marketing_qualified_leads_dataset omqld group by origin) as mql_cnt
					on omqld.origin = mql_cnt.origin
					where ocdd.mql_id is not null 
					group by omqld.origin, mql_cnt.mql_origin_cnt
					order by `Success rate by origin(%)` DESC ) as cvr_dataset
		left join (select omqld.origin, count(*) seller_cnt 
					from (select ooid.seller_id , count(*) as item_cnt from olist_order_items_dataset ooid 
						left outer join olist_sellers_dataset osd 
						on ooid.seller_id = osd.seller_id
						group by ooid.seller_id) as seller_datset
					left join olist_closed_deals_dataset ocdd on seller_datset.seller_id = ocdd.seller_id
					right join olist_marketing_qualified_leads_dataset omqld on ocdd.mql_id = omqld.mql_id
					where ocdd.seller_id is not null
					group by omqld.origin) as active_seller_dataset
		on cvr_dataset.origin = active_seller_dataset.origin
		where active_seller_dataset.origin is not null
		order by `Seller rate by origin(%)` DESC;

/*
 * 마케팅 채널별 실제로 제품을 판매한 판매자들 
 */
select omqld.origin, count(*) seller_cnt 
from (select ooid.seller_id , count(*) as item_cnt 
		from olist_order_items_dataset ooid 
		left outer join olist_sellers_dataset osd 
		on ooid.seller_id = osd.seller_id
		group by ooid.seller_id) as seller_datset
left join olist_closed_deals_dataset ocdd on seller_datset.seller_id = ocdd.seller_id
right join olist_marketing_qualified_leads_dataset omqld on ocdd.mql_id = omqld.mql_id
where ocdd.seller_id is not null
group by omqld.origin;


/*
  Business_Type별 유입고객 -> 활동고객 비율  
 */

select closed_deals_dataset.origin, closed_deals_dataset.business_type, closed_deals_dataset.closed_deals_cnt,
	   active_seller_dataset.seller_cnt, active_seller_dataset.seller_cnt/closed_deals_dataset.closed_deals_cnt * 100 `CVR(%)`
from (select omqld.origin , ocdd.business_type , count(*) closed_deals_cnt
			from olist_closed_deals_dataset ocdd
			right join olist_marketing_qualified_leads_dataset omqld on ocdd.mql_id = omqld.mql_id
			where ocdd.mql_id is not null
			group by omqld.origin ,ocdd.business_type 
			order by omqld.origin DESC, closed_deals_cnt DESC) as closed_deals_dataset
left join (select omqld.origin,ocdd.business_type, count(*) seller_cnt 
			from (select ooid.seller_id , count(*) as item_cnt 
					from olist_order_items_dataset ooid 
					left outer join olist_sellers_dataset osd 
					on ooid.seller_id = osd.seller_id
					group by ooid.seller_id) as seller_datset
			left join olist_closed_deals_dataset ocdd on seller_datset.seller_id = ocdd.seller_id
			right join olist_marketing_qualified_leads_dataset omqld on ocdd.mql_id = omqld.mql_id
			where ocdd.seller_id is not null
			group by omqld.origin, ocdd.business_type
			order by omqld.origin DESC, seller_cnt DESC) as active_seller_dataset
on closed_deals_dataset.business_type  = active_seller_dataset.business_type 
and closed_deals_dataset.origin = active_seller_dataset.origin;
