USE User;
/*
 * olist_order_dataset PK & UK Setting
 */

ALTER TABLE `User`.olist_orders_dataset  ADD PRIMARY KEY (order_id,customer_id);
ALTER TABLE `User`.olist_orders_dataset  ADD CONSTRAINT UNIQUE(order_id);
ALTER TABLE `User`.olist_orders_dataset  ADD CONSTRAINT UNIQUE(customer_id);

/*
 * olist_marketing_qualified_lead_dataset PK & UK Setting
 */

ALTER TABLE `User`.olist_marketing_qualified_leads_dataset ADD PRIMARY KEY (mql_id);
ALTER TABLE `User`.olist_marketing_qualified_leads_dataset ADD CONSTRAINT UNIQUE(mql_id);

/*
 * FK Setting
 */

ALTER TABLE `User`.olist_closed_deals_dataset 
ADD CONSTRAINT fk_mql_id
FOREIGN KEY (mql_id) REFERENCES  `User`.olist_marketing_qualified_leads_dataset(mql_id);

ALTER TABLE `User`.olist_order_payments_dataset 
ADD CONSTRAINT fk_order_id
FOREIGN KEY (order_id) REFERENCES `User`.olist_orders_dataset(order_id);

ALTER TABLE `User`.olist_order_items_dataset 
add constraint fk_item_order_id
FOREIGN key (order_id) REFERENCES `User`.olist_orders_dataset(order_id);

/*
ALTER TABLE `User`.olist_customers_dataset 
ADD CONSTRAINT fk_customers_id
FOREIGN KEY (customer_id) REFERENCES `User`.olist_orders_dataset(customer_id);
*/

ALTER TABLE `User`.olist_order_reviews_dataset 
ADD CONSTRAINT fk_review_order_id
FOREIGN KEY (order_id) REFERENCES `User`.olist_orders_dataset(order_id);

/*
 * order_dataset cutomer_id		: PK -> FK
 * customer_dataset customer_id : FK -> PK
 */
ALTER TABLE `User`.olist_orders_dataset DROP PRIMARY KEY, ADD PRIMARY KEY (order_id);
ALTER TABLE `User`.olist_customers_dataset DROP FOREIGN KEY fk_customers_id;
ALTER TABLE `User`.olist_orders_dataset  DROP INDEX customer_id;
ALTER TABLE `User`.olist_customers_dataset ADD PRIMARY KEY (customer_id);
ALTER TABLE `User`.olist_customers_dataset ADD CONSTRAINT UNIQUE (customer_id);
ALTER TABLE `User`.olist_orders_dataset 
ADD CONSTRAINT fk_customer_id
FOREIGN KEY (customer_id) REFERENCES `User`.olist_customers_dataset(customer_id);



