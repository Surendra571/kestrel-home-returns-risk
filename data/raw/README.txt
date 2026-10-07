KESTREL HOME â€” RETURNS DATA PACK (Variant A)
============================================

train.csv             Past orders, with the outcome.
test_unlabelled.csv   The most recent orders, same columns without `returned`.
                      This file is the snapshot the warehouse sees at dispatch.
  order_id                 Order number
  order_placed_at          IST
  customer_id              Key into customers.csv
  sku                      Key into products.csv
  sales_channel            app | web | marketplace | partner_outlet
  payment_mode             prepaid_upi | prepaid_card | cod | emi
  discount_pct             Discount applied at checkout
  qty                      Units
  order_value_inr          Order value as stored by the payment system
  promised_delivery_days   Days promised at checkout
  delivery_pincode         Delivery pincode. 000000 = system default when no address was captured (walk-in partner orders)
  is_gift                  Y/N
  customer_prior_orders    Customer's orders before this one
  customer_prior_returns   Customer's returns before this one
  delivery_note            Free text typed by the customer or the outlet
  last_service_event_type  Latest event in the service system for this order (INSTALL_BOOKED, INSTALL_DONE, DEMO_DONE, TECH_VISIT, REVERSE_PICKUP, NONE)
  pickup_scheduled_at      Reverse-pickup booking time from the logistics system, if any
  source                   crm | partner_feed (re-imported from the partner-outlet feed)
  returned                 1 = order was returned (train only)

customers.csv         customer_id, city, state, signup_date, shield_member (Y/N)
products.csv          sku, family, model_name, list_price_inr, warranty_months, launch_date
sample_submission.csv order_id, score
ops-policy.pdf        Kestrel operations policy v4.1 â€” costs, Shield, returns process, systems.
email-thread.txt      Messages already exchanged about this work.

No other documentation is available.