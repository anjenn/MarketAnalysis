Input Features (X):
ITEM_COUNT: Number of items in each offer.
REVIEW_RATIO: Ratio of review counts relative to the platform's maximum.
UNIT_PRICE: Price per standard quantity (e.g., 100ml).
QUANTITY: Volume of the product in each offer (e.g., 500ml, 1000ml).

Output (Y):
Optimal Unit Price: Predicted price to maximize sales/profit for a given quantity.


Experimented with Keras, lightgbm, xgboost, catboost
Other metrics to consider further: brand recognition
