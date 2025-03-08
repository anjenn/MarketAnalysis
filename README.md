## Predicting Optimal Pricing for Maximum Sales & Profit
This project explores machine learning models to predict the optimal unit price of products, aiming to maximize sales and profitability. By analyzing key pricing and product features, I experimented with different models to refine pricing strategies.

# Key Input Features (X):
ITEM_COUNT: Number of items in each offer.
REVIEW_RATIO: Ratio of review counts relative to the platform's maximum.
UNIT_PRICE: Price per standard quantity (e.g., per 100ml).
QUANTITY: Total volume of the product in each offer (e.g., 500ml, 1000ml).
# Target Output (Y):
Optimal Unit Price: The predicted price that balances sales and profitability for a given product quantity.
# Models Tested:
Deep Learning (Keras): Explored neural network-based pricing predictions.
Boosting Algorithms: Compared performance of LightGBM, XGBoost, and CatBoost for structured data.
# Further Considerations:
Brand Recognition: Investigating how brand perception impacts pricing and consumer demand.
Additional Metrics: Exploring other factors that could refine pricing recommendations.
