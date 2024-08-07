# Predicting Stock Market Close Prices with Streamlit and Binary ML Models

This project aims to develop a Streamlit web app to predict whether the closing price of a stock market will be higher the next day using binary machine learning models. The goal is to compare model performance using KPIs.

### Objectives:

- **Prediction Task:** Forecast if the next day's closing price will rise or fall.
- **Data Acquisition:** Historical stock market data by using [yfinance library](https://pypi.org/project/yfinance/) (open, close, high, low, volume).
- **Feature Engineering:** A variable (Target) that shows if there is difference (1) or not (0) between with the day before.
- **Model Selection:** Implementing and comparing models:
  - Logistic Regression
  - Support Vector Machines
  - Decision Tree
  - Gradient Boosting
  - Random Forest
  - XGBoost
- **Evaluation:** Compare models using accuracy, precision, ROC AUC Score, and Cohen Kappa Score.
- **Streamlit Integration:** An interactive interface for ticker, test size, and start date.

### Benefits:

- **User-Friendly:** Intuitive interface for easy interaction and visualization.
- **Comparison:** Allows to see the best-performing model.
- **Educational:** Demonstrates application of ML in stock market prediction.

### Conclusion:

This project combines ML, data analysis, and web development to create a tool for predicting stock market trends. By evaluating models with KPIs, it provides insights into effective prediction techniques based on historical data.