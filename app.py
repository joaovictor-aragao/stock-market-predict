import streamlit as st
from util.market_prediction import MarketPrediction

### Page config
st.set_page_config(
    page_title='Market Prediction',
    page_icon=':bar_chart:',
    layout='wide',
)
st.title('Market Prediction')
st.markdown("_v0.0.1_")

### Sidebar
with st.sidebar:
    ticker = st.text_input("Ticket", value='^GSPC')
    test_size = st.number_input("Test size", value=.2)
    start_date = st.text_input("Start date", value='2020-01-01')

    if st.button("Download and apply models"):
        if ticker and test_size and start_date:
            with st.spinner(text='Processing...'):
                st.session_state['opts'] = (
                    ticker, test_size, start_date
                )
                mp = MarketPrediction(*st.session_state['opts'])
                mp.download()
        else:
            st.error("Please, fill all fields.")


mp = MarketPrediction(*st.session_state['opts'])
data = mp.treat_data()

container0 = st.container()
container1 = st.container()

with container0:
    st.dataframe(mp.kpis_models(data))

with container1:
    st.markdown(
        '''
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
            - Random Forest and
            - XGBoost
        - **Evaluation:** Compare models using accuracy, precision, ROC AUC Score and Cohen Kappa Score.
        - **Streamlit Integration:** An interactive interface for ticker, test size and start date.

        ### Benefits:

        - **User-Friendly:** Intuitive interface for easy interaction and visualization.
        - **Comparison:** Allows to see the best-performing model.
        - **Educational:** Demonstrates application of ML in stock market prediction.

        ### Conclusion:

        This project combines ML, data analysis, and web development to create a tool for predicting stock market trends. By evaluating models with KPIs, it provides insights into effective prediction techniques based on historical data.
        '''
    )

print('---')