import os
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd

##### Variables
PREDICTORS = ["Close", "Volume", "Open", "High", "Low"]
MODELS = ['Logistic Regression', 'Support Vector Machines', 'Decision Tree', 'Gradient Boosting', 'Random Forest', 'XGB']

class MarketPrediction:

    def __init__(self, ticker, test_size, start_date):
        self.ticker = ticker 
        self.test_size = test_size
        self.start_date = start_date

    ## generate datasets
    def download(self):

        if os.path.exists("./data/data.csv"):
            data = pd.read_csv("./data/data.csv", index_col=0)
        else:
            data = yf.Ticker(self.ticker)
            data = data.history(start=self.start_date)
            data.to_csv("./data/data.csv")

    def treat_data(self):

        if os.path.exists("./data/data.csv"):
            data = pd.read_csv("./data/data.csv", index_col=0)

        ##### Cleaning dataset and create the target variable
        data.index = pd.to_datetime(data.index, utc=True)

        del data["Dividends"]
        del data["Stock Splits"]

        data["Tomorrow"] = data["Close"].shift(-1)
        data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

        ##### Split data to train and test

        X_train, X_val, Y_train, Y_val = train_test_split(
            data[PREDICTORS], 
            data["Target"], 
            ## proportion from dataset to be used as test
            test_size=self.test_size, 
            ## control shuffling applied to the data before applying the split
            random_state=42
            )

        return X_train, X_val, Y_train, Y_val
    
    ## generate kpis
    def kpis_models(self, datasets):

        ## generate train and test datasets
        X_train, X_val, Y_train, Y_val = datasets[0], datasets[1], datasets[2], datasets[3]
        models = {}

        models['Logistic Regression'] = LogisticRegression()
        models['Support Vector Machines'] = svm.SVC()
        models['Decision Tree'] = DecisionTreeClassifier(min_samples_split=100)
        models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, learning_rate=1)
        models['Random Forest'] = RandomForestClassifier(n_estimators=100, min_samples_split=100)
        models['XGB'] = XGBClassifier(n_estimators=100, learning_rate=1, objective='binary:logistic')

        ##### Metrics to compare models
        from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, cohen_kappa_score

        res = {}
        for model in MODELS:
            ## fit classifier
            models[model].fit(X_train, Y_train)

            ## perdict
            pred = models[model].predict(X_val)  

            ## kpis
            res[model] = {
                'Accuracy' : round(float(accuracy_score(Y_val, pred)), 4),
                'Precision' : round(float(precision_score(Y_val, pred)), 4), 
                'ROC AUC Score' : round(float(roc_auc_score(Y_val, pred)), 4), 
                'Cohen Kappa' : round(float(cohen_kappa_score(Y_val, pred)), 4)
            }

        return pd.DataFrame(res)