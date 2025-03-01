import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score , roc_auc_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    # read the data as dataframe
    try:
        df = pd.read_csv(URL,sep=";")
        return df   
    except Exception as e:
        raise e

def evaluate(y_true, y_pred, pred_prob):
    '''mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    r2 = r2_score(y_true, y_pred)'''

    accuracy = accuracy_score(y_true, y_pred)
    roc_score = roc_auc_score(y_true,pred_prob, multi_class = 'ovr')

    return accuracy, roc_score

def main(n_estimators , max_depth):
    df = get_data()
    train, test = train_test_split(df, random_state=42)
    # train, test spilit with the raw data
    X_train = train.drop(["quality"], axis = 1)
    X_test = test.drop(["quality"],axis = 1)

    y_train = train[["quality"]]
    y_test = test[["quality"]]

    #Model training
    
    '''lr = ElasticNet()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)'''
    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=n_estimators , max_depth=max_depth)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        
        pred_prob = rf.predict_proba(X_test)
        # Evaluating the model

        #mae ,mse, rmse,r2 = evaluate(y_test, pred)

        accuracy , roc_score  = evaluate(y_test, pred, pred_prob)
        
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_score", roc_score)

        #mlflow model logging
        mlflow.sklearn.log_model(rf, "Random Forst model")

        #print(f"mean absolute error:{mae:.2f}, mean sqauered error: {mse:.2f} , root mean squared error:{rmse:.2f} ,r2 score :{r2:.2f}  ")
        print(f"accuracy is: {accuracy:.3f}, roc_score is: {roc_score:.3f}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()  # its a class
    args.add_argument("--n_estimators", "-n", default = "50", type = int)
    args.add_argument("--max_depth", "-m", default = 5, type = int)
    parse_args = args.parse_args() 

    try:
        main(n_estimators = parse_args.n_estimators,max_depth = parse_args.max_depth)
    except Exception as e:
        raise e