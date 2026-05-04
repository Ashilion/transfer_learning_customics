import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis

def log_likelihood_cox(linear_predictors, event, time):
    order = np.argsort(-time)
    linear_predictors = linear_predictors[order]
    event = event[order]
    
    exp_predictors = np.exp(linear_predictors)
    risk_set = np.cumsum(exp_predictors)
    
    ll = np.sum(event * (linear_predictors - np.log(risk_set)))
    
    return ll


def vvh_cv(estimator,alpha, X_train, y_train, X_test, y_test):
    pred_train = estimator.predict(X_train, alpha =alpha)
    pred_test = estimator.predict(X_test, alpha = alpha)
    
    pred_whole = np.concatenate([pred_train, pred_test])
    
    event_train = y_train['status']
    time_train = y_train['time']
    event_test = y_test['status']
    time_test = y_test['time']
    
    event_whole = np.concatenate([event_train, event_test])
    time_whole = np.concatenate([time_train, time_test])
    
    ll_train = log_likelihood_cox(pred_train, event_train, time_train)
    ll_whole = log_likelihood_cox(pred_whole, event_whole, time_whole)
    
    return ll_whole - ll_train