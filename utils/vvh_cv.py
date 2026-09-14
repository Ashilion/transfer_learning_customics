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


def vvh_cv(estimator,alpha, X_train, y_train, X_test, y_test, ridge = False):
    if not ridge:
        pred_train = estimator.predict(X_train, alpha =alpha)
        pred_test = estimator.predict(X_test, alpha = alpha)
    else:
        pred_train = estimator.predict(X_train)
        pred_test = estimator.predict(X_test)
    pred_whole = np.concatenate([pred_train, pred_test])
    
    event_train = y_train['status']
    time_train = y_train['time']
    event_test = y_test['status']
    time_test = y_test['time']
    
    event_whole = np.concatenate([event_train, event_test])
    time_whole = np.concatenate([time_train, time_test])
    
    ll_train = log_likelihood_cox(pred_train, event_train, time_train)
    ll_whole = log_likelihood_cox(pred_whole, event_whole, time_whole)
    
    return -2 *(ll_whole - ll_train)


# import numpy as np

# def cv_partial_loglik_vvh(pred_train,
#                           pred_test,
#                           y_train,
#                           y_test):

#     # concaténer train + validation
#     eta = np.concatenate([pred_train, pred_test])

#     time = np.concatenate([y_train["time"], y_test["time"]])
#     event = np.concatenate([y_train["status"], y_test["status"]])

#     # indicateur : observation appartenant au fold de validation
#     is_test = np.concatenate([
#         np.zeros(len(pred_train), dtype=bool),
#         np.ones(len(pred_test), dtype=bool)
#     ])

#     # tri décroissant des temps
#     order = np.argsort(-time)

#     eta = eta[order]
#     time = time[order]
#     event = event[order]
#     is_test = is_test[order]

#     exp_eta = np.exp(eta)

#     # sommes cumulées = dénominateurs des ensembles de risque
#     risk_sum = np.cumsum(exp_eta)

#     ll = 0.0

#     for i in range(len(time)):
#         if event[i] and is_test[i]:
#             ll += eta[i] - np.log(risk_sum[i])

#     return ll

# def vvh_cv(estimator,
#            alpha,
#            X_train,
#            y_train,
#            X_test,
#            y_test,
#            ridge=False):

#     if ridge:
#         pred_train = estimator.predict(X_train)
#         pred_test = estimator.predict(X_test)
#     else:
#         pred_train = estimator.predict(X_train, alpha=alpha)
#         pred_test = estimator.predict(X_test, alpha=alpha)

#     ll = cv_partial_loglik_vvh(
#         pred_train,
#         pred_test,
#         y_train,
#         y_test
#     )

#     return -2 * ll