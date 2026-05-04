import pyreadr
import pandas as pd
import glob
import numpy as np
import os
import pickle
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.datasets import load_breast_cancer
import matplotlib.pyplot as plt

cancer_name = "COAD"
path = f"data/{cancer_name}_clinical.pickle"

X, y = load_breast_cancer()
print("x format\n", X.head())
#load the pancancer object
with open(path, "rb") as f:
    df = pickle.load(f)

print(df.head())
print(df.columns)
Xt = df[list(set(df.columns)-set(["time", "bcr_patient_barcode","status"]))]
y = np.array([(bool(i),j ) for i, j in zip(df["status"], df["time"])],dtype=[('status', 'bool_'), ('time', '<f4')])


print(Xt.head())

def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    print("coefs :\n", coefs)
    print("alpha_min :\n", alpha_min)
    print("coefs.loc[:, alpha_min] shape : \n",coefs.loc[:, alpha_min].shape)
    if len(coefs.loc[:, alpha_min].shape) > 1:
        top_coefs = coefs.loc[:, alpha_min].iloc[0].map(abs).sort_values().tail(n_highlight)
    else:
        top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")

### alphas predetermined
alphas = 10.0 ** np.linspace(-4, 4, 50)
coefficients = {}

cph = CoxPHSurvivalAnalysis()
for alpha in alphas:
    cph.set_params(alpha=alpha)
    cph.fit(Xt, y)
    key = round(alpha, 5)
    coefficients[key] = cph.coef_

coefficients = pd.DataFrame.from_dict(coefficients).rename_axis(index="feature", columns="alpha").set_index(Xt.columns)
plot_coefficients(coefficients, n_highlight=5)
plt.show()

## elastic net
# cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.0001, n_alphas=100)
# cox_elastic_net.fit(Xt, y)  

# coefficients_elastic_net = pd.DataFrame(
#     cox_elastic_net.coef_, index=Xt.columns, columns=np.round(cox_elastic_net.alphas_, 5)
# )


# plot_coefficients(coefficients_elastic_net, n_highlight=5)
# plt.show()

#choosing penalty strengh alpha
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw

coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.0001, max_iter=100, fit_baseline_model=True))
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)
coxnet_pipe.fit(Xt, y)

estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
cv = KFold(n_splits=5, shuffle=True, random_state=0)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in map(float, estimated_alphas)]},
    cv=cv,
    error_score=0.5,
    n_jobs=1,
).fit(Xt, y)

cv_results = pd.DataFrame(gcv.cv_results_)

alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)

plt.show()

### best model 

best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_coefs = pd.DataFrame(best_model.coef_, index=Xt.columns, columns=["coefficient"])

non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print(f"Number of non-zero coefficients: {non_zero}")

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)

plt.show()
### evaluate the model / BRIER score 
print(best_model)
survs = best_model.predict_survival_function(Xt)
times = np.arange(365, 1826)
preds = np.asarray([[fn(t) for t in times] for fn in survs])

score = integrated_brier_score(y, y, preds, times)
print("ibs : ", round(score, 4))

### C-index de uno
preds_for_cindex = best_model.predict(Xt)
c_index = concordance_index_ipcw(y, y, preds_for_cindex)
print("cindex : ",c_index)