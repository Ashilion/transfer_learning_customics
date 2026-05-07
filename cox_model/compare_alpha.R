library(glmnet)
library(survival)

Xt <- read.csv("data/compare_alpha/COAD_Xt.csv")
y <- read.csv("data/compare_alpha/COAD_y.csv")

y_surv <- Surv(time = y$time, event = y$status)

fit <- glmnet(
  x = as.matrix(Xt),
  y = y_surv,
  family = "cox",
  # alpha = 0.001 ,             # équivalent l1_ratio
  alpha = 0 ,             # équivalent l1_ratio

  nlambda = 100,               # équivalent n_alphas
  lambda.min.ratio = 0.0001 ,  # équivalent alpha_min_ratio
  standardize = FALSE  # IMPORTANT (déjà scalé en Python)
)

lambdas <- fit$lambda

write.csv(
  data.frame(lambda = lambdas),
  # "data/compare_alpha/COAD_lambda_R.csv",
  "data/compare_alpha/COAD_lambda_R_l2.csv",
  row.names = FALSE
)
print(lambdas)