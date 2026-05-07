library(glmnet)
library(survival)

# -----------------------
# LOAD DATA
# -----------------------
X_train <- as.matrix(read.csv("data/compare_alpha/COAD_X_train.csv"))
X_test  <- as.matrix(read.csv("data/compare_alpha/COAD_X_test.csv"))

y_train_df <- read.csv("data/compare_alpha/COAD_y_train.csv")
y_test_df  <- read.csv("data/compare_alpha/COAD_y_test.csv")

lambda_grid <- read.csv("data/compare_alpha/COAD_lambda_grid.csv")$lambda

y_train <- Surv(y_train_df$time, y_train_df$status)
y_test  <- Surv(y_test_df$time, y_test_df$status)

# -----------------------
# FIT MODEL (same lambda)
# -----------------------
fit <- glmnet(
  x = X_train,
  y = y_train,
  family = "cox",
  alpha = 0.05,
  lambda = lambda_grid,
  standardize = FALSE  # IMPORTANT (déjà scalé en Python)
)

# -----------------------
# PREDICTIONS R
# -----------------------
indices <- c(1, 26, 51, 76, 100) 
preds_list <- list()
vvh_list <- list()

for (i in seq_along(indices)) {
  idx <- indices[i]
  
  pred <- predict(
    fit,
    newx = X_test,
    s = lambda_grid[idx],
    type = "link"
  )
  
  preds_list[[paste0("alpha_", idx-1)]] <- as.vector(pred)

  #vvh 
  coef = coef(fit)
  whole_x = rbind(X_train, X_test)
  whole_y = c(y_train, y_test)
  pred_whole = predict(fit, newx=whole_x, s = lambda_grid[idx],
    type = "link")
  loglik_whole = glmnet::coxnet.deviance(pred_whole,
      y=c(y_train, y_test))
  pred_train  = predict(fit, newx=X_train, s = lambda_grid[idx],
    type = "link")
  loglik_train = glmnet::coxnet.deviance(pred=pred_train , y=y_train)
  res = loglik_whole - loglik_train
  vvh_list[[paste0("alpha_", idx-1)]] <- res
}

preds_df <- as.data.frame(preds_list)
write.csv(preds_df, "data/compare_alpha/COAD_pred_r_multi.csv", row.names = FALSE)

vvh_df <- as.data.frame(vvh_list)
write.csv(vvh_df, "data/compare_alpha/COAD_vvh_r.csv", row.names = FALSE)