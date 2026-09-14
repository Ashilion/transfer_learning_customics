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
  alpha = 0.01,
  lambda = lambda_grid,
  standardize = FALSE , # IMPORTANT (déjà scalé en Python)
  thresh = 1e-15
)

use_same_weights = TRUE

python_weights <- read.csv("data/compare_alpha/COAD_coef_python.csv")
if (use_same_weights){

}
# -----------------------
# PREDICTIONS R
# -----------------------
indices <- c(1, 26, 51, 76, 100) 
preds <- list()
vvh_score<- c()

for (i in seq_along(lambda_grid)) { 
  lambda_val <- lambda_grid[i]
  
  pred <- predict(
    fit,
    newx = X_test,
    s = lambda_val,
    type = "link"
  )
  
  preds[[i]] <- as.vector(pred)

  #vvh 
  whole_x = rbind(X_train, X_test)
  whole_y = c(y_train, y_test)
  pred_whole = predict(fit, newx=whole_x, s = lambda_val,
    type = "link")
  loglik_whole = glmnet::coxnet.deviance(pred_whole,
      y=whole_y)
  pred_train  = predict(fit, newx=X_train, s = lambda_val,
    type = "link")
  loglik_train = glmnet::coxnet.deviance(pred=pred_train , y=y_train)
  res = loglik_whole - loglik_train
  vvh_score[i] <- res
}

# -----------------------
# CONVERT TO DATAFRAMES
# -----------------------

# Python:
# preds_df = pd.DataFrame(preds)
#
# => chaque élément de preds devient une ligne
preds_df <- as.data.frame(do.call(rbind, preds))

vvh_df <- data.frame(vvh_score = vvh_score)

print(head(preds_df))
print(head(vvh_df))

# -----------------------
# EXPORT
# -----------------------
write.csv(
  preds_df,
  "data/compare_alpha/COAD_pred_r_multi_full.csv",
  row.names = FALSE
)

write.csv(
  vvh_df,
  "data/compare_alpha/COAD_vvh_r_full.csv",
  row.names = FALSE
)

# -----------------------
# COEFS
# -----------------------
coef_matrix = coef(fit)
print(coef_matrix)

library(Matrix)

# coef_matrix is your dgCMatrix
# df <- as.data.frame(as.matrix(coef_matrix))

df <- data.frame(
  variable = rownames(coef_matrix),
  as.matrix(coef_matrix),
  row.names = NULL
)

print(head(df))

if(!use_same_weights){

  write.csv(df, "data/compare_alpha/COAD_coef_r.csv", row.names = FALSE)
}