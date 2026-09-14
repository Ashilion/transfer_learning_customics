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

# ml3 =================================================================================
.libPaths("r_libs")
library(mlr3)
library(mlr3proba)
library(mlr3extralearners)

# data
train_dt <- data.table(
  time   = y_train_df$time,
  status = y_train_df$status,
  X_train
)

test_dt <- data.table(
  time   = y_test_df$time,
  status = y_test_df$status,
  X_test
)

colnames(train_dt)[3:ncol(train_dt)] <- paste0(
  "x", seq_len(ncol(X_train))
)

colnames(test_dt)[3:ncol(test_dt)] <- paste0(
  "x", seq_len(ncol(X_test))
)

task_train <- TaskSurv$new(
  id = "coad_train",
  backend = train_dt,
  time = "time",
  event = "status"
)

task_test <- TaskSurv$new(
  id = "coad_test",
  backend = test_dt,
  time = "time",
  event = "status"
)

# training
mlr_learners$get("surv.glmnet")

cox = lrn("surv.glmnet",
  alpha = 0.05,
  lambda = lambda_grid,
  standardize = FALSE , 
  thresh = 1e-15
  )

cox$train(task_train)

p = cox$predict(task_test)

graf_scores = numeric(length(lambda_grid))
cindex_scores =      numeric(length(lambda_grid))
times = sort(unique(y_test_df$time))
remove_last_time_point = FALSE
if (remove_last_time_point){
  times = times[-length(times)]
}

preds_array_survival_fn = vector("list", length(lambda_grid))
preds_array = vector("list", length(lambda_grid))


for (i in seq_along(lambda_grid)) {

  # choose lambda
  cox$param_set$values$s = lambda_grid[i]

  # predict
  p = cox$predict(task_test)

  surv_probs = p$distr$survival(times)
  preds_array_survival_fn[[i]] = surv_probs

  # score
  graf_scores[i] = p$score(msr("surv.graf", times = times))
   
  #cindex 
  # print(p$lp)
  preds_array[[i]] = p$lp
  cindex_scores[i] = p$score(msr("surv.cindex", weight_meth="G2", id="cindex_default"), task = task_train,train_set =task_train$row_ids )
  cat("lambda", i, "=", lambda_grid[i],
      " graf =", graf_scores[i], "\n")
}

# ============================================================= ibs dataframe result
scores_df = data.frame(
  lambda_index = seq_along(lambda_grid),
  lambda = lambda_grid,
  graf = graf_scores
)

# print(scores_df)

write.csv(
  scores_df,
  "data/compare_alpha/COAD_ibs_r.csv",
  row.names = FALSE
)
# ============================================================= cindex dataframe result
cindex_df = data.frame(
  lambda_index = seq_along(lambda_grid),
  lambda = lambda_grid,
  cindex = cindex_scores
)

# print(cindex_df)

write.csv(
  cindex_df,
  "data/compare_alpha/COAD_cindex_r.csv",
  row.names = FALSE
)

# ============================================================= survival function preds dataframe result
preds_json = list()

for (i in seq_along(preds_array_survival_fn)) {

  preds_json[[i]] = list(
    lambda = lambda_grid[i],
    times = times,
    preds = unname(preds_array_survival_fn[[i]])
  )
}

library(jsonlite)

write_json(
  preds_json,
  "data/compare_alpha/preds_array_survival_fn.json",
  pretty = TRUE
)

# ============================================================= survival time preds dataframe

preds_json = list()

for (i in seq_along(preds_array)) {

  preds_json[[i]] = list(
    lambda = lambda_grid[i],
    times = times,
    preds = unname(preds_array[[i]])
  )
}

library(jsonlite)

write_json(
  preds_json,
  "data/compare_alpha/preds_array.json",
  pretty = TRUE
)
