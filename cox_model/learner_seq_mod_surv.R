LearnerSeqMod = R6::R6Class("LearnerSeqMod",
  inherit = mlr3proba::LearnerSurv,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, param_set=ps(), predict_types="distr",
                          feature_types=character(), properties=character(),
                          packages=character(),
                          label=NA_character_, man=NA_character_) {

      necessary_pars = c("blocks", "clinical_fav", "nlambdas", "nfolds",
        "CV_measure", "cv_save_path", "seed")
        # check if necessary pars are in pars set
        if (!all(necessary_pars %in% param_set$ids())) {
            missing_params = setdiff(necessary_pars,
                intersect(param_set$ids(), necessary_pars))
            stop(paste("missing parameters: ",
                 paste(missing_params, collapse=", ")))
        }

        packages = c(packages, "glmnet", "survival", "survAUC", "SurvMetrics",
            "caret", "data.table", "mlr3extralearners")

      super$initialize(
        id = id,
        param_set=param_set, 
        predict_types = predict_types,
        feature_types = feature_types,
        properties = properties,
        packages = packages,
        label = label,
        man = man
      )
    }
  ),
  private = list(
    train_jdr = function(x, y, pars) {
        warning("not implemented, abstract class")
        return(list(x=as.matrix(x), jdr=NULL))
    },
    predict_jdr = function(newx, jdr, pars) {
        warning("not implemented, abstract class")
        return(newx)
    },
    .train = function(task) {

      #source("R/misc/learner_utils.R")
      # get parameters for training
      pars = self$param_set$get_values(tags="train")
      pars$task_id = task$id

      # Load data
      x = task$data(cols=task$feature_names)
      ysurv = task$truth()

      # setting seed for internal CV reproducibility
      set.seed(pars$seed)

      # Some checks

      # clinical must not be present in the blocks
      if (("clinical" %in% pars$blocks) & pars$clinical_fav)
        stop("Clinical cannot be in both blocks and latent space")

      # ncomp must be inferior or equal to smallest block
      # add check

      # Get blocks
      block_list = lapply(
        pars$blocks,
        function(bl) {
            as.matrix(x[, .SD, .SDcols=names(x) %like% paste0("_", bl, "$")])
        }
      )
      names(block_list) <- pars$blocks

      if (pars$clinical_fav) {
        clinicals = as.matrix(x[, .SD, .SDcols=names(x) %like% "_clinical$"])
      } 

      # special block for supervising RGCCA with residuals of clinical model
      # && is necessary here to get lazy evaluation
      if ("supervised_mode" %in% names(pars) &&
        pars$supervised_mode=="clinical") {
        if (!pars$clinical_fav)
            stop("clinical supervision should only be used with clinfav")
        block_list[["clinical"]] = clinicals
      }

      #mean1=apply(block_list[[1]], 2, mean)
      JDR = private$train_jdr(block_list, ysurv, pars)
      #mean2=apply(block_list[[1]], 2, mean)
      #print(mean1-mean2)

      if (pars$clinical_fav) { # adding clinical data if necessary
        JDR$x = cbind(JDR$x, clinicals)
      }

      #print(JDR$x)

      # implement case where there is only 1 comp and no clinical data
      if (ncol(JDR$x)==1) {
        one_column_flag = TRUE
        JDR$x = cbind(JDR$x, rep(0, nrow(JDR$x)))
        lambdas = 0
        glmnet_fit = glmnet::glmnet(JDR$x, ysurv, family="cox", alpha=0,
                                    lambda=lambdas)
      } else {
        one_column_flag = FALSE
        #Fit glmnet to get lambda sequence used in internal CV loop
        glmnet_fit = glmnet::glmnet(JDR$x, ysurv, family="cox", alpha=0,
                                    nlambda=pars$nlambdas)
        lambdas = glmnet_fit$lambda
      }

      #keeping standard deviations for later
      standevs = apply(JDR$x, 2, sd)

      folds = caret::createFolds(factor(ysurv[, "status"]), k=pars$nfolds, list=TRUE)

      #results = vector("list", length=pars$nfolds)
      if(pars$CV_measure=="linpred") {
        results = matrix(NA, nrow=nrow(JDR$x), ncol=length(lambdas))
      } else {
        results = matrix(NA, nrow=pars$nfolds, ncol=length(lambdas))
      }
      for (i in seq_along(folds)) { # Internal CV loop
        print(i)
        #train
        f = folds[[i]]
        ytrain = ysurv[-f]
        fold_train = lapply(block_list, function(x) x[-f,])

        JDR_train = private$train_jdr(fold_train, ytrain, pars)

        if(pars$clinical_fav) {
          JDR_train$x = cbind(JDR_train$x, clinicals[-f,])
        }
        if (one_column_flag) {
          JDR_train$x = cbind(JDR_train$x, rep(0, nrow(JDR_train$x)))
        }
        
        # fit glmnet on train data
        glmnet_train = glmnet::glmnet(JDR_train$x, ytrain,
                                      family="cox", alpha=0, lambda=lambdas)

        #test
        ytest = ysurv[f]
        fold_test = lapply(
          block_list, function(x) x[f,]
        )
        names(fold_test) = names(block_list)

        JDR_test = private$predict_jdr(fold_test, JDR_train$jdr, pars)
        if(pars$clinical_fav) {
          JDR_test$x = cbind(JDR_test$x, clinicals[f,])
        }
        if (one_column_flag) {
          JDR_test$x = cbind(JDR_test$x, rep(0, nrow(JDR_test$x)))
        }

        # Computing CV measure
        if (pars$CV_measure == "cindex") {
          fit = predict(glmnet_train, newx=JDR_test$x)
          res = apply(fit, 2, function(lp) survAUC::UnoC(ytrain, ytest, lp))
          results[i, ] = res

        } else if (pars$CV_measure == "ibs") {
          res = vector(mode="numeric", length=length(lambdas))
          times = sort(unique(ytrain[, "time"]))
          coef = coef(glmnet_train)
          for (j in seq_along(lambdas)) {
            cox_helper = transform_cox_model(coef[, j], JDR_train$x, ytrain)
            newdata <- as.data.frame(cbind(ytest, JDR_test$x))
            colnames(newdata)[1:2] <- c("OS_days", "OS")
            score = survival::brier(cox_helper, newdata=newdata, times=times)
            res[j] = Dint(times, score$brier,
              range=c(times[1], times[length(times)]))
          }
          results[i, ] = res/(max(times)-min(times))

        } else if (pars$CV_measure == "ibsRR") {
          res = vector(mode="numeric", length=length(lambdas))
          times = sort(unique(ytrain[, "time"]))
          maxtime = times[length(times)]
          coef = coef(glmnet_train)
          for (j in seq_along(lambdas)) {
            cox_helper = transform_cox_model(coef[, j], JDR_train$x, ytrain)
            newdata <- as.data.frame(cbind(ytest, JDR_test$x))
            colnames(newdata)[1:2] <- c("OS_days", "OS")
            pred = riskRegression::Score(list(cox_helper), Hist(OS_days, OS)~1,
              data=newdata,
              metrics="Brier",
              summary="ibs",
              times=times,
              se.fit=F,
              conf.int=F)
            #res[j] = pred$Brier$score[
              #model=="coxph" & times==maxtime,
              #IBS
            #]
            res[j] = tail(pred$Brier$score[model=="coxph", IBS], 1)
          }
          results[i, ] = res

        } else if (pars$CV_measure == "basic") {
          fit = predict(glmnet_train, newx=JDR_test$x)
          loglik_test = glmnet::coxnet.deviance(pred=fit, y=ytest)
          results[i, ] = loglik_test

        } else if (pars$CV_measure == "V&VH") {
          coef = coef(glmnet_train)
          whole_x = rbind(JDR_train$x, JDR_test$x)
          whole_y = c(ytrain, ytest)
          pred_whole = predict(glmnet_train, newx=whole_x)
          loglik_whole = glmnet::coxnet.deviance(pred_whole,
              y=c(ytrain, ytest))
          fit = predict(glmnet_train, newx=JDR_train$x)
          loglik_train = glmnet::coxnet.deviance(pred=fit, y=ytrain)
          results[i, ] = loglik_whole - loglik_train

        } else if (pars$CV_measure == "deviance") {
          res = vector(mode="numeric", length=length(lambdas))
          coef = coef(glmnet_train)
          for (j in seq_along(lambdas)) {
            cox_helper = transform_cox_model(coef[, j], JDR_test$x, ytest)
            res[j] = sum(residuals(cox_helper, type="deviance")^2)
          }
          results[i, ] = res

        } else if (pars$CV_measure == "linpred") {
          res = vector(mode="numeric", length=length(lambdas))
          results[f, ] = predict(glmnet_train, newx=JDR_test$x)

        } else if (pars$CV_measure == "auc") {
          fit = predict(glmnet_train, newx=JDR_test$x)
          res = apply(fit, 2,
            function(lp) survAUC::AUC.uno(ytrain, ytest, lp,
                                          times=sort(ytrain[,"time"])))
          res = sapply(res, `[[`, "iauc")
          results[i, ] = res
        }
      }
      #results = Reduce(rbind, results)

      #select_best = ifelse(pars$CV_measure %in% c("cindex, auc"), # LA CONNERIE !
      select_best = if(pars$CV_measure %in% c("cindex", "auc", "C")) {
          which.max
      } else {
          which.min
      }
      if(pars$CV_measure=="linpred") {
        aggr_results = glmnet::coxnet.deviance(results, y=ysurv)
      } else if(pars$CV_measure=="basic" | pars$CV_measure=="V&VH" ) {
        #aggr_results = -2*apply(results, 2, sum)
        aggr_results = apply(results, 2, sum)
      } else if(pars$CV_measure=="deviance") {
        aggr_results = apply(results, 2, sum)
      } else {
        aggr_results = apply(results, 2, mean, na.rm=T)
      }
      best_lambda = lambdas[select_best(aggr_results)]

      list(JDR=JDR$jdr,
           glmnet=glmnet_fit,
           best_lambda=best_lambda,
           standevs=standevs,
           cv_grid=results,
           cv_results=aggr_results,
           folds=folds,
           y=ysurv,
           x=JDR$x)
    },

    .predict = function(task) {
      # get parameters with tag "predict"
      pars = self$param_set$get_values(tags="predict")

      # get newdata and ensure same ordering in train and predict
      newx = mlr3extralearners:::ordered_features(task, self)
      block_list = lapply(
        pars$blocks,
        function(bl) as.matrix(newx[,
          .SD,
          .SDcols=names(newx) %like% paste0("_", bl, "$")
        ])
      )
      names(block_list) = pars$blocks

      JDR_new = private$predict_jdr(block_list, self$model$JDR, pars)

      if(pars$clinical_fav) {
        clinicals = as.matrix(newx[, .SD, .SDcols=names(newx) %like% "_clinical$"])
        JDR_new$x = cbind(JDR_new$x, clinicals)
      }

      # if there is only one comp
      if (ncol(JDR_new$x)==1) {
        #fit = survival::survfit(self$model$coxph,
        #                        newdata=as.data.frame(JDR_new$x),
        #                        se.fit=FALSE)
        #lp = predict(self$model$coxph,
        #               newdata=as.data.frame(JDR_new$x))
        #return(mlr3proba::.surv_return(times=fit$time,
        #                               surv=t(fit$surv),
        #                               lp=lp))
        JDR_new$x = cbind(JDR_new$x, rep(0, nrow(JDR_new$x)))
      }
      
      # Calculate predictions for the selected predict type.
      which_lambda = which(self$model$best_lambda==self$model$glmnet$lambda)
      fit = survival::survfit(self$model$glmnet,
                              s=self$model$best_lambda,
                              x=self$model$x,
                              y=self$model$y,
                              newx=JDR_new$x,
                              se.fit=FALSE)
      lp = predict(self$model$glmnet,
                   newx=JDR_new$x,
                   s=self$model$best_lambda)

      # getting coefficients
      beta = coef(self$model$glmnet, s=self$model$best_lambda)[,1]
      stanbeta = self$model$standevs * beta

      if (!is.null(pars$cv_save_path)) {
        to_save = list(
          learner_id = self$id,
          task_id = task$id,
          fold = task$row_ids,
          lambdas = self$model$glmnet$lambda,
          best_lambda = self$model$best_lambda,
          cv_grid = self$model$cv_grid,
          cv_results = self$model$cv_results,
          beta=beta,
          standevs = self$model$standevs,
          stanbeta=stanbeta
        )
        saveRDS(to_save, file=paste0(pars$cv_save_path, "/",
          rlang::hash(to_save), ".rds"))
      }

      return(mlr3proba::surv_return(times=fit$time,
                                    surv=t(fit$surv),
                                    lp=lp))

    }
  )
)
