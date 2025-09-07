suppressPackageStartupMessages({
  library(caret)               # modeling & CV
  library(pbapply)             # progress (optional)
  library(dplyr)               # data wrangling
  library(pROC)                # ROC/AUC
  library(gbm)                 # GBM (caret backend)
  library(e1071)               # SVM (caret backend)
  library(kknn)                # KNN (caret backend)
  library(DMwR)                # SMOTE (used by caret sampling)
  library(lightgbm)            # LightGBM
  library(data.table)          # fast write
  library(rBayesianOptimization)  # Bayesian optimization
})

# --------------------
# I/O (relative paths, reproducible)
# --------------------
in_train <- file.path("results", "train.csv")  # produced by 01_preprocess.R
in_test  <- file.path("results", "test.csv")
train <- read.csv(in_train, stringsAsFactors = FALSE, check.names = FALSE)
test  <- read.csv(in_test,  stringsAsFactors = FALSE, check.names = FALSE)

# Minimal working example: select a few predictors (edit as needed)
vars_keep <- c("infection","Drinking","Smoking","HighGradeMalignancy","eGFR")
train <- train[, intersect(vars_keep, names(train)), drop = FALSE]
test  <- test[,  intersect(vars_keep, names(test)),  drop = FALSE]

# Binary outcome as factor: No/Yes
train$infection <- factor(ifelse(train$infection == 1, "Yes", "No"), levels = c("No","Yes"))
test$infection  <- factor(ifelse(test$infection  == 1, "Yes", "No"), levels = c("No","Yes"))

dev <- train   # development (train) set
vad <- test    # validation (test) set

# --------------------
# Models & CV settings
# --------------------
models <- c("glm","svmRadial","gbm","xgbTree","kknn")
models_names <- list(Logistic="glm", SVM="svmRadial", GBM="gbm", Xgboost="xgbTree", KNN="kknn")

set.seed(520)
train.control <- trainControl(
  method = "repeatedcv", number = 10, repeats = 5,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  sampling = "smote", savePredictions = "final", verboseIter = TRUE
)
attr(train.control, "positive") <- "Yes"

# Lightweight CV for Bayesian search
tune.control <- trainControl(
  method = "cv", number = 5,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  sampling = "smote", savePredictions = "none", verboseIter = FALSE
)

# --------------------
# Bayesian optimization helpers (maximize CV AUC)
# --------------------
.safe_score <- function(x) if (is.null(x) || !is.finite(x) || is.na(x)) 1e-6 else max(1e-6, x)

safe_train_auc <- function(method, grid) {
  fit <- tryCatch({
    caret::train(infection ~ ., data = dev, method = method,
                 metric = "ROC", trControl = tune.control, tuneGrid = grid)
  }, error = function(e) NULL)
  if (is.null(fit) || !"ROC" %in% names(fit$results)) return(1e-6)
  .safe_score(max(fit$results$ROC, na.rm = TRUE))
}

# SVM-RBF
svm_cv_bayes <- function(log10C, log10sigma) {
  grid <- expand.grid(C = 10^log10C, sigma = 10^log10sigma)
  list(Score = safe_train_auc("svmRadial", grid), Pred = 0)
}
# GBM
gbm_cv_bayes <- function(n_trees, depth, shrink, minobs) {
  grid <- expand.grid(
    n.trees = as.integer(round(n_trees)),
    interaction.depth = as.integer(round(depth)),
    shrinkage = shrink,
    n.minobsinnode = as.integer(round(minobs))
  )
  list(Score = safe_train_auc("gbm", grid), Pred = 0)
}
# XGBoost
xgb_cv_bayes <- function(nrounds, max_depth, eta, gamma, colsample, min_child_weight, subsample) {
  grid <- expand.grid(
    nrounds = as.integer(round(nrounds)),
    max_depth = as.integer(round(max_depth)),
    eta = eta, gamma = gamma,
    colsample_bytree = colsample,
    min_child_weight = min_child_weight,
    subsample = subsample
  )
  list(Score = safe_train_auc("xgbTree", grid), Pred = 0)
}
# KNN
knn_cv_bayes <- function(kmax) {
  grid <- expand.grid(kmax = as.integer(round(kmax)), distance = 1, kernel = "rectangular")
  list(Score = safe_train_auc("kknn", grid), Pred = 0)
}

# --------------------
# Run Bayesian search (small budgets demo; increase if needed)
# --------------------
set.seed(2025)
opt_svm <- BayesianOptimization(
  FUN = svm_cv_bayes,
  bounds = list(log10C = c(-2, 2), log10sigma = c(-3, 0)),
  init_points = 5, n_iter = 8, acq = "ucb", kappa = 2.0, verbose = TRUE
)
opt_gbm <- BayesianOptimization(
  FUN = gbm_cv_bayes,
  bounds = list(n_trees = c(120L, 500L), depth = c(2L, 4L),
                shrink = c(0.03, 0.12), minobs = c(5L, 20L)),
  init_points = 5, n_iter = 8, acq = "ucb", kappa = 2.0, verbose = TRUE
)
opt_xgb <- BayesianOptimization(
  FUN = xgb_cv_bayes,
  bounds = list(nrounds = c(80L, 400L), max_depth = c(2L, 4L), eta = c(0.03, 0.3),
                gamma = c(0, 2), colsample = c(0.6, 1.0),
                min_child_weight = c(1, 12), subsample = c(0.7, 1.0)),
  init_points = 6, n_iter = 10, acq = "ucb", kappa = 2.0, verbose = TRUE
)
opt_knn <- BayesianOptimization(
  FUN = knn_cv_bayes, bounds = list(kmax = c(5L, 60L)),
  init_points = 4, n_iter = 8, acq = "ucb", kappa = 2.0, verbose = TRUE
)

# --------------------
# Assemble tuned grids
# --------------------
glm.tune.grid <- NULL
svm.tune.grid <- expand.grid(C = 10^opt_svm$Best_Par["log10C"],
                             sigma = 10^opt_svm$Best_Par["log10sigma"])
gbm.tune.grid <- expand.grid(
  n.trees = as.integer(round(opt_gbm$Best_Par["n_trees"])),
  interaction.depth = as.integer(round(opt_gbm$Best_Par["depth"])),
  shrinkage = opt_gbm$Best_Par["shrink"],
  n.minobsinnode = as.integer(round(opt_gbm$Best_Par["minobs"]))
)
xgb.tune.grid <- expand.grid(
  nrounds = as.integer(round(opt_xgb$Best_Par["nrounds"])),
  max_depth = as.integer(round(opt_xgb$Best_Par["max_depth"])),
  eta = opt_xgb$Best_Par["eta"],
  gamma = opt_xgb$Best_Par["gamma"],
  colsample_bytree = opt_xgb$Best_Par["colsample"],
  min_child_weight = opt_xgb$Best_Par["min_child_weight"],
  subsample = opt_xgb$Best_Par["subsample"]
)
knn.tune.grid <- expand.grid(kmax = as.integer(round(opt_knn$Best_Par["kmax"])),
                             distance = 1, kernel = "rectangular")
Tune_table <- list(glm = glm.tune.grid, svmRadial = svm.tune.grid, gbm = gbm.tune.grid,
                   xgbTree = xgb.tune.grid, kknn = knn.tune.grid)

# --------------------
# Train caret models (SMOTE in-CV; scale only for SVM/KNN)
# --------------------
train_probe <- data.frame(infection = dev$infection)
test_probe  <- data.frame(infection = vad$infection)
importance  <- list()
ML_calss_model <- list()

preProcess_methods <- c("center", "scale")
pb <- txtProgressBar(min = 0, max = length(models), style = 3)

for (i in seq_along(models)) {
  model <- models[i]
  model_name <- names(models_names)[which(models_names == model)]
  pre_process <- if (model %in% c("svmRadial", "kknn")) preProcess_methods else NULL
  set.seed(52)
  fit <- caret::train(
    infection ~ ., data = dev, method = model,
    tuneGrid = Tune_table[[model]], trControl = train.control,
    metric = "ROC", preProcess = pre_process
  )
  train_Pro <- predict(fit, newdata = dev, type = "prob")
  test_Pro  <- predict(fit, newdata = vad, type = "prob")
  train_probe[[model_name]] <- train_Pro$Yes
  test_probe[[model_name]]  <- test_Pro$Yes
  ML_calss_model[[model_name]] <- fit
  importance[[model_name]] <- tryCatch(varImp(fit, scale = TRUE), error = function(e) NULL)
  setTxtProgressBar(pb, i)
}
close(pb)

# --------------------
# LightGBM: Bayes -> CV best_iter -> fit on full dev (no test leakage)
# --------------------
y_dev <- ifelse(dev$infection == "Yes", 1, 0)
X_dev <- as.matrix(dev[, setdiff(names(dev), "infection")])
dtrain_cv <- lgb.Dataset(X_dev, label = y_dev)

lgb_cv_bayes <- function(learning_rate, num_leaves, min_data_in_leaf, feature_fraction,
                         bagging_fraction, lambda_l1, lambda_l2) {
  params <- list(
    objective = "binary", metric = "auc",
    learning_rate = learning_rate,
    num_leaves = as.integer(round(num_leaves)),
    min_data_in_leaf = as.integer(round(min_data_in_leaf)),
    feature_fraction = feature_fraction, bagging_fraction = bagging_fraction,
    bagging_freq = 1, lambda_l1 = lambda_l1, lambda_l2 = lambda_l2,
    feature_pre_filter = FALSE, verbosity = -1
  )
  cv <- tryCatch(lgb.cv(params = params, data = dtrain_cv, nfold = 4,
                        nrounds = 1000, early_stopping_rounds = 20,
                        stratified = TRUE, verbose = -1), error = function(e) NULL)
  score <- if (!is.null(cv) && !is.null(cv$best_score)) cv$best_score else 1e-6
  list(Score = .safe_score(score), Pred = 0)
}

set.seed(2025)
opt_lgb <- BayesianOptimization(
  FUN = lgb_cv_bayes,
  bounds = list(learning_rate = c(0.02, 0.12), num_leaves = c(15L, 63L),
                min_data_in_leaf = c(8L, 30L), feature_fraction = c(0.7, 1.0),
                bagging_fraction = c(0.7, 1.0), lambda_l1 = c(0, 5), lambda_l2 = c(0, 5)),
  init_points = 8, n_iter = 12, acq = "ucb", kappa = 2.0, verbose = TRUE
)

best_params <- list(
  objective = "binary", metric = "auc",
  learning_rate = opt_lgb$Best_Par["learning_rate"],
  num_leaves = as.integer(round(opt_lgb$Best_Par["num_leaves"])),
  min_data_in_leaf = as.integer(round(opt_lgb$Best_Par["min_data_in_leaf"])),
  feature_fraction = opt_lgb$Best_Par["feature_fraction"],
  bagging_fraction = opt_lgb$Best_Par["bagging_fraction"],
  bagging_freq = 1,
  lambda_l1 = opt_lgb$Best_Par["lambda_l1"],
  lambda_l2 = opt_lgb$Best_Par["lambda_l2"],
  feature_pre_filter = FALSE, verbosity = -1
)

cv_final <- lgb.cv(params = best_params, data = dtrain_cv, nfold = 5, nrounds = 2000,
                   early_stopping_rounds = 50, stratified = TRUE, verbose = -1)
best_iter <- if (!is.null(cv_final$best_iter)) cv_final$best_iter else 200

dtrain_full <- lgb.Dataset(X_dev, label = y_dev)
lightgbm_model <- lgb.train(params = best_params, data = dtrain_full, nrounds = best_iter, verbose = -1)

features <- setdiff(names(dev), "infection")
train_probe$LightGBM <- predict(lightgbm_model, as.matrix(dev[, features]))
test_probe$LightGBM  <- predict(lightgbm_model, as.matrix(vad[, features]))

lgb_imp <- lgb.importance(lightgbm_model, percentage = TRUE)

# --------------------
# Evaluation & plots (no leakage; thresholds fixed from Train)
# --------------------
suppressPackageStartupMessages({
  library(ggplot2); library(tibble); library(cvms); library(rmda)
})

out_dir <- file.path("results", "modeling")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
safe_name <- function(x) gsub("[^[:alnum:]_.-]", "_", x, perl = TRUE)
safe_path <- function(fname) file.path(out_dir, safe_name(fname))
open_pdf <- function(path, width = 7, height = 7, family = "serif") {
  tryCatch(pdf(path, width = width, height = height, family = family),
           error = function(e) stop(sprintf("Cannot open PDF for writing: %s\nClose the file and retry.\n%s", path, e$message)))
}
safe_write_csv <- function(obj, path, ...) {
  tryCatch(write.csv(obj, path, ...), error = function(e) {
    alt <- safe_path(paste0(basename(path), ".bak.csv")); message(sprintf("Write failed, writing to backup: %s", alt)); write.csv(obj, alt, ...)
  })
}

# Feature importance (LightGBM and caret models)
if (!is.null(lgb_imp) && nrow(lgb_imp) > 0) {
  rt <- lgb_imp[order(lgb_imp$Gain), , drop = FALSE]
  p_lgb <- ggplot(rt, aes(x = Gain, y = reorder(Feature, Gain))) +
    geom_col(width = 0.6) + theme_classic() + labs(x = "Importance", y = "Features", title = "LightGBM")
  open_pdf(safe_path("LightGBM_importance.pdf"), 7, 5); print(p_lgb); dev.off()
}
caret_models <- c("Logistic","SVM","GBM","Xgboost","KNN")
for (model_name in caret_models) {
  imp <- importance[[model_name]]; if (is.null(imp)) next
  tab <- as.data.frame(imp$importance); tab$Features <- rownames(tab)
  score_col <- if ("Yes" %in% colnames(tab)) "Yes" else if ("Overall" %in% colnames(tab)) "Overall" else NA
  if (is.na(score_col)) next
  gp <- ggplot(tab, aes(x = .data[[score_col]], y = reorder(Features, .data[[score_col]]))) +
    geom_col(width = 0.6) + theme_classic() + labs(x = "Importance", y = "Features", title = model_name)
  open_pdf(safe_path(paste0(model_name, "_importance.pdf")), 7, 5); print(gp); dev.off()
}

# Calibration (Brier + 95%CI via bootstrap on Brier)
plot_calibration_brier <- function(data, label_col, model_names, newdata_tt, n_bins = 8, n_boot = 200) {
  y_true <- ifelse(data[[label_col]] == "Yes", 1, 0)
  calib_list <- list(); metrics_list <- list()
  for (mod in model_names) {
    pred <- data[[mod]]; if (is.null(pred) || all(is.na(pred)) || length(unique(pred)) <= 1) next
    brier <- mean((pred - y_true)^2, na.rm = TRUE)
    set.seed(2025)
    boot_brier <- replicate(n_boot, {
      idx <- sample.int(length(y_true), replace = TRUE)
      yb <- y_true[idx]; pb <- pred[idx]
      if (length(unique(yb)) < 2) return(NA_real_)
      mean((pb - yb)^2, na.rm = TRUE)
    })
    boot_brier <- boot_brier[is.finite(boot_brier)]
    if (length(boot_brier) == 0) next
    ci_low  <- unname(quantile(boot_brier, 0.025))
    ci_high <- unname(quantile(boot_brier, 0.975))
    metrics_list[[mod]] <- data.frame(
      Model = mod,
      Label = paste0(mod, " (Brier=", sprintf("%.3f", brier),
                     ", 95%CI:", sprintf("%.3f", ci_low), "-", sprintf("%.3f", ci_high), ")"),
      stringsAsFactors = FALSE
    )
    qs <- unique(quantile(pred, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
    if (length(qs) < 2) next
    df <- data.frame(pred = pred, obs = y_true) |>
      dplyr::filter(!is.na(pred)) |>
      dplyr::mutate(bin = cut(pred, breaks = qs, include.lowest = TRUE)) |>
      dplyr::group_by(bin) |>
      dplyr::summarise(mean_pred = mean(pred, na.rm = TRUE),
                       obs_rate  = mean(obs,  na.rm = TRUE), .groups = "drop") |>
      dplyr::mutate(Model = mod)
    calib_list[[mod]] <- df
  }
  calib_df   <- dplyr::bind_rows(calib_list)
  metrics_df <- dplyr::bind_rows(metrics_list)
  if (nrow(calib_df) == 0) return(ggplot() + ggtitle(paste0(newdata_tt, " (no data)")))
  calib_df <- dplyr::left_join(calib_df, metrics_df, by = "Model")
  lab_vec <- metrics_df$Label; names(lab_vec) <- metrics_df$Model
  ggplot(calib_df, aes(x = mean_pred, y = obs_rate, color = Model, group = Model)) +
    geom_line(size = 0.6) + geom_point(size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
    labs(title = paste0(newdata_tt, " Calibration (Brier ± 95%CI)"),
         x = "Mean predicted probability", y = "Observed event rate") +
    coord_fixed(ratio = 0.5) +
    theme_minimal(base_size = 14) + theme(legend.title = element_blank())
}

# Metrics at a fixed threshold (from Train via Youden J)
get_metrics <- function(probs, labels, thr) {
  predlab <- factor(ifelse(probs >= thr, "Yes","No"), levels=c("No","Yes"))
  cm <- caret::confusionMatrix(predlab, labels, positive="Yes", mode="everything")
  c(Threshold = thr,
    Accuracy   = unname(cm$overall["Accuracy"]),
    Sensitivity= unname(cm$byClass["Sensitivity"]),
    Specificity= unname(cm$byClass["Specificity"]),
    Precision  = unname(cm$byClass["Precision"]),
    F1         = unname(cm$byClass["F1"]))
}

# Point estimates + counts
metric_points <- function(labels_factor, probs, thr){
  y_true <- as.integer(labels_factor == "Yes")
  y_pred <- as.integer(probs >= thr)
  TP <- sum(y_true==1 & y_pred==1); TN <- sum(y_true==0 & y_pred==0)
  FP <- sum(y_true==0 & y_pred==1); FN <- sum(y_true==1 & y_pred==0)
  sens <- ifelse((TP+FN)>0, TP/(TP+FN), NA_real_)
  spec <- ifelse((TN+FP)>0, TN/(TN+FP), NA_real_)
  ppv  <- ifelse((TP+FP)>0, TP/(TP+FP), NA_real_)
  npv  <- ifelse((TN+FN)>0, TN/(TN+FN), NA_real_)
  f1   <- ifelse((2*TP + FP + FN)>0, 2*TP/(2*TP + FP + FN), NA_real_)
  acc  <- (TP+TN)/max(1, TP+TN+FP+FN)
  list(counts = c(TP=TP, TN=TN, FP=FP, FN=FN),
       point  = c(Accuracy=acc, Sensitivity=sens, Specificity=spec, PPV=ppv, NPV=npv, F1=f1))
}

# Stratified bootstrap CI for metrics
metric_ci_boot <- function(labels_factor, probs, thr, B=2000, seed=2025){
  set.seed(seed)
  y <- as.integer(labels_factor=="Yes")
  idx_pos <- which(y==1); idx_neg <- which(y==0)
  n_pos <- length(idx_pos); n_neg <- length(idx_neg)
  mat <- matrix(NA_real_, nrow=B, ncol=6)
  colnames(mat) <- c("Accuracy","Sensitivity","Specificity","PPV","NPV","F1")
  for (b in 1:B){
    samp <- c(sample(idx_pos, n_pos, replace=TRUE),
              sample(idx_neg, n_neg, replace=TRUE))
    mp <- metric_points(labels_factor[samp], probs[samp], thr)$point
    mat[b,] <- as.numeric(mp[colnames(mat)])
  }
  ci <- t(apply(mat, 2, function(v){
    v <- v[is.finite(v)]
    if (length(v) < 10) return(c(NA_real_, NA_real_))
    as.numeric(quantile(v, c(0.025, 0.975), na.rm=TRUE, names=FALSE))
  }))
  as.data.frame(ci)
}

# Thresholds from Train via Youden J (fixed for Train/Test)
plot_models_all <- c(caret_models, "LightGBM")
Train <- train_probe; Test <- test_probe
thr_map <- list()
for (m in intersect(plot_models_all, colnames(Train))) {
  roc_tr <- pROC::roc(response = Train$infection, predictor = Train[[m]],
                      levels = c("No","Yes"), direction = "<", quiet = TRUE)
  thr <- roc_tr$thresholds[which.max(roc_tr$sensitivities + roc_tr$specificities - 1)]
  if (!is.finite(thr) || length(thr)==0) thr <- 0.5
  thr_map[[m]] <- thr
}

# Calibration plots
for (split in c("Train","Test")) {
  dat <- if (split=="Train") Train else Test
  avail <- intersect(plot_models_all, colnames(dat)); if (length(avail)==0) next
  open_pdf(safe_path(paste0(split, "_Calibration_Brier.pdf")), 8, 4)
  print(plot_calibration_brier(dat, "infection", avail, split))
  dev.off()
}

# ROC curves + metrics tables + confusion matrices (fixed thresholds)
for (split in c("Train","Test")) {
  dat <- if (split=="Train") Train else Test
  avail <- intersect(plot_models_all, colnames(dat)); if (length(avail)==0) next
  
  # ROC
  ROC_list <- list(); ROC_label <- list()
  for (m in avail) {
    ROC <- tryCatch(pROC::roc(response = dat$infection, predictor = dat[[m]],
                              levels = c("No","Yes"), direction = "<", quiet = TRUE),
                    error = function(e) NULL)
    if (is.null(ROC)) next
    AUC <- round(pROC::auc(ROC), 3)
    CI  <- tryCatch(suppressWarnings(pROC::ci.auc(ROC)), error = function(e) c(NA, NA, NA))
    ROC_list[[m]]  <- ROC
    ROC_label[[m]] <- paste0(m, " (AUC=", sprintf("%0.3f", AUC),
                             ", 95%CI:", sprintf("%0.3f", CI[1]), "-", sprintf("%0.3f", CI[3]), ")")
  }
  if (length(ROC_list) > 0) {
    breaks_vec <- intersect(avail, names(ROC_list))
    label_vec  <- unlist(ROC_label)[breaks_vec]
    cols <- c(Logistic="#F4A55C",SVM="#FFFF00",KNN="#76C67F",GBM="#33B3E2",Xgboost="#00A08A",LightGBM="#CC6699")
    roc_plot <- pROC::ggroc(ROC_list[breaks_vec], size = 1.5, legacy.axes = TRUE) +
      theme_bw() + labs(title = paste0(split, " ROC")) +
      geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), colour = 'grey', linetype = 'dotdash') +
      scale_color_manual(values = cols[breaks_vec], breaks = breaks_vec, labels = label_vec) +
      scale_x_continuous(breaks = seq(0, 1, 0.2)) + scale_y_continuous(breaks = seq(0, 1, 0.2))
    open_pdf(safe_path(paste0(split, "_ROC.pdf")), 7, 7); print(roc_plot); dev.off()
  }
  
  # Metrics & confusion matrices (fixed thresholds from Train)
  Evaluation_metrics <- data.frame(Model=character(), Threshold=double(),
                                   Accuracy=double(), Sensitivity=double(),
                                   Specificity=double(), PPV=double(), F1=double())
  for (m in avail) {
    thr <- thr_map[[m]]; if (is.null(thr)) next
    met <- get_metrics(dat[[m]], dat$infection, thr)
    Evaluation_metrics <- rbind(Evaluation_metrics,
                                data.frame(Model=m, t(round(met, 3)), row.names = NULL))
    predlab <- factor(ifelse(dat[[m]] >= thr, "Yes", "No"), levels = c("No","Yes"))
    cm_tbl  <- tibble::as_tibble(table(data.frame(reference=dat$infection, prediction=predlab)))
    cm_plot <- cvms::plot_confusion_matrix(
      cm_tbl, target_col = "reference", prediction_col = "prediction", counts_col = "n",
      add_sums = FALSE, add_counts = TRUE, add_normalized = TRUE,
      palette = "Blues", theme_fn = ggplot2::theme_minimal, place_x_axis_above = TRUE,
      rotate_y_text = TRUE, digits = 1
    )
    open_pdf(safe_path(paste0(split, "_", m, "_cm.pdf")), 5, 5); print(cm_plot); dev.off()
  }
  safe_write_csv(Evaluation_metrics, safe_path(paste0(split, "_Evaluation_metrics.csv")), row.names = FALSE)
  
  # Metrics with 95%CI (bootstrap) + AUC CIs (DeLong)
  Evaluation_metrics_with_CI <- data.frame(
    Model=character(), Threshold=double(),
    Accuracy=double(), Accuracy_L=double(), Accuracy_U=double(),
    Sensitivity=double(), Sensitivity_L=double(), Sensitivity_U=double(),
    Specificity=double(), Specificity_L=double(), Specificity_U=double(),
    PPV=double(), PPV_L=double(), PPV_U=double(),
    NPV=double(), NPV_L=double(), NPV_U=double(),
    F1=double(), F1_L=double(), F1_U=double(),
    AUC=double(), AUC_L=double(), AUC_U=double(),
    TP=double(), TN=double(), FP=double(), FN=double(),
    stringsAsFactors = FALSE
  )
  for (m in avail) {
    thr <- thr_map[[m]]; if (is.null(thr)) next
    pts <- metric_points(dat$infection, dat[[m]], thr)
    ci  <- metric_ci_boot(dat$infection, dat[[m]], thr, B = 2000, seed = 2025)
    roc_obj <- tryCatch(pROC::roc(dat$infection, dat[[m]],
                                  levels = c("No","Yes"), direction = "<", quiet = TRUE),
                        error=function(e) NULL)
    auc_est <- if(!is.null(roc_obj)) as.numeric(pROC::auc(roc_obj)) else NA_real_
    auc_ci  <- if(!is.null(roc_obj)) as.numeric(pROC::ci.auc(roc_obj)) else c(NA,NA,NA)
    one <- data.frame(
      Model = m, Threshold = thr,
      Accuracy = pts$point["Accuracy"], Accuracy_L = ci["Accuracy",1], Accuracy_U = ci["Accuracy",2],
      Sensitivity = pts$point["Sensitivity"], Sensitivity_L = ci["Sensitivity",1], Sensitivity_U = ci["Sensitivity",2],
      Specificity = pts$point["Specificity"], Specificity_L = ci["Specificity",1], Specificity_U = ci["Specificity",2],
      PPV = pts$point["PPV"], PPV_L = ci["PPV",1], PPV_U = ci["PPV",2],
      NPV = pts$point["NPV"], NPV_L = ci["NPV",1], NPV_U = ci["NPV",2],
      F1  = pts$point["F1"],  F1_L  = ci["F1",1],  F1_U  = ci["F1",2],
      AUC = auc_est, AUC_L = auc_ci[1], AUC_U = auc_ci[3],
      TP = pts$counts["TP"], TN = pts$counts["TN"], FP = pts$counts["FP"], FN = pts$counts["FN"],
      check.names = FALSE
    )
    Evaluation_metrics_with_CI <- rbind(Evaluation_metrics_with_CI, one)
  }
  num_cols <- setdiff(names(Evaluation_metrics_with_CI), c("Model"))
  Evaluation_metrics_with_CI[num_cols] <- lapply(Evaluation_metrics_with_CI[num_cols], function(x){
    if (is.numeric(x)) round(x, 3) else x
  })
  if ("Threshold" %in% names(Evaluation_metrics_with_CI))
    Evaluation_metrics_with_CI$Threshold <- round(as.numeric(Evaluation_metrics_with_CI$Threshold), 6)
  safe_write_csv(Evaluation_metrics_with_CI,
                 safe_path(paste0(split, "_Evaluation_metrics_with_CI.csv")), row.names = FALSE)
}

# Decision curve analysis (probabilities only; no fixed threshold)
for (split in c("Train","Test")) {
  dat <- if (split=="Train") Train else Test
  avail <- intersect(plot_models_all, colnames(dat)); if (length(avail) == 0) next
  dca_data <- dat; dca_data$infection <- ifelse(dca_data$infection == "Yes", 1, 0)
  DCA_list <- lapply(avail, function(m){
    fml <- as.formula(paste("infection ~", m))
    set.seed(123); rmda::decision_curve(fml, data = dca_data, study.design = "cohort", bootstraps = 50)
  })
  names(DCA_list) <- avail
  open_pdf(safe_path(paste0(split, "_DCA.pdf")), 7, 7)
  print(rmda::plot_decision_curve(DCA_list, curve.names = avail,
                                  cost.benefit.axis = FALSE, confidence.intervals = "none", lwd = 4,
                                  legend.position = "topright"))
  dev.off()
  safe_write_csv(dca_data, safe_path(paste0(split, "_DCA_data.csv")), row.names = FALSE)
}

# --------------------
# SHAP for GBM (fastshap + shapviz) on Train only (no leakage)
# --------------------
suppressPackageStartupMessages({ library(fastshap); library(shapviz) })

if ("GBM" %in% names(ML_calss_model)) {
  gbm_caret <- ML_calss_model[["GBM"]]
  features  <- setdiff(names(dev), "infection")
  pred_yes <- function(object, newdata) as.numeric(predict(object, newdata = newdata, type = "prob")[, "Yes"])
  X_bg   <- dev[, features, drop = FALSE]
  X_shap <- X_bg[seq_len(min(145, nrow(X_bg))), , drop = FALSE]
  set.seed(123)
  shap_gbm <- fastshap::explain(object = gbm_caret, X = X_bg, pred_wrapper = pred_yes,
                                newdata = X_shap, nsim = 200, adjust = TRUE)
  baseline_gbm <- mean(pred_yes(gbm_caret, X_shap))
  sv_gbm <- shapviz::shapviz(as.matrix(shap_gbm), X = as.data.frame(X_shap), baseline = baseline_gbm)
  
  open_pdf(safe_path("SHAP_GBM_beeswarm.pdf"), 7, 5)
  print(sv_importance(sv_gbm, kind = "beeswarm")); dev.off()
  
  open_pdf(safe_path("SHAP_GBM_bar.pdf"), 7, 5)
  print(sv_importance(sv_gbm, kind = "bar", show_numbers = FALSE)); dev.off()
  
  dep_list <- intersect(c("Drinking","Smoking","HighGradeMalignancy","eGFR"), colnames(sv_gbm$X))
  for (v in dep_list) {
    open_pdf(safe_path(paste0("SHAP_GBM_dependence_", v, ".pdf")), 6, 5)
    print(sv_dependence(sv_gbm, v = v)); dev.off()
  }
  
  set.seed(123)
  sample_ids <- sample(seq_len(nrow(X_shap)), size = min(5, nrow(X_shap)))
  for (row_id in sample_ids) {
    open_pdf(safe_path(paste0("SHAP_GBM_force_case", row_id, ".pdf")), 7, 5)
    print(sv_force(sv_gbm, row_id = row_id, size = 9)); dev.off()
    open_pdf(safe_path(paste0("SHAP_GBM_waterfall_case", row_id, ".pdf")), 5, 5)
    print(sv_waterfall(sv_gbm, row_id = row_id)); dev.off()
  }
  
  # Two case studies on Test (10th/90th percentile)
  probs_test <- test_probe$GBM; stopifnot(length(probs_test) == nrow(vad))
  q_lo <- as.numeric(quantile(probs_test, 0.10, na.rm = TRUE))
  q_hi <- as.numeric(quantile(probs_test, 0.90, na.rm = TRUE))
  id_lo <- which.min(abs(probs_test - q_lo)); id_hi <- which.min(abs(probs_test - q_hi))
  thr_gbm <- if (!is.null(thr_map[["GBM"]])) thr_map[["GBM"]] else 0.5
  X_case <- vad[c(id_lo, id_hi), features, drop = FALSE]
  set.seed(123)
  shap_cases <- fastshap::explain(object = gbm_caret, X = X_bg, pred_wrapper = pred_yes,
                                  newdata = X_case, nsim = 300, adjust = TRUE)
  baseline_cases <- mean(pred_yes(gbm_caret, X_case))
  sv_cases <- shapviz::shapviz(as.matrix(shap_cases), X = as.data.frame(X_case), baseline = baseline_cases)
  case_probs <- probs_test[c(id_lo, id_hi)]
  case_tags  <- c("LowRisk_P10", "HighRisk_P90")
  for (j in 1:2) {
    tag <- paste0(case_tags[j], "_prob", sprintf("%.3f", case_probs[j]),
                  "_thr", sprintf("%.3f", thr_gbm),
                  "_pred", ifelse(case_probs[j] >= thr_gbm, "Yes","No"))
    open_pdf(safe_path(paste0("SHAP_GBM_Force_", tag, ".pdf")), 7, 5)
    print(sv_force(sv_cases, row_id = j, size = 9)); dev.off()
    open_pdf(safe_path(paste0("SHAP_GBM_Waterfall_", tag, ".pdf")), 5, 5)
    print(sv_waterfall(sv_cases, row_id = j)); dev.off()
  }
}

# --------------------
# Export best params & thresholds (for reproducibility)
# --------------------
suppressPackageStartupMessages({ library(tidyr) })

extract_caret_params2 <- function(fit, model_name){
  if (is.null(fit)) return(tibble::tibble())
  if (!is.null(fit$bestTune)) {
    df <- fit$bestTune; df[] <- lapply(df, function(x) if (is.factor(x)) as.character(x) else x)
    tibble::tibble(Model = model_name, Parameter = names(df), Value = as.character(unlist(df[1, , drop = TRUE])))
  } else if (model_name == "Logistic") {
    fam  <- tryCatch(fit$finalModel$family$family, error = function(e) "binomial")
    link <- tryCatch(fit$finalModel$family$link,   error = function(e) "logit")
    tibble::tibble(Model = model_name, Parameter = c("family", "link"), Value = c(fam, link))
  } else tibble::tibble()
}

extract_lgb_params2 <- function(best_params, best_iter){
  keys <- c("learning_rate","num_leaves","min_data_in_leaf",
            "feature_fraction","bagging_fraction","bagging_freq",
            "lambda_l1","lambda_l2")
  vals <- sapply(keys, function(k) if (is.null(best_params[[k]])) NA else best_params[[k]])
  tibble::tibble(Model = "LightGBM", Parameter = c(keys, "nrounds"),
                 Value = as.character(c(vals, best_iter)))
}

order_models <- c("Logistic","SVM","GBM","Xgboost","KNN","LightGBM")
params_long <- dplyr::bind_rows(lapply(order_models[1:5], function(m){
  extract_caret_params2(ML_calss_model[[m]], m)
}), extract_lgb_params2(best_params, best_iter))

if (length(thr_map)) {
  thr_tbl <- dplyr::bind_rows(lapply(names(thr_map), function(m){
    tibble::tibble(Model = m, Parameter = "Train_threshold", Value = sprintf("%.6f", thr_map[[m]]))
  }))
  params_long <- dplyr::bind_rows(params_long, thr_tbl)
}

is_num <- suppressWarnings(!is.na(as.numeric(params_long$Value)))
params_long$Value[is_num] <- format(round(as.numeric(params_long$Value[is_num]), 6), trim = TRUE)
params_long <- params_long |> dplyr::arrange(factor(Model, levels = order_models), Parameter)
params_wide <- tryCatch(tidyr::pivot_wider(params_long, names_from = Parameter, values_from = Value),
                        error = function(e) params_long)

write.csv(params_long, safe_path("Best_Params_Long.csv"), row.names = FALSE)
write.csv(params_wide, safe_path("Best_Params_Wide.csv"), row.names = FALSE)

cat("\nDone. Key outputs saved under 'results/modeling/'.\n")
