suppressPackageStartupMessages({
  library(caret)      # Stratified sampling / cross-validation
  library(tidyverse)  # Data wrangling and visualization
  library(openxlsx)   # Export Excel tables and figures
  library(ggplot2)    # Plotting
  library(pROC)       # ROC curve, AUC, Youden's J
  library(recipes)    # KNN imputation
  library(kknn)       # Dependency for step_impute_knn
})

set.seed(52)
options(stringsAsFactors = FALSE)

# File paths (use relative paths for reproducibility)
in_path  <- "data/synthetic/nhl_synth.csv"
out_dir  <- "results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Load dataset
data <- read.csv(in_path, header = TRUE, encoding = "UTF-8", check.names = FALSE)

# Normalize outcome variable to No/Yes
normalize_outcome <- function(y){
  y_chr <- trimws(as.character(y))
  pos <- c("1","Yes","Y","是","陽性","阳性","Positive","Pos","true","TRUE")
  factor(ifelse(y_chr %in% pos, "Yes", "No"), levels = c("No","Yes"))
}
data$infection <- normalize_outcome(data$infection)

# Summarize missing values
missing_summary <- data.frame(
  Variable   = names(data),
  MissingN   = sapply(data, function(x) sum(is.na(x))),
  MissingPct = sapply(data, function(x) mean(is.na(x)) * 100)
)
openxlsx::write.xlsx(missing_summary,
                     file.path(out_dir, "missing_summary.xlsx"),
                     rowNames = FALSE)

# Stratified split (70/30)
inTrain   <- createDataPartition(y = data$infection, p = 0.7, list = FALSE)
train_raw <- data[inTrain, ]
test_raw  <- data[-inTrain, ]

# Standardize predictors to 0/1
normalize_to_01 <- function(x){
  if (is.factor(x)) x <- as.character(x)
  x <- trimws(as.character(x))
  x[x %in% c("1","Yes","Y","是","陽性","阳性","Positive","Pos","true","TRUE")] <- "1"
  x[x %in% c("0","No","N","否","陰性","阴性","Negative","Neg","false","FALSE")] <- "0"
  suppressWarnings(as.numeric(x))
}
prep_binary_predictors <- function(df){
  df2 <- df
  pred_cols <- setdiff(names(df2), "infection")
  df2[pred_cols] <- lapply(df2[pred_cols], normalize_to_01)
  df2$infection <- df$infection
  df2
}
train_bin <- prep_binary_predictors(train_raw)
test_bin  <- prep_binary_predictors(test_raw)

# Re-binarize predictors (convert KNN-imputed fractional values back to 0/1)
rebin_01 <- function(df){
  pred_cols <- setdiff(names(df), "infection")
  df[pred_cols] <- lapply(df[pred_cols], function(x) as.integer(as.numeric(x) >= 0.5))
  df
}

# Plot missingness and outcome distribution
missing_plot <- ggplot(missing_summary, aes(x = reorder(Variable, MissingPct), y = MissingPct)) +
  geom_bar(stat = "identity") + theme_minimal() + coord_flip()
ggsave(file.path(out_dir, "missing_plot.png"), missing_plot, width = 8, height = 6, dpi = 300)

# Baseline model: KNN imputation + Logistic regression (k = 3, 5, 7)
ks <- c(3, 5, 7)
ctrl <- trainControl(
  method = "repeatedcv", number = 10, repeats = 3,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final"
)
results_list <- list()

for (kk in ks) {
  rec_knn <- recipe(infection ~ ., data = train_bin) %>%
    step_impute_knn(all_predictors(), neighbors = kk) %>%
    step_zv(all_predictors())
  rec_prep <- prep(rec_knn, training = train_bin, retain = TRUE)
  train_imp <- rebin_01(bake(rec_prep, new_data = train_bin))
  test_imp  <- rebin_01(bake(rec_prep, new_data = test_bin))
  
  fit_glm <- train(
    infection ~ ., data = train_imp,
    method = "glm", family = binomial(),
    trControl = ctrl, metric = "ROC"
  )
  train_cv_auc <- max(fit_glm$results$ROC, na.rm = TRUE)
  
  train_prob <- predict(fit_glm, newdata = train_imp, type = "prob")[, "Yes"]
  test_prob  <- predict(fit_glm, newdata = test_imp,  type = "prob")[, "Yes"]
  
  roc_train <- pROC::roc(train_imp$infection, train_prob, levels = c("No","Yes"), quiet = TRUE)
  thr <- as.numeric(pROC::coords(roc_train, x = "best", best.method = "youden", ret = "threshold"))
  roc_test <- pROC::roc(test_imp$infection, test_prob, levels = c("No","Yes"), quiet = TRUE)
  
  test_pred_cls <- factor(ifelse(test_prob >= thr, "Yes", "No"), levels = c("No","Yes"))
  cm <- confusionMatrix(test_pred_cls, test_imp$infection, positive = "Yes")
  
  results_list[[as.character(kk)]] <- data.frame(
    KNN_k = kk,
    TrainCV_AUC = round(train_cv_auc, 3),
    Test_AUC = round(as.numeric(pROC::auc(roc_test)), 3),
    Test_Accuracy = round(cm$overall["Accuracy"], 3),
    Test_Sensitivity = round(cm$byClass["Sensitivity"], 3),
    Test_Specificity = round(cm$byClass["Specificity"], 3),
    Threshold_Youden_from_Train = round(thr, 6)
  )
  
  # Export imputed train/test sets
  write.csv(train_imp, file.path(out_dir, sprintf("train_k%d.csv", kk)), row.names = FALSE)
  write.csv(test_imp,  file.path(out_dir, sprintf("test_k%d.csv",  kk)), row.names = FALSE)
}

# Save sensitivity analysis results
knn_k_results <- bind_rows(results_list)
write.csv(knn_k_results, file.path(out_dir, "knn_k_results.csv"), row.names = FALSE)

# Use k = 5 as the primary dataset for downstream modeling
PRIMARY_K <- 5
file.copy(file.path(out_dir, sprintf("train_k%d.csv", PRIMARY_K)), file.path(out_dir, "train.csv"), overwrite = TRUE)
file.copy(file.path(out_dir, sprintf("test_k%d.csv",  PRIMARY_K)), file.path(out_dir, "test.csv"),  overwrite = TRUE)

message("Preprocessing completed. Results saved to 'results/'")
