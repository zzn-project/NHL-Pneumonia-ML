suppressPackageStartupMessages({
  library(glmnet)        # LASSO (cv.glmnet)
  library(caret)         # CV / RFE wrappers
  library(randomForest)  # Base learner for RFE
  library(pROC)          # ROC / AUC
  library(dplyr)         # Data wrangling
  library(ggplot2)       # Plotting
  library(openxlsx)      # Export tables
  library(corrplot)      # Correlation heatmap
  library(car)           # VIF
  library(purrr)         # Functional tools
})

set.seed(123)
options(stringsAsFactors = FALSE)

# --------------------
# I/O (relative paths for reproducibility)
# --------------------
in_csv  <- file.path("results", "train.csv")                 # produced by 01_preprocess.R
out_dir <- file.path("results", "feature_selection")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --------------------
# Load data
# --------------------
data <- read.csv(in_csv, stringsAsFactors = FALSE, check.names = FALSE)

# Ensure binary outcome with consistent levels: negative (0) / positive (1)
normalize_outcome_np <- function(y) {
  if (is.numeric(y)) return(factor(ifelse(y == 1, "positive", "negative"), levels = c("negative","positive")))
  if (is.logical(y)) return(factor(ifelse(y, "positive", "negative"), levels = c("negative","positive")))
  yc <- trimws(as.character(y))
  if (all(yc %in% c("0","1"))) return(factor(ifelse(yc == "1", "positive", "negative"), levels = c("negative","positive")))
  f <- factor(yc)
  if (length(levels(f)) != 2) stop("Outcome must be binary.")
  tab <- table(f)
  pos <- names(which.min(tab))                               # rarer class as positive (conservative default)
  f <- relevel(f, ref = setdiff(levels(f), pos))
  levels(f) <- c("negative","positive")
  f
}
data$infection <- normalize_outcome_np(data$infection)

n_events <- sum(data$infection == "positive", na.rm = TRUE)
n_total  <- nrow(data)
cat(sprintf("N = %d; events (positive) = %d (%.1f%%)\n", n_total, n_events, 100 * n_events / n_total))

# Design matrix for glmnet
x <- model.matrix(infection ~ ., data = data)[, -1, drop = FALSE]
y <- ifelse(data$infection == "positive", 1, 0)

# --------------------
# 1) LASSO with 10-fold CV
# --------------------
cv_model <- cv.glmnet(x, y, family = "binomial", alpha = 1, nfolds = 10)

pdf(file.path(out_dir, "lasso_cv_curve.pdf"), width = 6.5, height = 6.5)
plot(cv_model); title("CV deviance vs log(lambda)")
dev.off()

fit_path <- glmnet(x, y, family = "binomial", alpha = 1, standardize = TRUE)

make_coef_path <- function(fmt = c("pdf","png")) {
  fmt <- match.arg(fmt)
  if (fmt == "pdf") {
    pdf(file.path(out_dir, "lasso_coef_path.pdf"), width = 6.5, height = 6.5)
  } else {
    png(file.path(out_dir, "lasso_coef_path.png"), width = 2000, height = 2000, res = 300)
  }
  plot(fit_path, xvar = "lambda", label = FALSE, lwd = 2.8, col = "grey30")
  abline(v = log(cv_model$lambda.min), lty = 2, lwd = 2.8, col = "#D55E00")  # lambda.min
  abline(v = log(cv_model$lambda.1se), lty = 2, lwd = 2.8, col = "#0072B2")  # lambda.1se
  legend("topright", bty = "n",
         legend = c(expression(lambda[min]), expression(lambda[1*se])),
         lwd = 2.8, lty = 2, col = c("#D55E00", "#0072B2"))
  title("LASSO coefficient paths")
  dev.off()
}
make_coef_path("pdf"); make_coef_path("png")

lambda_to_use <- "lambda.1se"                               # parsimonious choice
lasso_coef <- coef(cv_model, s = lambda_to_use)
lasso_coef_matrix <- as.matrix(lasso_coef)
lasso_features <- setdiff(rownames(lasso_coef_matrix)[lasso_coef_matrix[,1] != 0], "(Intercept)")

cat("LASSO selected =", length(lasso_features), "features\n")
if (length(lasso_features)) cat("LASSO features: ", paste(lasso_features, collapse = ", "), "\n")

openxlsx::write.xlsx(data.frame(LASSO_Selected = lasso_features),
                     file.path(out_dir, "LASSO_selected_vars.xlsx"), rowNames = FALSE)

# --------------------
# 2) caret RFE (candidate set = LASSO features; EPV ≥ 5; max 10)
# --------------------
custom_summary <- function(data, lev = NULL, model = NULL) {
  pos <- if (!is.null(lev)) lev[2] else "positive"
  if (!pos %in% colnames(data)) pos <- tail(colnames(data), 1)
  roc_obj <- pROC::roc(response = data$obs, predictor = data[[pos]],
                       levels = rev(levels(data$obs)), quiet = TRUE)
  c(AUC = as.numeric(pROC::auc(roc_obj)))
}
custom_rfFuncs <- rfFuncs
custom_rfFuncs$summary <- custom_summary
rfe_ctrl <- rfeControl(functions = custom_rfFuncs, method = "cv",
                       number = 5, verbose = TRUE, returnResamp = "final")

if (length(lasso_features) < 2) stop("Not enough LASSO-retained features for RFE.")
max_vars_allowed <- max(2, min(10, floor(n_events / 5)))
upper_main <- min(max_vars_allowed, length(lasso_features))
if (upper_main < 2) stop("Insufficient features after EPV constraint for RFE.")
sizes_seq <- 2:upper_main

df_rfe <- data[, c(lasso_features, "infection")]
df_rfe <- df_rfe[complete.cases(df_rfe), , drop = FALSE]
if (length(unique(df_rfe$infection)) < 2) stop("Only one outcome class after cleaning; cannot compute ROC.")

X_rfe <- df_rfe %>% dplyr::select(all_of(lasso_features))
y_rfe <- df_rfe$infection

set.seed(123)
rfe_result <- rfe(x = X_rfe, y = y_rfe, sizes = sizes_seq,
                  rfeControl = rfe_ctrl, metric = "AUC")

# RFE performance curve
rfe_df <- rfe_result$results %>% dplyr::select(Variables, AUC)
vars_each_step <- rfe_result$variables %>%
  dplyr::group_by(Variables) %>%
  dplyr::summarise(Feature = dplyr::last(var), .groups = "drop")
rfe_df <- dplyr::left_join(rfe_df, vars_each_step, by = "Variables")
rfe_df$Step <- rfe_df$Variables
best_point <- rfe_df[which.max(rfe_df$AUC), , drop = FALSE]

p_auc_rfe <- ggplot(rfe_df, aes(x = Step, y = AUC)) +
  geom_line(size = 1.0) +
  geom_point(size = 1.6) +
  geom_point(data = best_point, aes(x = Step, y = AUC), color = "red", size = 3) +
  geom_text(data = best_point,
            aes(label = paste0("Best: ", Step, " vars\nAUC=", sprintf("%.3f", AUC))),
            vjust = -1.0, hjust = 0.5, size = 3.5, color = "red") +
  theme_classic(base_size = 12) +
  labs(title = "RF-RFE after LASSO: AUC vs number of features",
       x = "Number of features", y = "Cross-validated AUC")

ggsave(filename = file.path(out_dir, "RFE_AUC_curve_LASSO.pdf"),
       plot = p_auc_rfe, width = 18, height = 14, units = "cm", dpi = 300)
openxlsx::write.xlsx(rfe_df, file.path(out_dir, "RFE_AUC_curve_data.xlsx"), rowNames = FALSE)

# Final feature set from RFE
selected_vars <- caret::predictors(rfe_result)
cat("RFE final features =", length(selected_vars), "\n")
if (length(selected_vars)) cat("RFE features: ", paste(selected_vars, collapse = ", "), "\n")
openxlsx::write.xlsx(data.frame(RFE_Final_Selected = selected_vars),
                     file.path(out_dir, "RFE_final_selected_vars.xlsx"), rowNames = FALSE)

# --------------------
# 3) Collinearity and correlation among selected predictors
# --------------------
if (length(selected_vars) >= 2) {
  X_final <- data %>% dplyr::select(all_of(selected_vars))
  X_mm <- as.data.frame(model.matrix(~ . , data = X_final))[ , -1, drop = FALSE]
  lm_dummy <- lm(stats::rnorm(nrow(X_mm)) ~ ., data = X_mm)   # proxy to compute VIFs
  v <- car::vif(lm_dummy)
  if (is.matrix(v)) {
    vif_tbl <- data.frame(
      Term = rownames(v), GVIF = v[, "GVIF"], Df = v[, "Df"],
      GVIF_adj = v[, "GVIF"]^(1/(2*v[, "Df"])), row.names = NULL
    )
  } else {
    vif_tbl <- data.frame(Term = names(v), VIF = as.numeric(v), row.names = NULL)
  }
  openxlsx::write.xlsx(vif_tbl, file.path(out_dir, "VIF_results.xlsx"), rowNames = FALSE)
  
  cor_mat <- suppressWarnings(cor(X_mm, use = "pairwise.complete.obs", method = "pearson"))
  openxlsx::write.xlsx(as.data.frame(cor_mat),
                       file.path(out_dir, "final_selected_vars_correlation_matrix.xlsx"),
                       rowNames = TRUE)
  pdf(file.path(out_dir, "final_selected_vars_correlation_plot.pdf"), width = 8, height = 7)
  corrplot(cor_mat, method = "color", type = "upper",
           tl.col = "black", tl.srt = 45,
           addCoef.col = "black", number.cex = 0.7,
           mar = c(0,0,1,0))
  title("Correlation among final selected predictors", line = 0.5)
  dev.off()
}

# EPV check
epv <- ifelse(length(selected_vars) > 0, n_events / length(selected_vars), NA)
cat(sprintf("EPV = events / variables = %d / %d = %.2f\n",
            n_events, length(selected_vars), epv))

# --------------------
# 4) Stability analyses (optional but recommended)
# --------------------
nested_cv_feature_stability <- function(dat, max_vars = 10,
                                        outer_folds = 5, repeats = 20,
                                        use_lambda = c("lambda.1se","lambda.min"),
                                        seed = 2025) {
  use_lambda <- match.arg(use_lambda); set.seed(seed)
  yfac <- dat$infection
  all_vars <- setdiff(names(dat), "infection")
  sel_counts <- setNames(numeric(length(all_vars)), all_vars)
  oof_res <- list(); fid <- 0
  for (rep in seq_len(repeats)) {
    folds <- caret::createFolds(yfac, k = outer_folds, returnTrain = TRUE)
    for (k in seq_along(folds)) {
      fid <- fid + 1
      tr_idx <- folds[[k]]; te_idx <- setdiff(seq_len(nrow(dat)), tr_idx)
      tr <- dat[tr_idx, , drop = FALSE]; te <- dat[te_idx, , drop = FALSE]
      
      xtr <- model.matrix(infection ~ ., data = tr)[, -1, drop = FALSE]
      ytr <- ifelse(tr$infection == "positive", 1, 0)
      cv_lasso <- glmnet::cv.glmnet(xtr, ytr, family = "binomial", alpha = 1, nfolds = 5)
      b <- as.matrix(coef(cv_lasso, s = use_lambda))
      feats_lasso <- setdiff(rownames(b)[b[,1] != 0], "(Intercept)")
      if (length(feats_lasso) < 2) next
      
      n_events_tr <- sum(tr$infection == "positive")
      fold_max_vars <- max(2, min(max_vars, floor(n_events_tr / 5), length(feats_lasso)))
      if (length(feats_lasso) > fold_max_vars) {
        ord <- order(abs(b[feats_lasso,1]), decreasing = TRUE)
        feats_lasso <- feats_lasso[ord][1:fold_max_vars]
      }
      if (fold_max_vars < 2) next
      
      sizes_seq <- 2:fold_max_vars
      rfe_res <- tryCatch(
        caret::rfe(x = tr[, feats_lasso, drop = FALSE], y = tr$infection,
                   sizes = sizes_seq, rfeControl = rfe_ctrl, metric = "AUC"),
        error = function(e) NULL
      )
      if (is.null(rfe_res)) next
      feats_final <- caret::predictors(rfe_res)
      if (length(feats_final) < 2) next
      
      mdl <- tryCatch(
        glm(infection ~ ., data = tr[, c(feats_final, "infection")], family = binomial()),
        error = function(e) NULL
      )
      if (is.null(mdl)) next
      prob <- predict(mdl, newdata = te[, feats_final, drop = FALSE], type = "response")
      keep <- is.finite(prob) & !is.na(te$infection)
      if (sum(keep) < 2 || length(unique(te$infection[keep])) < 2) next
      
      roc_obj <- pROC::roc(response = te$infection[keep], predictor = prob[keep],
                           levels = c("negative","positive"), direction = ">")
      auc_val <- as.numeric(pROC::auc(roc_obj))
      
      sel_counts[feats_final] <- sel_counts[feats_final] + 1
      oof_res[[fid]] <- data.frame(rep = rep, fold = k, AUC = auc_val,
                                   n_vars = length(feats_final),
                                   selected = paste(feats_final, collapse = ", "))
    }
  }
  oof_df <- dplyr::bind_rows(oof_res)
  freq <- sel_counts / (outer_folds * repeats)
  list(oof = oof_df, sel_freq = sort(freq[freq > 0], decreasing = TRUE))
}

lasso_boot_stability <- function(dat, B = 200, lambda = c("lambda.1se","lambda.min"), seed = 9) {
  lambda <- match.arg(lambda); set.seed(seed)
  all_vars <- setdiff(names(dat), "infection")
  counts <- setNames(numeric(length(all_vars)), all_vars)
  for (b in 1:B) {
    idx <- sample(seq_len(nrow(dat)), replace = TRUE)
    boot <- dat[idx, , drop = FALSE]
    xb <- model.matrix(infection ~ ., data = boot)[, -1, drop = FALSE]
    yb <- ifelse(boot$infection == "positive", 1, 0)
    cvb <- glmnet::cv.glmnet(xb, yb, family = "binomial", alpha = 1, nfolds = 5)
    bb  <- as.matrix(coef(cvb, s = lambda))
    feats <- setdiff(rownames(bb)[bb[,1] != 0], "(Intercept)")
    counts[feats] <- counts[feats] + 1
  }
  sort(counts / B, decreasing = TRUE)
}

nest <- nested_cv_feature_stability(
  dat = data, max_vars = 10, outer_folds = 5, repeats = 20,
  use_lambda = "lambda.1se", seed = 7
)
if (!is.null(nest$oof) && nrow(nest$oof) > 0) {
  openxlsx::write.xlsx(nest$oof, file.path(out_dir, "NestedCV_OOF_AUCs.xlsx"), rowNames = FALSE)
}
if (length(nest$sel_freq) > 0) {
  openxlsx::write.xlsx(
    data.frame(Feature = names(nest$sel_freq), Selection_Frequency = as.numeric(nest$sel_freq)),
    file.path(out_dir, "NestedCV_Selection_Frequency.xlsx"), rowNames = FALSE
  )
  freq_df <- data.frame(Feature = names(nest$sel_freq), Freq = as.numeric(nest$sel_freq))
  p_freq <- ggplot(freq_df, aes(x = reorder(Feature, Freq), y = Freq)) +
    geom_col() + coord_flip() + ylim(0, 1) +
    labs(title = "Selection frequency across nested CV", x = "Feature", y = "Frequency") +
    theme_classic(base_size = 12)
  ggsave(file.path(out_dir, "Selection_Frequency_NestedCV.pdf"),
         p_freq, width = 18, height = 12, units = "cm", dpi = 300)
}

freq_lasso_only <- lasso_boot_stability(data, B = 200, lambda = "lambda.1se", seed = 11)
if (length(freq_lasso_only) > 0) {
  openxlsx::write.xlsx(
    data.frame(Feature = names(freq_lasso_only), Selection_Frequency = as.numeric(freq_lasso_only)),
    file.path(out_dir, "LASSO_Stability_Bootstrap.xlsx"), rowNames = FALSE
  )
}

cat("\nDone.\n",
    "- LASSO CV curve: lasso_cv_curve.pdf\n",
    "- LASSO path: lasso_coef_path.pdf/.png\n",
    "- LASSO features: LASSO_selected_vars.xlsx\n",
    "- RFE curve & data: RFE_AUC_curve_LASSO.pdf / RFE_AUC_curve_data.xlsx\n",
    "- RFE final features: RFE_final_selected_vars.xlsx\n",
    "- Correlation/VIF: final_selected_vars_correlation_matrix.xlsx / final_selected_vars_correlation_plot.pdf / VIF_results.xlsx\n",
    "- Nested CV OOF AUCs & selection freq: NestedCV_OOF_AUCs.xlsx / NestedCV_Selection_Frequency.xlsx / Selection_Frequency_NestedCV.pdf\n",
    "- LASSO bootstrap stability: LASSO_Stability_Bootstrap.xlsx\n", sep = "")
