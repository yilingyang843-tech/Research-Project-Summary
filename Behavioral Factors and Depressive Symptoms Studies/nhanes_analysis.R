PROJECT_DIR <- "nhanes_depression_r"
DATA_DIR <- "nhanes_depression_r/data"
OUTPUT_DIR <- "nhanes_depression_r/results"

if (!dir.exists(PROJECT_DIR)) stop("找不到项目文件夹：", PROJECT_DIR)
dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
SEED <- 3700L
DOWNLOAD_MISSING <- TRUE  
INSTALL_MISSING <- TRUE
N_TREES <- 500L
XGB_MAX_ROUNDS <- 800L
XGB_EARLY_STOP <- 40L
N_THREADS <- 2L
SAVE_RESULTS_TO_FILES <- FALSE


FIRST_DRINK_CSV <- NULL

setup_packages <- function() {
  local_lib <- file.path(PROJECT_DIR, ".r-library")
  dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(local_lib, .libPaths()))
  packages <- c("haven", "randomForest", "xgboost", "survey")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) && INSTALL_MISSING) {
    dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
    install.packages(missing, lib = local_lib, repos = "https://cloud.r-project.org")
    .libPaths(c(local_lib, .libPaths()))
    missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  }
  if (length(missing)) stop("请先安装缺少的 R 包：install.packages(c(",
                            paste(sprintf('"%s"', missing), collapse = ", "), "))")
}

write_csv <- function(x, name) {
  large_data <- c("analysis_data.csv", "data_split.csv", "test_predictions.csv",
                  "simple_regression_prediction_intervals.csv")
  if (!name %in% large_data) {
    cat("\n==================== ", name, " ====================\n", sep = "")
    print(x, row.names = FALSE)
  }
  if (SAVE_RESULTS_TO_FILES) {
    write.csv(x, file.path(OUTPUT_DIR, name), row.names = FALSE, fileEncoding = "UTF-8")
  }
  invisible(x)
}
save_text <- function(x, name) {
  cat("\n==================== ", name, " ====================\n", sep = "")
  print(x)
  if (SAVE_RESULTS_TO_FILES) {
    writeLines(capture.output(print(x)), file.path(OUTPUT_DIR, name), useBytes = TRUE)
  }
  invisible(x)
}
valid_values <- function(x, allowed) {
  x <- as.numeric(x)
  x[!x %in% allowed] <- NA_real_
  x
}
read_nhanes <- function(name) {
  path <- file.path(DATA_DIR, paste0(name, ".XPT"))
  if (!file.exists(path)) {
    matches <- list.files(DATA_DIR, full.names = TRUE)
    matches <- matches[toupper(basename(matches)) == paste0(name, ".XPT")]
    if (length(matches)) path <- matches[1]
  }
  if (!file.exists(path)) {
    if (!DOWNLOAD_MISSING) stop("缺少数据文件：", path)
    url <- paste0("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/",
                  name, ".xpt")
    download.file(url, path, mode = "wb", quiet = FALSE)
  }
  d <- as.data.frame(haven::read_xpt(path))
  names(d) <- toupper(names(d))
  if (!"SEQN" %in% names(d) || anyDuplicated(d$SEQN)) stop(name, " 的 SEQN 有误。")
  d
}


prepare_data <- function() {
  demo <- read_nhanes("DEMO_J")
  dpq <- read_nhanes("DPQ_J")
  alq <- read_nhanes("ALQ_J")
  smq <- read_nhanes("SMQ_J")
  demo_vars <- c("SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA")
  qvars <- sprintf("DPQ%03d", seq(10, 90, 10))
  d <- Reduce(function(x, y) merge(x, y, by = "SEQN", all.x = TRUE, sort = FALSE),
              list(demo[, demo_vars], dpq[, c("SEQN", qvars)],
                   alq[, c("SEQN", "ALQ111", "ALQ121", "ALQ130")],
                   smq[, c("SEQN", "SMQ020", "SMD030", "SMQ040")]))
  stopifnot(nrow(d) == nrow(demo), !anyDuplicated(d$SEQN))
  for (v in qvars) d[[v]] <- valid_values(d[[v]], 0:3)
  d$phq9 <- rowSums(d[, qvars], na.rm = FALSE)
  d$age <- as.numeric(d$RIDAGEYR) 
  d$female <- valid_values(d$RIAGENDR, 1:2) - 1
  d$income_pir <- as.numeric(d$INDFMPIR)
  d$income_pir[d$income_pir < 0 | d$income_pir > 5] <- NA_real_
  
  d$ever_drink <- valid_values(d$ALQ111, 1:2)
  d$drink_frequency <- valid_values(d$ALQ121, 0:10)
  d$drinks_on_drinking_day <- valid_values(d$ALQ130, c(1:13, 15))
  
  alcohol <- rep(NA_character_, nrow(d))
  alcohol[d$ever_drink %in% 2] <- "Never_drank"
  alcohol[d$ever_drink %in% 1 & d$drink_frequency %in% 0] <- "None_past_year"
  alcohol[d$ever_drink %in% 1 & d$drink_frequency %in% 1:5] <- "Weekly_or_more"
  alcohol[d$ever_drink %in% 1 & d$drink_frequency %in% 6:7] <- "Monthly"
  alcohol[d$ever_drink %in% 1 & d$drink_frequency %in% 8:10] <- "Less_than_monthly"
  d$alcohol_group <- factor(alcohol, levels = c("Never_drank", "None_past_year",
                                                "Less_than_monthly", "Monthly", "Weekly_or_more"))
  
  d$age_start_smoking <- valid_values(d$SMD030, 1:79)
  bad_age <- !is.na(d$age_start_smoking) & d$age_start_smoking > d$age
  d$age_start_smoking[bad_age] <- NA_real_
  sg <- rep(NA_character_, nrow(d))
  sg[d$SMQ020 %in% 2] <- "Under_100_lifetime"
  sg[d$SMQ020 %in% 1 & d$SMD030 %in% 0] <- "Never_regular"
  a <- d$age_start_smoking
  regular <- d$SMQ020 %in% 1 & !is.na(a)
  sg[regular & a <= 14] <- "Start_le14"
  sg[regular & a >= 15 & a <= 17] <- "Start_15_17"
  sg[regular & a >= 18 & a <= 20] <- "Start_18_20"
  sg[regular & a >= 21] <- "Start_ge21"
  d$smoking_group <- factor(sg, levels = c("Under_100_lifetime", "Never_regular",
                                           "Start_le14", "Start_15_17", "Start_18_20", "Start_ge21"))
  d$age_first_drink <- NA_real_
  if (!is.null(FIRST_DRINK_CSV)) {
    extra <- read.csv(FIRST_DRINK_CSV)
    stopifnot(all(c("SEQN", "age_first_drink") %in% names(extra)),
              !anyDuplicated(extra$SEQN), all(extra$SEQN %in% d$SEQN))
    d$age_first_drink <- as.numeric(extra$age_first_drink[match(d$SEQN, extra$SEQN)])
    invalid <- !is.na(d$age_first_drink) &
      (d$age_first_drink < 1 | d$age_first_drink > d$age)
    if (any(invalid)) stop("补充的首次饮酒年龄包含不合理值，请核实原始编码。")
  }
  flow <- data.frame(stage = c("DEMO records", "Age >= 18", "Complete PHQ-9 + age + sex"),
                     n = c(nrow(d), sum(d$age >= 18, na.rm = TRUE),
                           sum(d$age >= 18 & complete.cases(d[, c("phq9", "age", "female")]), na.rm = TRUE)))
  write_csv(flow, "sample_flow.csv")
  write_csv(data.frame(variable = c("age_first_drink", "age_start_smoking", "phq9"),
                       source = c(if (is.null(FIRST_DRINK_CSV)) "Unavailable in ALQ_J" else FIRST_DRINK_CSV,
                                  "SMD030: age started smoking regularly", "DPQ010-DPQ090: all 9 items required")),
            "variable_audit.csv")
  d
}

correlation_result <- function(d, xname, conf = 0.99) {
  z <- d[complete.cases(d[, c(xname, "phq9")]), c(xname, "phq9")]
  if (nrow(z) < 4 || sd(z[[xname]]) == 0 || sd(z$phq9) == 0) return(NULL)
  t <- cor.test(z[[xname]], z$phq9, method = "pearson", conf.level = conf)
  # cor.test() 的 Pearson 相关区间使用 Fisher z 变换。
  data.frame(variable = xname, n = nrow(z), r = unname(t$estimate),
             lower = t$conf.int[1], upper = t$conf.int[2], p_value = t$p.value, confidence = conf)
}
exploratory_analysis <- function(d) {
  
  cat("\n==================== EXPLORATORY STATISTICAL TESTS ====================\n")
  {
    cat("Exploratory, UNWEIGHTED analyses; confidence level = 99%\n")
    cat("Sample size:", nrow(d), "\n")
    print(summary(d$phq9))
    set.seed(SEED)
    print(shapiro.test(sample(d$phq9, min(5000L, nrow(d)))))
    
    print(t.test(d$phq9, mu = 0, conf.level = .99))
    cat("\nWelch t-test: ever had alcohol vs never (NOT age at first drink)\n")
    w <- d[!is.na(d$ever_drink), ]
    if (all(table(w$ever_drink) >= 2) && length(unique(w$ever_drink)) == 2)
      print(t.test(phq9 ~ ever_drink, data = w, conf.level = .99))
    s <- droplevels(d[!is.na(d$age_start_smoking) & d$SMQ020 %in% 1, ])
    cat("\nANOVA/Tukey: regular-smoking onset age groups only\n")
    fit <- aov(phq9 ~ smoking_group, data = s)
    print(summary(fit)); print(TukeyHSD(fit, conf.level = .99))
    cat("\nWelch ANOVA / Kruskal-Wallis sensitivity checks\n")
    print(oneway.test(phq9 ~ smoking_group, data = s))
    print(kruskal.test(phq9 ~ smoking_group, data = s))
    simple <- lm(phq9 ~ age_start_smoking, data = s)
    cat("\nSimple linear regression: smoking onset age\n")
    print(summary(simple)); print(anova(simple))
    pi_data <- data.frame(age_start_smoking = sort(unique(s$age_start_smoking)))
    write_csv(cbind(pi_data, predict(simple, pi_data, interval = "prediction", level = .99)),
              "simple_regression_prediction_intervals.csv")
    if (!is.null(FIRST_DRINK_CSV)) {
      e <- d[!is.na(d$age_first_drink), ]
      e$early_drink <- factor(ifelse(e$age_first_drink <= 16, "le16", "gt16"))
      if (all(table(e$early_drink) >= 2) && nlevels(e$early_drink) == 2)
        print(t.test(phq9 ~ early_drink, data = e, conf.level = .99))
      if (nrow(e) >= 4 && length(unique(e$age_first_drink)) > 1) {
        drink_lm <- lm(phq9 ~ age_first_drink, data = e)
        print(summary(drink_lm)); print(anova(drink_lm))
      }
    } else cat("\nFirst-drink-age analyses skipped: variable absent from ALQ_J.\n")
  }
  cor_rows <- lapply(c("age_start_smoking", "age_first_drink", "income_pir", "age"),
                     function(v) correlation_result(d, v))
  cors <- do.call(rbind, cor_rows)
  cors$p_holm <- p.adjust(cors$p_value, method = "holm")
  write_csv(cors, "correlations_99CI.csv")
  if (SAVE_RESULTS_TO_FILES) {
    png(file.path(OUTPUT_DIR, "phq9_distribution.png"), width = 1200, height = 700, res = 120)
  }
  hist(d$phq9, breaks = seq(-.5, 27.5, 1), main = "PHQ-9 score distribution",
       xlab = "PHQ-9 total score", col = "steelblue", border = "white")
  if (SAVE_RESULTS_TO_FILES) dev.off()
  if (SAVE_RESULTS_TO_FILES) {
    png(file.path(OUTPUT_DIR, "smoking_onset_boxplot.png"), width = 1300, height = 750, res = 120)
  }
  s <- droplevels(d[!is.na(d$age_start_smoking) & d$SMQ020 %in% 1, ])
  boxplot(phq9 ~ smoking_group, data = s, xlab = "Age started smoking regularly",
          ylab = "PHQ-9", main = "Regular smokers with valid onset age", col = "lightblue")
  if (SAVE_RESULTS_TO_FILES) dev.off()
}

survey_analysis <- function(full_data) {
  eligible <- with(full_data, is.finite(WTMEC2YR) & WTMEC2YR > 0 &
                     !is.na(SDMVPSU) & !is.na(SDMVSTRA))
  x <- full_data[eligible, ]
  f <- phq9 ~ age + female + income_pir + alcohol_group + smoking_group
  x$analysis_domain <- x$age >= 18 & complete.cases(x[, all.vars(f)])
  design <- survey::svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA,
                              weights = ~WTMEC2YR, nest = TRUE, data = x)
  domain <- subset(design, analysis_domain)
  fit <- survey::svyglm(f, design = domain)
  co <- as.data.frame(coef(summary(fit)))
  ci <- confint(fit, level = .95)
  write_csv(data.frame(term = rownames(co), co, CI_lower = ci[, 1], CI_upper = ci[, 2],
                       row.names = NULL, check.names = FALSE), "survey_weighted_regression.csv")
  cat("\n==================== SURVEY-WEIGHTED MULTIPLE REGRESSION ====================\n")
  {
    cat("Survey-weighted complete-case association model; no causal interpretation.\n")
    cat("Complete-case domain size:", sum(x$analysis_domain), "\n")
    print(summary(fit))
    # 均值使用所有 PHQ-9 有效成人，不额外限制协变量完整性。
    mean_domain <- subset(design, age >= 18 & !is.na(phq9))
    m <- survey::svymean(~phq9, mean_domain)
    print(m); print(confint(m))
  }
  if (SAVE_RESULTS_TO_FILES) {
    saveRDS(fit, file.path(OUTPUT_DIR, "survey_model.rds"))
  }
}

FEATURES <- c("age", "female", "income_pir", "alcohol_group", "smoking_group")

stratified_take <- function(y, proportion, seed) {
  set.seed(seed)
  bands <- cut(y, breaks = c(-Inf, 4, 9, 14, 19, Inf), labels = FALSE)
  unlist(lapply(split(seq_along(y), bands), function(idx) {
    n <- floor(length(idx) * proportion)
    if (n == 0) integer(0) else idx[sample.int(length(idx), n)]
  }), use.names = FALSE)
}
fit_preprocessor <- function(d) {
  medians <- vapply(d[, c("age", "female", "income_pir")], median, numeric(1), na.rm = TRUE)
  if (any(!is.finite(medians))) stop("训练集中存在完全缺失的数值变量。")
  cats <- lapply(d[, c("alcohol_group", "smoking_group")], function(v) {
    sort(unique(as.character(v[!is.na(v)])))
  })
  list(medians = medians, cats = cats)
}
transform_features <- function(d, prep) {
  out <- as.data.frame(d[, c("age", "female", "income_pir")])
  for (v in names(prep$medians)) {
    out[[paste0(v, "_missing")]] <- as.numeric(is.na(out[[v]]))
    out[[v]][is.na(out[[v]])] <- prep$medians[[v]]
  }
  for (v in names(prep$cats)) {
    z <- as.character(d[[v]])
    for (lev in prep$cats[[v]]) out[[paste(v, lev, sep = "__")]] <- as.numeric(!is.na(z) & z == lev)
    out[[paste0(v, "__Unknown")]] <- as.numeric(is.na(z) | !z %in% prep$cats[[v]])
  }
  m <- as.matrix(out); storage.mode(m) <- "double"
  stopifnot(!anyNA(m), all(is.finite(m)))
  m
}
independent_columns <- function(x) {
  x1 <- cbind(Intercept = 1, x)
  qr1 <- qr(x1, tol = 1e-8)
  idx <- qr1$pivot[seq_len(qr1$rank)]
  colnames(x1)[idx[idx != 1]]
}
metrics <- function(y, pred) {
  stopifnot(length(y) == length(pred), all(is.finite(pred)))
  c(RMSE = sqrt(mean((y - pred)^2)), MAE = mean(abs(y - pred)),
    R2 = 1 - sum((y - pred)^2) / sum((y - mean(y))^2))
}
fit_lm <- function(x, y) lm(phq9 ~ ., data = data.frame(phq9 = y, x, check.names = FALSE))

predictive_models <- function(d) {
  trainval_idx <- stratified_take(d$phq9, .8, SEED)
  test_idx <- setdiff(seq_len(nrow(d)), trainval_idx)
  train_local <- stratified_take(d$phq9[trainval_idx], .75, SEED + 1L)
  train_idx <- trainval_idx[train_local]
  val_idx <- setdiff(trainval_idx, train_idx)
  stopifnot(length(intersect(train_idx, val_idx)) == 0,
            length(intersect(trainval_idx, test_idx)) == 0,
            length(c(train_idx, val_idx, test_idx)) == nrow(d))
  write_csv(data.frame(SEQN = d$SEQN, split = ifelse(seq_len(nrow(d)) %in% train_idx,
                                                     "train", ifelse(seq_len(nrow(d)) %in% val_idx, "validation", "test"))), "data_split.csv")
  tr <- d[train_idx, ]; va <- d[val_idx, ]; te <- d[test_idx, ]
  prep <- fit_preprocessor(tr)
  xtr <- transform_features(tr, prep); xva <- transform_features(va, prep)
  lm_cols <- independent_columns(xtr)
  linear <- fit_lm(xtr[, lm_cols, drop = FALSE], tr$phq9)
  lm_val <- predict(linear, data.frame(xva[, lm_cols, drop = FALSE]))
  
  grid_rf <- expand.grid(mtry = unique(pmin(ncol(xtr), c(2L, 4L, 6L))), nodesize = c(5L, 15L))
  grid_rf$RMSE <- NA_real_
  for (i in seq_len(nrow(grid_rf))) {
    set.seed(SEED)
    f <- randomForest::randomForest(x = xtr, y = tr$phq9, ntree = N_TREES,
                                    mtry = grid_rf$mtry[i], nodesize = grid_rf$nodesize[i])
    grid_rf$RMSE[i] <- metrics(va$phq9, predict(f, xva))["RMSE"]
  }
  best_rf <- grid_rf[which.min(grid_rf$RMSE), ]
  write_csv(grid_rf, "rf_validation_search.csv")
  
  dtr <- xgboost::xgb.DMatrix(xtr, label = tr$phq9)
  dva <- xgboost::xgb.DMatrix(xva, label = va$phq9)
  base_params <- list(objective = "reg:squarederror", eval_metric = "rmse",
                      subsample = .8, colsample_bytree = .8, min_child_weight = 5,
                      nthread = N_THREADS, tree_method = "hist")
  grid_xgb <- expand.grid(max_depth = c(2L, 3L, 5L), eta = c(.03, .1))
  grid_xgb$best_iteration <- NA_integer_; grid_xgb$RMSE <- NA_real_
  for (i in seq_len(nrow(grid_xgb))) {
    set.seed(SEED)
    pars <- c(base_params, list(max_depth = grid_xgb$max_depth[i], eta = grid_xgb$eta[i]))
    f <- xgboost::xgb.train(params = pars, data = dtr, nrounds = XGB_MAX_ROUNDS,
                            watchlist = list(validation = dva), early_stopping_rounds = XGB_EARLY_STOP,
                            verbose = 0)
    grid_xgb$best_iteration[i] <- f$best_iteration
    grid_xgb$RMSE[i] <- min(f$evaluation_log$validation_rmse)
  }
  best_xgb <- grid_xgb[which.min(grid_xgb$RMSE), ]
  write_csv(grid_xgb, "xgboost_validation_search.csv")
  write_csv(data.frame(model = c("Linear regression", "Random Forest", "XGBoost"),
                       validation_RMSE = c(metrics(va$phq9, lm_val)["RMSE"], best_rf$RMSE, best_xgb$RMSE)),
            "validation_comparison.csv")
  
  tv <- d[trainval_idx, ]
  final_prep <- fit_preprocessor(tv)
  xtv <- transform_features(tv, final_prep); xte <- transform_features(te, final_prep)
  final_cols <- independent_columns(xtv)
  final_lm <- fit_lm(xtv[, final_cols, drop = FALSE], tv$phq9)
  set.seed(SEED)
  final_rf <- randomForest::randomForest(x = xtv, y = tv$phq9, ntree = N_TREES,
                                         mtry = best_rf$mtry, nodesize = best_rf$nodesize, importance = TRUE)
  set.seed(SEED)
  final_xgb <- xgboost::xgb.train(
    params = c(base_params, list(max_depth = best_xgb$max_depth, eta = best_xgb$eta)),
    data = xgboost::xgb.DMatrix(xtv, label = tv$phq9),
    nrounds = best_xgb$best_iteration, verbose = 0)
  preds <- list(
    Mean_baseline = rep(mean(tv$phq9), nrow(te)),
    Linear_regression = as.numeric(predict(final_lm, data.frame(xte[, final_cols, drop = FALSE]))),
    Random_Forest = as.numeric(predict(final_rf, xte)),
    XGBoost = as.numeric(predict(final_xgb, xgboost::xgb.DMatrix(xte))))
  # 主结果不截断预测值；避免模型间使用不同的后处理。记录超出0-27范围的比例。
  score <- do.call(rbind, lapply(names(preds), function(n) {
    data.frame(model = n, n_test = nrow(te), as.list(metrics(te$phq9, preds[[n]])),
               outside_0_27 = mean(preds[[n]] < 0 | preds[[n]] > 27), row.names = NULL)
  }))
  write_csv(score, "test_metrics.csv")
  write_csv(data.frame(SEQN = te$SEQN, observed = te$phq9, preds), "test_predictions.csv")
  write_csv(data.frame(term = names(coef(final_lm)), coefficient = unname(coef(final_lm))),
            "prediction_lm_coefficients.csv")
  # RF 为训练过程的 OOB 置换重要性，XGBoost 为分裂 Gain；两者不是因果效应。
  rf_imp <- randomForest::importance(final_rf, type = 1, scale = FALSE)
  write_csv(data.frame(feature = rownames(rf_imp), OOB_permutation_importance = rf_imp[, 1]),
            "rf_importance.csv")
  write_csv(as.data.frame(xgboost::xgb.importance(model = final_xgb)), "xgboost_importance.csv")
  if (SAVE_RESULTS_TO_FILES) {
    saveRDS(list(preprocessor = final_prep, lm_columns = final_cols, lm = final_lm,
                 random_forest = final_rf, parameters = list(rf = best_rf, xgboost = best_xgb)),
            file.path(OUTPUT_DIR, "prediction_models.rds"))
    xgboost::xgb.save(final_xgb, file.path(OUTPUT_DIR, "xgboost_model.json"))
    png(file.path(OUTPUT_DIR, "prediction_comparison.png"), width = 1600, height = 600, res = 120)
  }
  par(mfrow = c(1, 3))
  for (n in names(preds)[-1]) {
    plot(te$phq9, preds[[n]], pch = 16, col = adjustcolor("steelblue", .25),
         xlab = "Observed PHQ-9", ylab = "Predicted PHQ-9", main = n,
         xlim = c(0, 27), ylim = range(c(0, 27, preds[[n]])))
    abline(0, 1, col = "firebrick", lwd = 2)
  }
  if (SAVE_RESULTS_TO_FILES) dev.off()
  if (SAVE_RESULTS_TO_FILES) {
    png(file.path(OUTPUT_DIR, "linear_model_diagnostics.png"), width = 1200, height = 1000, res = 120)
  }
  par(mfrow = c(2, 2))
  plot(final_lm, which = c(1, 2, 3, 5))
  if (SAVE_RESULTS_TO_FILES) dev.off()
  cat("\n==================== FINAL TEST-SET METRICS ====================\n")
  print(score, row.names = FALSE)
  invisible(score)
}

main <- function() {
  setup_packages()
  dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  set.seed(SEED)
  full <- prepare_data()
  analysis <- full[full$age >= 18 & complete.cases(full[, c("phq9", "age", "female")]), ]
  if (nrow(analysis) < 100) stop("有效样本过少，请检查变量和缺失值编码。")
  stopifnot(all(analysis$phq9 >= 0 & analysis$phq9 <= 27))
  write_csv(analysis[, c("SEQN", "phq9", FEATURES, "age_start_smoking", "age_first_drink")],
            "analysis_data.csv")
  message("有效成人 PHQ-9 样本：", nrow(analysis))
  exploratory_analysis(analysis)
  survey_analysis(full)
  predictive_models(analysis)
  save_text(sessionInfo(), "sessionInfo.txt")
  if (SAVE_RESULTS_TO_FILES) {
    message("完成。结果保存到：", OUTPUT_DIR)
  } else {
    message("完成。统计结果已输出到控制台，图形显示在RStudio的Plots窗口。")
  }
}

Sys.unsetenv("NHANES_SKIP_MAIN")
main()
