
Sys.setenv(NHANES_SKIP_MAIN = "1")

if (.Platform$OS.type == "windows" && !isTRUE(l10n_info()[["UTF-8"]])) {
  Sys.setlocale("LC_CTYPE", "English_United States.utf8")
}

source(
  "nhanes_analysis.R",
  encoding = "UTF-8"
)

setup_packages()

stopifnot(
  identical(
    valid_values(c(0, 1, 3, 7, 9, NA), 0:3),
    c(0, 1, 3, NA_real_, NA_real_, NA_real_)
  )
)

q <- matrix(
  c(
    rep(0, 9),
    rep(NA_real_, 9),
    c(1, rep(0, 7), NA)
  ),
  nrow = 3,
  byrow = TRUE
)

stopifnot(
  identical(
    rowSums(q, na.rm = FALSE),
    c(0, NA_real_, NA_real_)
  )
)


tr <- data.frame(
  age = c(20, 40, 60),
  female = c(0, 1, 0),
  income_pir = c(1, 3, NA),
  alcohol_group = c("A", "B", "A"),
  smoking_group = c("S", "T", "S")
)

te <- data.frame(
  age = 30,
  female = 1,
  income_pir = NA_real_,
  alcohol_group = "New",
  smoking_group = NA_character_
)

prep <- fit_preprocessor(tr)
x <- transform_features(te, prep)

stopifnot(
  prep$medians["income_pir"] == 2,
  x[1, "income_pir"] == 2,
  x[1, "income_pir_missing"] == 1,
  x[1, "alcohol_group__Unknown"] == 1,
  x[1, "smoking_group__Unknown"] == 1,
  identical(
    colnames(x),
    colnames(transform_features(tr, prep))
  )
)


split <- read.csv(
  "nhanes_depression_r/results/data_split.csv"
)

pred <- read.csv(
  "nhanes_depression_r/results/test_predictions.csv"
)

scores <- read.csv(
  "nhanes_depression_r/results/test_metrics.csv"
)

stopifnot(
  nrow(split) == 5068,
  !anyDuplicated(split$SEQN),
  setequal(
    pred$SEQN,
    split$SEQN[split$split == "test"]
  ),
  all(scores$n_test == nrow(pred))
)

for (i in seq_len(nrow(scores))) {
  got <- metrics(
    pred$observed,
    pred[[scores$model[i]]]
  )
  
  stopifnot(
    max(
      abs(got - unlist(scores[i, names(got)]))
    ) < 1e-10
  )
}

stopifnot(
  !any(
    c(
      "SEQN",
      "phq9",
      sprintf("DPQ%03d", seq(10, 90, 10))
    ) %in% FEATURES
  )
)

cat(
  paste0(
    "PASS: coding, PHQ-9 missingness, ",
    "training-only preprocessing, split isolation, metrics.\n"
  )
)