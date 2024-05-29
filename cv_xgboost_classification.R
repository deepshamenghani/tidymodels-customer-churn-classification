# Loading packages
packages <- c("tidyverse", "tidymodels")

package.check <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, dependencies = TRUE)
    library(x, character.only = TRUE)
  }
})

# Load the dataset
bankchurn_df <- read.csv("./data/bank_churn.csv")

# Glimpse the data
bankchurn_df |> 
  glimpse()

# Data preprocessing
bankchurn_df_upd <- bankchurn_df |> 
  select(Exited, everything()) |> 
  mutate(Exited = factor(Exited, levels = c("1", "0"))) |> 
  select(-contains("Surname"))


# Splitting the data into training and test sets
set.seed(123)
bc_split <- initial_split(bankchurn_df_upd, prop = 3/4, strata = Exited)
train_data <- training(bc_split)
test_data <- testing(bc_split)

# Cross-validation setup
set.seed(123)
bc_folds <- vfold_cv(bankchurn_df_upd, v = 5, strata = Exited)

# Feature engineering
bc_recipe <- recipe(Exited ~ ., data = bankchurn_df_upd) |>
  step_dummy(all_nominal(), -all_outcomes()) |>
  step_zv(all_numeric()) |>
  step_normalize(all_numeric()) |>
  prep()

# XGBoost model specification
xgb_mod <- boost_tree(
  mode = "classification",
  trees = 1000,
  tree_depth = 6,
  learn_rate = 0.01,
  loss_reduction = 0,
  sample_size = 1,
  mtry = 3,
  min_n = 10
) |> 
  set_engine("xgboost")

# XGBoost workflow
xgb_workflow <- workflow() |> 
  add_model(xgb_mod) |> 
  add_recipe(bc_recipe)

# Cross-validation for XGBoost
doParallel::registerDoParallel()
xgb_res <- xgb_workflow |> 
  fit_resamples(resamples = bc_folds, 
                metrics = metric_set(roc_auc, 
                                     accuracy, 
                                     sensitivity, 
                                     specificity), 
                control = control_resamples(save_pred = TRUE))

# Summarize XGBoost results
xgb_res |> 
  collect_metrics()

# Confusion matrix from cross-validation
xgb_res |> 
  conf_mat_resampled()

# ROC curves for XGBoost during cross-validation
xgb_res %>%
  collect_predictions() |> 
  group_by(id) |> 
  roc_curve(Exited, .pred_1) |> 
  autoplot()

# Pull model from cross-validation
# Extract the best model based on ROC AUC
best_params <- xgb_res |> 
  select_best(metric = "roc_auc")

# Finalize the workflow with the best parameters
final_xgb_workflow <- xgb_workflow |> 
  finalize_workflow(best_params)

# Fit the final model on the entire training dataset and evaluate on the test dataset
doParallel::registerDoParallel()
final_fit <- final_xgb_workflow |> 
  last_fit(bc_split)

# Collecting the final predictions
test_predictions <- final_fit |> 
  collect_predictions()

# Collecting the final metrics
final_metrics <- final_fit |> 
  collect_metrics()

# Visualizing the ROC curve for the final model
test_predictions |> 
  roc_curve(truth = Exited, .pred_1) |>
  autoplot()

# Print final metrics
print(final_metrics)

