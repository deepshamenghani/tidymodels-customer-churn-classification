# Loading packages

packages <- c("tidyverse", "tidymodels", "skimr", "GGally")

package.check <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, dependencies = TRUE)
    library(x, character.only = TRUE)
  }
})

# Data

bankchurn_df <- read.csv("./data/bank_churn.csv")

# EDA

bankchurn_df |> 
  select(Exited, everything()) |> 
  glimpse()

bankchurn_df |> 
  count(Exited)

bankchurn_df_upd <- bankchurn_df |> 
  select(Exited, everything()) |> 
  mutate(Exited = as.factor(Exited)) |> 
  select(-contains("Surname"))  

bankchurn_df_upd |> 
  skim()

bankchurn_df_upd |> 
  select(Exited, CreditScore, Age, Tenure, Balance) |> 
  ggpairs(mapping = aes(colour = Exited, alpha = 0.3)) +
  scale_fill_manual(values=c('darkgreen', 'red')) +
  scale_colour_manual(values=c('darkgreen', 'red'))

bankchurn_df_upd |> 
  select(Exited, Germany, Spain, France, Male, Female) |> 
  ggpairs(mapping = aes(colour = Exited, alpha = 0.3)) +
  scale_fill_manual(values=c('darkgreen', 'red')) +
  scale_colour_manual(values=c('darkgreen', 'red'))

# Modeling

## Split data

set.seed(123)
bc_split <- initial_split(bankchurn_df_upd, prop = 3/4, strata = "Exited")

train_data <- training(bc_split)
test_data <- testing(bc_split)

## Feature engineering    
  
bc_recipe <- recipe(Exited ~ ., data = bankchurn_df_upd) |>
  step_dummy(all_nominal(), -all_outcomes()) |>
  step_zv(all_numeric()) |>
  step_normalize(all_numeric()) |>
  prep()

bc_recipe |> 
  bake(new_data = NULL)  
  
## Model specification and workflow

lr_mod <- logistic_reg() |> 
  set_mode("classification") |> 
  set_engine("glm")

lr_mod

lr_workflow <- 
  workflow() |> 
  add_model(lr_mod) |> 
  add_recipe(bc_recipe)

lr_workflow

rand_forest_ranger_model <- rand_forest(
  mode = "classification", mtry = 10, trees = 500, min_n = 20) |>
  set_engine("ranger", importance = "impurity") 

rand_forest_ranger_model

rf_workflow <- workflow() |> 
  add_model(rand_forest_ranger_model) |> 
  add_recipe(bc_recipe)

rf_workflow


## Fit the data
  
lr_fit <- 
  lr_workflow |> 
  fit(data = train_data)

rf_fit <- 
  rf_workflow |>
  fit(data = train_data)

## Feature importance

lr_fit |> 
  extract_fit_parsnip() |> 
  tidy() |> 
  arrange(p.value)

extract_fit_parsnip(rf_fit)$fit |> 
  ranger::importance() |> 
  enframe() |> 
  arrange(desc(value)) 

## Predict on test data

class_pred_lr <- predict(lr_fit, test_data)

prob_pred_lr <- predict(lr_fit, test_data, type = "prob")

lr_preds_combined <- 
  data.frame(class_pred_lr, prob_pred_lr) |> 
  select(class = .pred_class, prob_no = .pred_0, prob_yes = .pred_1) |> 
  bind_cols(test_data)

class_pred_rf <- predict(rf_fit, test_data)

prob_pred_rf <- predict(rf_fit, test_data, type = "prob")

rf_preds_combined <- 
  data.frame(class_pred_rf, prob_pred_rf) |> 
  select(class = .pred_class, prob_no = .pred_0, prob_yes = .pred_1) |> 
  bind_cols(test_data)


## ROC plot

lr_roc <- lr_preds_combined |> 
  roc_curve(truth = Exited, prob_no) |> 
  mutate(model = "Linear model") 

rf_roc <- rf_preds_combined |> 
  roc_curve(truth = Exited, prob_no) |> 
  mutate(model = "Random forest")

lr_roc |> 
  bind_rows(rf_roc) |> 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line() +
  geom_abline(lty = 2) +
  labs(y = "True Positive Rate", 
       x = "False Positive Rate",
       title = "ROC curve") +
  theme_bw()

### Confusion Matrix 

caret::confusionMatrix(lr_preds_combined$Exited,
                                lr_preds_combined$class,
                                positive = "1")

caret::confusionMatrix(rf_preds_combined$Exited,
                                rf_preds_combined$class,
                                positive = "1")

