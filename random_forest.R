##Libraries##
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(ggmosaic)
library(doParallel)
library(discrim) # for naive bayes
library(themis) # for unbalanced data (smote, upsample, downsample)

##Read In Data##
amazon_test <- vroom("test.csv")
amazon_train <- vroom("train.csv")  %>%
  mutate(ACTION = as.factor(ACTION))

balanced_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_smote(all_outcomes(), neighbors = 5)

# Random Forest -----------------------------------------------------------

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# set workflow
forest_workflow <- workflow() %>%
  add_recipe(balanced_recipe) %>%
  add_model(forest_mod)

## Grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

# run Cross validation
CV_results <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

final_forest_workflow <- forest_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

# predict
forest_preds <- predict(final_forest_workflow,
                        new_data = amazon_test,
                        type = "prob")

final_forest_preds <- tibble(id = amazon_test$id,
                             ACTION = forest_preds$.pred_1)

vroom_write(final_forest_preds, "forest_predictions.csv", delim = ",")
