##Libraries##
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(ggmosaic)
library(doParallel)
library(discrim) # for naive bayes

##Read In Data##
amazon_test <- vroom("test.csv")
amazon_train <- vroom("train.csv")  %>%
  mutate(ACTION = as.factor(ACTION))

pca_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .9)


# Polynomial SVM
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Radial SVM
svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Linear SVM
svmLinear <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_workflow <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(svmLinear)

## Tune 

## Grid of tuning values
tuning_grid <- grid_regular(cost(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

# run Cross validation
CV_results <- svm_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

final_svm_workflow <- svm_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

# predict
svm_preds <- predict(final_svm_workflow,
                     new_data = amazon_test,
                     type = "prob")

final_svm_preds <- tibble(id = amazon_test$id,
                          ACTION = svm_preds$.pred_1)

vroom_write(final_svm_preds, "svm_predictions.csv", delim = ",")
