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

## EDA

# box plot
ggplot(amazon_train) +
  geom_boxplot(aes(x = ACTION, y = ROLE_TITLE))

# counts
amazon_train %>%
  group_by(ROLE_FAMILY) %>%
  summarize(total = n(),
         YES = sum(ACTION == 1),
         NO = sum(ACTION == 0),
         frac = YES/total)
# mosaic plot
ggplot(data=amazon_train) + 
  geom_mosaic(data = amazon_train, aes(x=product(ROLE_FAMILY), fill=ACTION))


## Create Recipe

my_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
  #step_dummy(all_nominal_predictors())

pca_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .9)

prep <- prep(pca_recipe)
baked <- bake(prep, new_data = amazon_train)


# Logistic Regression -----------------------------------------------------

log_mod <- logistic_reg() %>% # set logistic regression model
  set_engine("glm")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_mod) %>%
  fit(data = amazon_train)

log_preds <- predict(amazon_workflow,
                     new_data = amazon_test,
                     type = "prob")

final_log_preds <- tibble(id = amazon_test$id,
                          ACTION = log_preds$.pred_1)

vroom_write(final_log_preds, "logistic_predictions.csv", delim = ",")

# Penalized Logistic Regression --------------------------------------------

pen_mod <- logistic_reg(mixture=tune(),
                        penalty = tune()) %>%
  set_engine("glmnet")

# set workflow
pen_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pen_mod)

## Grid of tuning values
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

# run Cross validation
CV_results <- pen_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

final_pen_workflow <- pen_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

# predict
pen_preds <- predict(final_pen_workflow,
                     new_data = amazon_test,
                     type = "prob")

final_pen_preds <- tibble(id = amazon_test$id,
                          ACTION = pen_preds$.pred_1)

vroom_write(final_pen_preds, "penalized_predictions.csv", delim = ",")


# Random Forest -----------------------------------------------------------

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# set workflow
forest_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

## Grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

# Parallel Processing
num_cores <- parallel::detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

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

stopCluster(cl)

vroom_write(final_forest_preds, "forest_predictions.csv", delim = ",")


# Naive Bayes -------------------------------------------------------------

nb_model <- naive_Bayes(Laplace = tune(), 
                        smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_workflow <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(nb_model)

## Tune 

## Grid of tuning values
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

# run Cross validation
CV_results <- nb_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

final_nb_workflow <- nb_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

# predict
nb_preds <- predict(final_nb_workflow,
                        new_data = amazon_test,
                        type = "prob")

final_nb_preds <- tibble(id = amazon_test$id,
                             ACTION = nb_preds$.pred_1)

vroom_write(final_nb_preds, "nb_predictions.csv", delim = ",")



# K-Nearest Neighbors -----------------------------------------------------

knn_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(knn_model)

## Tune 

## Grid of tuning values
tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

# run Cross validation
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

final_knn_workflow <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

# predict
knn_preds <- predict(final_knn_workflow,
                    new_data = amazon_test,
                    type = "prob")

final_knn_preds <- tibble(id = amazon_test$id,
                         ACTION = knn_preds$.pred_1)

vroom_write(final_knn_preds, "knn_predictions.csv", delim = ",")


# Support Vector Machines -------------------------------------------------

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


# Parallel Processing
num_cores <- parallel::detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

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


stopCluster(cl)
