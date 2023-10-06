##Libraries##
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)

##Read In Data##
amazon_test <- vroom("test.csv")
amazon_train <- vroom("train.csv")  %>%
  mutate(ACTION = as.factor(ACTION))

## EDA



## Create Recipe

my_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazon_train)

