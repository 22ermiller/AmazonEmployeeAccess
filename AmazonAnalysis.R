##Libraries##
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(ggmosaic)

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
         frac = YES/total) %>% View()

# mosaic plot
ggplot(data=amazon_train) + 
  geom_mosaic(data = amazon_train, aes(x=product(ROLE_FAMILY), fill=ACTION))


## Create Recipe

my_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazon_train)


# Logistic Regression -----------------------------------------------------

log_mod <- logistic_reg() %>%
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

write_csv(final_log_preds, "logistic_predictions.csv")

ggplot(data = log_preds) +
  + geom_histogram(aes(x = ACTION)) # Histogram of predictions
